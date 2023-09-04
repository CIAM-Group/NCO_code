import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

from pymoo.factory import get_performance_indicator
import numpy as np


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, val_dataset, opts):
    # Validate
    print('Validating...')
    hv_fn = get_performance_indicator("hv", ref_point=np.array([opts.graph_size for _ in range(opts.num_objs)]))

    cost, all_objs_list = rollout(model, val_dataset, opts)
    print('Time: {:.3f}'.format(time.time() - opts.start_time))

    # HV
    all_objs = torch.stack(all_objs_list, dim=2)

    reference_point = opts.reference_point * torch.ones_like(all_objs, device=all_objs.device)
    all_objs_rep = all_objs.unsqueeze(2).expand((-1, -1, opts.num_weights, -1))
    all_objs_org = all_objs.unsqueeze(1).expand((-1, opts.num_weights, -1, -1))
    dominated = (all_objs_rep - all_objs_org > 0).all(3)

    nondominated = ~(dominated.any(2))
    mask = nondominated.unsqueeze(2).expand_as(all_objs)

    NDS = torch.where(mask, all_objs, reference_point)
    hv_list = []
    for i in range(NDS.shape[0]):
        hv = hv_fn.do(NDS[i].cpu().numpy())
        hv_list.append(hv)

    all_hv = torch.from_numpy(np.stack(hv_list, axis=0))
    print('{:.3f} +- {:.3f}'.format(all_hv.mean().item(), torch.std(all_hv).item()))
    return cost, all_objs_list, NDS, all_hv, None


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")

    def eval_model_bat(bat, model):
        with torch.no_grad():
            cost, _, all_objs, _ = model(
                move_to(bat, opts.device),
                opts.w_list,
                num_objs=opts.num_objs,
                mix_objs=opts.mix_objs
            )
        return cost.data.cpu(), all_objs

    cost_list = []
    obj_list = []
    for o in range(opts.num_objs):
        obj_list.append([])
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
        cost, all_objs = eval_model_bat(bat, model)
        cost_list.append(cost)
        for o in range(opts.num_objs):
            obj_list[o].append(all_objs[o])
    return torch.cat(cost_list, 0), [torch.cat(obj_list[o], 0) for o in range(opts.num_objs)]


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """

    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]

    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.epoch_size,
        distribution=opts.data_distribution,
        correlation=opts.correlation,
        num_objs=opts.num_objs,
        mix_objs=opts.mix_objs
    ))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward, all_objs, NDS, HV, num_NDS = validate(model, val_dataset, opts)
    if not opts.no_tensorboard:
        for i in range(opts.num_weights):
            opts.logger_list[i].log_value('val_avg_reward', avg_reward[:, i].mean().item(), step)
            for j in range(opts.num_objs):
                opts.logger_list[i].log_value('val_dist{}'.format(j), all_objs[j][:, i].mean().item(), step)
        tb_logger.log_value('HV', HV.mean().item(), step)
        print('Epoch{} HV: '.format(epoch), HV.mean().item())

    if lr_scheduler.get_last_lr()[0] > opts.lr_critic:
        lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    set_decode_type(model, "sampling")
    x = move_to(batch, opts.device)
    cost, log_likelihood, all_dists, coef = model(x, opts.w_list, num_objs=opts.num_objs, mix_objs=opts.mix_objs)
    log_likelihood = log_likelihood.reshape(-1, opts.num_weights)

    obj_tensor = torch.stack(all_dists, dim=2).unsqueeze(1).expand(-1, opts.num_weights, -1, -1)
    if torch.cuda.device_count() > 1:
        w_tensor = torch.stack(opts.w_list, dim=0)[:, :opts.num_objs].unsqueeze(0).unsqueeze(2).expand_as(obj_tensor).to(obj_tensor.device)
    else:
        w_tensor = torch.stack(opts.w_list, dim=0).unsqueeze(0).unsqueeze(2).expand_as(obj_tensor).to(obj_tensor.device)

    score = (w_tensor * obj_tensor).sum(-1).sort(-1)[0]
    reinforce_loss = ((cost - score[:, :, :opts.num_top].mean(-1)) * log_likelihood)
    loss = reinforce_loss
    optimizer.zero_grad()
    loss.mean().backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, loss, reinforce_loss, all_dists, coef, tb_logger, opts)

