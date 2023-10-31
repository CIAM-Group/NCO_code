import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model_sl import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat[0], opts.device), move_to(bat[1], opts.device)['tour'].long())
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


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
    # training_dataset = baseline.wrap_dataset(problem.make_sl_dataset(
    #     size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))

    training_dataset = problem.make_sl_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
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

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
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
    # x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(batch[0], opts.device)
    sol = move_to(batch[1], opts.device)
    if opts.data_augmentation:
        coor = torch.cat([x['depot'].unsqueeze(1), x['loc']], dim=1)
        coor = data_aug(coor, opts.device)
        x['depot'] = coor[:, 0]
        x['loc'] = coor[:, 1:]

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood1 = model(x, sol['tour'].long())

    # Calculate loss
    cost, log_likelihood2 = model(x, sol['tour_r'].long())
    log_likelihood = torch.stack([log_likelihood1, log_likelihood2], dim=0).max(0)[0]

    # Perform backward pass and optimization step
    loss = -log_likelihood.mean()  # crossentropy
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, loss, 0, tb_logger, opts)


def data_aug(problems, device):
    # only rotation and flip
    batch_size = problems.shape[0]

    problems = problems - 0.5
    theta = torch.atan2(problems[:, :, 1], problems[:, :, 0])
    rho = torch.linalg.norm(problems, dim=2)
    theta = move_to(theta, device)
    rho = move_to(rho, device)

    rotation = torch.rand(batch_size) * 2 * math.pi
    rotation = move_to(rotation, device)
    # rotation
    theta = theta + rotation.unsqueeze(-1).expand_as(theta)

    # flip
    symmetry = torch.rand(batch_size).unsqueeze(-1).expand_as(theta) > 0.5
    symmetry = move_to(symmetry, device)
    theta[symmetry] = -theta[symmetry]
    x = rho * torch.cos(theta) + 0.5
    y = rho * torch.sin(theta) + 0.5
    problems = torch.stack([x, y], dim=-1)
    return problems