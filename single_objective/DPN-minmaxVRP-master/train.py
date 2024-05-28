import os
import time
from tqdm import tqdm
import torch
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools  # for permutation list
import torch

from matplotlib import cm
from torchvision import models

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from utils.problem_augment import augment
import random


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost_file = open(os.path.join(opts.save_dir, 'gap.txt'), mode='a+')
    # for i in [1, 2, 3, 5, 7, 10, 20]:
    for i in opts.agent_list:
        print('Validating...,with' + str(i) + 'agents\n')
        model.agent_num = i
        model.decay = 0
        model.depot_num = opts.depot_eval
        model.embedder.agent_num = i
        cost = rollout(model, dataset, i, opts)
        avg_cost = cost.mean()
        print('Validation overall avg_cost: {} +- {}\n'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))
        cost_file.write(str(avg_cost.item()) + ' ')
    cost_file.write('\n')

    return avg_cost


def validate2(model, dataset, opts):
    # Validate
    print('Validating...')
    cost_file = open(os.path.join(opts.save_dir, 'gap2.txt'), mode='a+')
    # for i in [1, 2, 3, 5, 7, 10, 20]:
    for i in [3, 5, 7]:
        print('Validating...,with' + str(i) + 'agents\n')
        model.agent_num = i
        model.embedder.agent_num = i
        model.decay = 0
        model.depot_num = 8
        cost = rollout(model, dataset, i, opts)
        # print(cost.shape)
        avg_cost = cost.mean()
        print('Validation overall avg_cost: {} +- {}\n'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))
        cost_file.write(str(avg_cost.item()) + ' ')
    cost_file.write('\n')

    return avg_cost


def rollout(model, dataset, i, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat, batch_size, agt, aug=8):
        with torch.no_grad():
            agent_per = torch.arange(agt).cuda()[None, :].expand(opts.r_eval, -1)
            if (opts.r_eval > 1):
                for i in range(100):
                    a = torch.randint(0, agt, (opts.r_eval,)).cuda()
                    b = torch.randint(0, agt, (opts.r_eval,)).cuda()
                    p = agent_per[torch.arange(opts.r_eval), a].clone()
                    q = agent_per[torch.arange(opts.r_eval), b].clone()
                    agent_per = agent_per.scatter(dim=1, index=b[:, None], src=p[:, None])
                    agent_per = agent_per.scatter(dim=1, index=a[:, None], src=q[:, None])
                agent_per[0] = torch.arange(agt).cuda()
            model.agent_per = agent_per
            cost, _, route = model(move_to(bat, opts.device), return_pi=True)
            cost, _ = cost.min(-1)
            # route = route.view(aug * batch_size, opts.r_eval, -1).gather(1, _[:, None, None].expand(-1, -1, route.size(-1)))
            cost, _ = cost.view(aug, -1).min(0, keepdim=True)
            # code related to printing a solution out
            '''
            print(cost)
            route = route.view(aug, batch_size, -1).gather(0, _[:, :, None].expand(-1, -1, route.size(-1)))[0]
            cost = cost.transpose(0, 1)
            plt.rcParams['pdf.use14corefonts'] = True
            # plt.rcParams['pdf.fonttype'] = 42
            # plt.rcParams['ps.fonttype'] = 42
            colorboard = np.array(sns.color_palette("deep", 100))
            k = 0
            if opts.problem == 'mtsp':
                bat2 = bat[k, ...]
                x = bat[k, ...]
                plt.text(bat2[0, 0], bat2[0, 1], 'Depot1', fontsize=10)
                route = route[k] - agt + 1
            elif opts.problem == 'mpdp':
                bat2 = torch.cat((bat['depot'][k, ...], bat['loc'][k, ...]), dim=0)
                x = torch.cat((bat['depot'][k, ...], bat['loc'][k, ...]), dim=0)
                plt.text(bat2[0, 0], bat2[0, 1], 'Depot1', fontsize=10)
                route = route[k] - agt + 1
            elif opts.problem == 'mdvrp' or opts.problem == 'fmdvrp':
                bat2 = torch.cat((bat['depot'][k, :model.depot_num, :], bat['loc'][k, ...]), dim=0)
                x = torch.cat((bat['depot'][k, :model.depot_num, :], bat['loc'][k, ...]), dim=0)
                for i in range(model.depot_num):
                    plt.text(bat['depot'][k, i, 0], bat['depot'][k, i, 1], 'Depot' + str(i + 1), fontsize=10)
                route = route[k]
            route[route <= 0] = 0
            x = x.gather(0, route.long().unsqueeze(-1).expand(-1, 2).cpu())
            lengths = []
            leh = []
            lehlabel = []
            a = 0
            max = 0
            color = 0
            dist = torch.cdist(x, x, p=2)
            if opts.problem == 'mtsp' or opts.problem == 'mdvrp' or opts.problem == 'fmdvrp':
                for node in bat2:
                    plt.scatter(x=node[0], y=node[1], c='gray', marker='o', s=10)
            elif opts.problem == 'mpdp':
                plt.scatter(x=bat2[0, 0], y=bat2[0, 1], c='gray', marker='o', s=10)
                for i in range(1, bat2.size(0) // 2 + 1):
                    plt.scatter(x=bat2[i][0], y=bat2[i][1], c='b', marker='D', s=20)
                for i in range(bat2.size(0) // 2 + 1, bat2.size(0)):
                    plt.scatter(x=bat2[i][0], y=bat2[i][1], c='r', marker='*', s=30)
            my_colors = colorboard[color]

            if opts.problem == 'mtsp' or opts.problem == 'mpdp':
                plt.plot([x[0, 0], x[-1, 0]], [x[0, 1], x[-1, 1]], c=my_colors)
            for i in range(1, x.size(0)):
                my_colors = colorboard[color]
                if opts.problem == 'mdvrp' or opts.problem == 'fmdvrp':
                    if route[i] < model.depot_num and route[i - 1] >= model.depot_num:
                        plt.scatter(x=x[i, 0], y=x[i, 1], c='r', marker='o', s=20)
                        lastone, = plt.plot([x[i, 0], x[i - 1, 0]], [x[i, 1], x[i - 1, 1]], c=my_colors)
                        lehlabel.append('Route' + str(color + 1))
                        leh.append(lastone)
                    elif route[i] < model.depot_num and route[i - 1] < model.depot_num:
                        color += 1
                        if max < a:
                            max = a
                        lengths.append(a)
                        a = 0
                    else:
                        a += dist[i, i - 1]
                        plt.plot([x[i, 0], x[i - 1, 0]], [x[i, 1], x[i - 1, 1]], c=my_colors)
                else:
                    a += dist[i, i - 1]
                    if route[i] == 0:
                        color += 1
                        if max < a:
                            max = a
                        lengths.append(a)
                        a = 0
                        lastone, = plt.plot([x[i, 0], x[i - 1, 0]], [x[i, 1], x[i - 1, 1]], c=my_colors)
                        lehlabel.append('Route' + str(color))
                        leh.append(lastone)
                    else:
                        plt.plot([x[i, 0], x[i - 1, 0]], [x[i, 1], x[i - 1, 1]], c=my_colors)
            if opts.problem == 'mdvrp' or opts.problem == 'fmdvrp':
                if max < a:
                    max = a
                lengths.append(a)
            a = '{'
            for i in range(agt):
                a = a + str(round(lengths[i].item(), 3))
                if i < agt - 1:
                    a += ','
                else:
                    a += '}'
            plt.title('DPN Routes=' + a + ', Obj.=' + str(round(max.item(), 5)), fontsize=10, font='Times New Roman')
            plt.legend(handles=leh, labels=lehlabel, fontsize=10, prop={'family': 'Times New Roman', 'size': 10})
            plt.xlabel('X', {'family': 'Times New Roman', 'size': 10})
            plt.ylabel('Y', {'family': 'Times New Roman', 'size': 10})
            plt.xlim(-0.02,1.02)
            plt.ylim(-0.02,1.02)
            plt.xticks(font='Times new roman')
            plt.yticks(font='Times new roman')
            plt.show()
            '''

        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(augment(bat, opts.aug_eval), batch_size=opts.eval_batch_size, agt=i, aug=opts.aug_eval)
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


def train_epoch(model, optimizer, lr_scheduler, epoch, val_dataset, val_dataset2, problem, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    graph_size = opts.graph_size

    training_dataset = problem.make_dataset(
        size=graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    # model = get_inner_model(model)
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        agent_num = random.sample(range(opts.agent_min, opts.agent_max + 1), 1)[0]
        if opts.problem == 'mdvrp' or opts.problem == 'fmdvrp':
            depot_num = random.sample(range(opts.depot_min, opts.depot_max + 1), 1)[0]
            model.depot_num = depot_num
            model.embedder.depot_num = depot_num
        model.agent_num = agent_num
        model.embedder.agent_num = agent_num
        if epoch < 0:
            model.decay = 0.2
        else:
            model.decay = 0

        cost = train_batch(
            model,
            optimizer,
            agent_num,
            batch,
            opts
        )
        if opts.problem == 'mdvrp' or opts.problem == 'fmdvrp':
            if batch_id % 10 == 0:
                print('current cost:' + str(cost.item()) + ' ' + str(depot_num) + ' ' + str(agent_num))
                cost_file = open(os.path.join(opts.save_dir, 'curve.txt'), mode='a+')
                cost_file.write(str(cost.item()) + ' ' + str(depot_num) + ' ' + str(agent_num) + '\n')
        if opts.problem == 'mtsp' or opts.problem == 'mpdp':
            if batch_id % 10 == 0:
                print('current cost:' + str(cost.item()) + ' ' + str(agent_num))
                cost_file = open(os.path.join(opts.save_dir, 'curve.txt'), mode='a+')
                cost_file.write(str(cost.item()) + ' ' + str(agent_num) + '\n')

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    epoch_duration = time.time() - start_time

    step += 1

    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    lr_scheduler.step()

    validate(model, val_dataset, opts)
    validate2(model, val_dataset2, opts)


def train_batch(
        model,
        optimizer,
        agent_num,
        batch,
        opts
):
    x = move_to(batch, opts.device)
    agent_per = torch.arange(agent_num).cuda()[None, :].expand(opts.pomo_size, -1)
    if (opts.pomo_size > 1):
        for i in range(100):
            a = torch.randint(0, agent_num, (opts.pomo_size,)).cuda()
            b = torch.randint(0, agent_num, (opts.pomo_size,)).cuda()
            p = agent_per[torch.arange(opts.pomo_size), a].clone()
            q = agent_per[torch.arange(opts.pomo_size), b].clone()
            agent_per = agent_per.scatter(dim=1, index=b[:, None], src=p[:, None])
            agent_per = agent_per.scatter(dim=1, index=a[:, None], src=q[:, None])
    # Evaluate model, get costs and log probabilities
    x_aug = augment(x, opts.N_aug)
    model.agent_per = agent_per
    if opts.subloss == 1:
        makespan, partspan, cost_route, log_likelihood = model(x_aug, subloss=True)
    if opts.subloss == 0:
        makespan, partspan, cost_route, log_likelihood = model(x_aug, subloss=False)
    log_likelihood = log_likelihood.view(opts.N_aug, -1, opts.pomo_size, log_likelihood.size(-1)).permute(1, 0, 2, 3)
    makespan = makespan.view(opts.N_aug, -1, opts.pomo_size).permute(1, 0, 2)
    ll = log_likelihood.sum(-1)
    advantage_makespan = (makespan - makespan.mean(dim=1).mean(dim=-1)[:, None, None])
    loss = ((advantage_makespan) * ll).mean()
    if opts.subloss == 1:
        # Calculate loss
        partspan = partspan.view(opts.N_aug, -1, opts.pomo_size).permute(1, 0, 2)
        advantage_part = (partspan - partspan.mean(dim=1).mean(dim=-1)[:, None, None])
        loss += 0.1 * ((advantage_part) * ll).mean()
        cost_route = cost_route.view(opts.N_aug, -1, opts.pomo_size, cost_route.size(-1)).permute(1, 0, 2, 3)
        loss_route = (cost_route * log_likelihood).mean()
        loss += 0.1 * loss_route
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cost_mean = makespan.mean().view(-1, 1)

    return cost_mean
