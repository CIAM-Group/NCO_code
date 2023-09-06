import os
import random
import time
import torch
from queue import Queue
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import move_to, clip_grad_norms, get_inner_model
from utils.logger import log_to_screen, log_to_tb_train, log_to_tb_val


class Mean_vector:
    # 对m维空间，目标方向个数H
    def __init__(self, H=5, m=3):
        self.H = H
        self.m = m
        self.stepsize = 1 / H

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
        # 生成权均匀向量
        H = self.H
        m = self.m
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws


def eval_only(problem, model, tb_logger, opts):
    checkpoint = torch.load(opts.checkpoint_inference, map_location=opts.device)
    model.load_state_dict(checkpoint['model'])
    batch = np.load(opts.dataset_inference)[0:opts.batch_size, ...]
    batch = torch.from_numpy(batch).type(torch.float32)
    validate(problem, batch, model, None, tb_logger, opts)
    # validate_attach(problem, batch, model, sel_model, None, tb_logger, opts)


def validate(problem, batch, model, val_dataset, tb_logger, opts, _id=None):
    start_time = time.time()
    model.eval()
    if problem.NAME == 'tsp':
        x = batch
    else:
        assert False, "Unsupported problem: {}".format(problem.NAME)
    x_input = move_to(x, opts.device)  # batch_size, graph_size, 2
    batch = move_to(batch, opts.device)  # batch_size, graph_size, 2
    pareto = [[0] for i in range(batch.size(0))]
    vec_best = None

    '''
    solution = np.load('datasets/kroab_100_solution.npy')[0:10,...]
    solution = torch.from_numpy(solution).view(10, 101, -1)
    f = problem.get_costs(batch, solution[:, 0, :]).unsqueeze(1)
    for i in range(1, solution.size(1)):
        cost = problem.get_costs(batch, solution[:, i, :])
        f = torch.cat((f, cost.unsqueeze(1)), dim=1)
    '''
    if opts.init_type == 'greedy':
        n_vec = torch.zeros(2, dtype=torch.float32, device=opts.device)
        n_vec[1] = 1 - n_vec[0]
        solution, f = problem.greddy(x_input, n_vec, opts.device)
        n_vec = torch.ones(2, dtype=torch.float32, device=opts.device)
        n_vec[1] = 1 - n_vec[0]
        _, __ = problem.greddy(x_input, n_vec, opts.device)
        solution = torch.cat((solution, _), dim=1)
        f = torch.cat((f, __), dim=1)
        for i in range(2, opts.population_size):
            n_vec = torch.rand(opts.num_obj)
            n_vec[1] = 1 - n_vec[0]
            _, __ = problem.greddy(x_input, n_vec, opts.device)
            solution = torch.cat((solution, _), dim=1)
            f = torch.cat((f, __), dim=1)
    else:
        sol_now = move_to(problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
        initial_cost = problem.get_costs(batch, sol_now)
        solution = sol_now.unsqueeze(1)
        f = initial_cost.unsqueeze(1)
        for i in range(1, opts.population_size):
            sol_now = move_to(problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
            cost = problem.get_costs(batch, sol_now)
            solution = torch.cat((sol_now.unsqueeze(1), solution), dim=1)
            f = torch.cat((cost.unsqueeze(1), f), dim=1)

    hv, pareto, pa = problem.calhv(solution.size(-1), f)
    print(hv.mean(0), hv.std(0))
    np.save('datasets\\kroab100_sy_greedy.npy', f.numpy())
    mv = Mean_vector(99, 2)
    v = move_to(torch.tensor(mv.get_mean_vectors()), opts.device)
    inp = torch.zeros((f.size(0), f.size(1)), dtype=torch.float32, device=opts.device)
    n_step = opts.n_step
    T = opts.val_T
    t = 0
    exchange = None
    vec_sel = None
    pa = torch.zeros((f.size(0), f.size(1)), dtype=torch.float32, device=opts.device)
    batchv = v.t().unsqueeze(0).expand((batch.size(0), -1, -1))
    while t < T:
        inp = pa
        p = torch.matmul(f, batchv)
        tag = p.min(1)[1]
        inp[torch.arange(opts.batch_size)[:, None], tag] = 1
        t_s = t
        vec_sel = move_to(torch.randint(opts.vector_init_num, (opts.batch_size, 1)).squeeze(-1), opts.device)
        v_sel = torch.gather(v, dim=0, index=vec_sel.unsqueeze(-1).expand(-1, v.size(-1)))
        p = (f * v_sel.unsqueeze(1).expand(-1, f.size(1), -1)).sum(-1)
        f_sel = p
        _ = f_sel.min(-1)[1]
        f_sel[inp > 0] = -1 * np.inf
        _place = f_sel.max(-1)[1]
        full = (inp.sum(-1) >= min(opts.vector_init_num, opts.population_size)).bool()
        _place[full] = _[full]
        span = _.unsqueeze(-1).unsqueeze(-1)
        span2 = _place.unsqueeze(-1).unsqueeze(-1)
        f_0 = torch.gather(f, dim=1, index=span.expand(-1, -1, f.size(-1))).squeeze(1)
        f_1 = torch.gather(f, dim=1, index=span2.expand(-1, -1, f.size(-1))).squeeze(1)
        s_0 = torch.gather(solution, dim=1, index=span.expand(-1, -1, solution.size(-1))).squeeze(1)
        s_1 = torch.gather(solution, dim=1, index=span2.expand(-1, -1, solution.size(-1))).squeeze(1)
        select = s_0.clone()
        f_select = f_0.clone()
        select_best = s_0.clone()
        f_select_best = f_0.clone()
        while t - t_s < n_step and not (t == T):
            exchange, log_lh = model(x_input, select, exchange, v_sel, do_sample=True)
            select = problem.opt2(select, exchange)
            select = move_to(select, opts.device)
            cost = problem.get_costs(batch, select)
            # f_select = cost
            p = (cost * v_sel).sum(dim=-1)
            best_for_now = torch.cat(((f_select_best * v_sel).sum(dim=-1)[None, :], p[None, :]), 0).min(0)[0]
            r = (f_select_best * v_sel).sum(dim=-1) - best_for_now
            select_best[(r > 0)] = select[(r > 0)]
            f_select_best[(r > 0)] = cost[(r > 0)]
            t = t + 1
        f_select = cost
        select[full] = select_best[full]
        full_f = full[:, None].expand_as(f_select)
        f_select[full_f] = f_select_best[full_f]
        f_select[(f_select == f_0)[:, 0]] = f_1[(f_select == f_0)[:, 0]]
        select[(f_select == f_0)[:, 0]] = s_1[(f_select == f_0)[:, 0]]
        # Get discounted R
        f = torch.scatter(input=f, dim=1, index=span2.expand(-1, -1, f.size(-1)), src=f_select.unsqueeze(1))
        solution = torch.scatter(input=solution, dim=1, index=span2.expand(-1, -1, solution.size(-1)), src=select.unsqueeze(1))
        if ((t - 4) / n_step) % 400 == 0:
            hv, pareto, pa = problem.calhv(solution.size(-1), f)
            print("now:", t, ":", hv.mean(0), hv.std(0), full.float().mean())
            vec_file = open('./stdr_random.txt', mode='a+')
            vec_file.write(str(hv.std().item()) + '\n')
            hv_file = open('./hvr_random.txt', mode='a+')
            hv_file.write(str(hv.mean().item()) + '\n')
            print(t)
            print(time.time() - start_time)
    print(hv.mean(dim=0), f.size(1))
    np.save(opts.solution_inference, f.cpu().numpy())


def validate_attach(problem, batch, model, val_dataset, tb_logger, opts, _id=None): # another way to validate by disabling mutation between weight vectors
    start_time = time.time()
    model.eval()
    if problem.NAME == 'tsp':
        x = batch
    else:
        assert False, "Unsupported problem: {}".format(problem.NAME)
    x_input = move_to(x, opts.device)  # batch_size, graph_size, 2
    batch = move_to(batch, opts.device)  # batch_size, graph_size, 2
    pareto = [[0] for i in range(batch.size(0))]
    vec_best = None
    '''
    solution = np.load('datasets/kroab_100_solution.npy')[0:10,...]
    solution = torch.from_numpy(solution).view(10, 101, -1)
    f = problem.get_costs(batch, solution[:, 0, :]).unsqueeze(1)
    for i in range(1, solution.size(1)):
        cost = problem.get_costs(batch, solution[:, i, :])
        f = torch.cat((f, cost.unsqueeze(1)), dim=1)
    '''
    if opts.init_type == 'greedy':
        n_vec = torch.zeros(2, dtype=torch.float32, device=opts.device)
        n_vec[1] = 1 - n_vec[0]
        solution, f = problem.greddy(x_input, n_vec, opts.device)
        n_vec = torch.ones(2, dtype=torch.float32, device=opts.device)
        n_vec[1] = 1 - n_vec[0]
        _, __ = problem.greddy(x_input, n_vec, opts.device)
        solution = torch.cat((solution, _), dim=1)
        f = torch.cat((f, __), dim=1)
        for i in range(2, opts.population_size):
            n_vec = torch.rand(opts.num_obj)
            n_vec[1] = 1 - n_vec[0]
            _, __ = problem.greddy(x_input, n_vec, opts.device)
            solution = torch.cat((solution, _), dim=1)
            f = torch.cat((f, __), dim=1)
    else:
        sol_now = move_to(problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
        initial_cost = problem.get_costs(batch, sol_now)
        solution = sol_now.unsqueeze(1)
        f = initial_cost.unsqueeze(1)
        for i in range(1, opts.population_size):
            sol_now = move_to(problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
            cost = problem.get_costs(batch, sol_now)
            solution = torch.cat((sol_now.unsqueeze(1), solution), dim=1)
            f = torch.cat((cost.unsqueeze(1), f), dim=1)

    hv, pareto, pa = problem.calhv(solution.size(-1), f)
    print(hv.mean(0), hv.std(0))
    # np.save('datasets\\tsp200_greedy.npy', f.numpy())
    mv = Mean_vector(99, 2)
    v = move_to(torch.tensor(mv.get_mean_vectors()), opts.device)
    n_step = opts.n_step
    T = opts.val_T
    t = 0
    exchange = None
    while t < T:

        t_s = t
        vec_sel = move_to(torch.randint(opts.vector_init_num, (opts.batch_size, 1)).squeeze(-1), opts.device)
        v_sel = torch.gather(v, dim=0, index=vec_sel.unsqueeze(-1).expand(-1, v.size(-1)))
        _ = vec_sel
        _place = vec_sel
        span = _.unsqueeze(-1).unsqueeze(-1)
        span2 = _place.unsqueeze(-1).unsqueeze(-1)
        f_0 = torch.gather(f, dim=1, index=span.expand(-1, -1, f.size(-1))).squeeze(1)
        s_0 = torch.gather(solution, dim=1, index=span.expand(-1, -1, solution.size(-1))).squeeze(1)
        select = s_0.clone()
        f_select = f_0.clone()
        select_best = s_0.clone()
        f_select_best = f_0.clone()
        while t - t_s < n_step and not (t == T):
            exchange, log_lh = model(x_input, select, exchange, v_sel, do_sample=True)
            select = problem.opt2(select, exchange)
            select = move_to(select, opts.device)
            cost = problem.get_costs(batch, select)
            # f_select = cost
            p = (cost * v_sel).sum(dim=-1)
            best_for_now = torch.cat(((f_select_best * v_sel).sum(dim=-1)[None, :], p[None, :]), 0).min(0)[0]
            r = (f_select_best * v_sel).sum(dim=-1) - best_for_now
            select_best[(r > 0)] = select[(r > 0)]
            f_select_best[(r > 0)] = cost[(r > 0)]
            t = t + 1
        select = select_best
        f_select = f_select_best
        # f_select = cost
        # Get discounted R
        f = torch.scatter(input=f, dim=1, index=span2.expand(-1, -1, f.size(-1)), src=f_select.unsqueeze(1))
        solution = torch.scatter(input=solution, dim=1, index=span2.expand(-1, -1, solution.size(-1)), src=select.unsqueeze(1))
        if ((t - 4) / n_step) % 400 == 0:
            hv, pareto, pa = problem.calhv(solution.size(-1), f)
            print("now:", t, ":", hv.mean(0), hv.std(0))
            vec_file = open('./stdr_random.txt', mode='a+')
            vec_file.write(str(hv.std().item()) + '\n')
            hv_file = open('./hvr_random.txt', mode='a+')
            hv_file.write(str(hv.mean().item()) + '\n')
            print(t)
            print(time.time() - start_time)
    print(hv.mean(dim=0), f.size(1))
    np.save(opts.solution_inference, f.cpu().numpy())


def train_epoch(problem, model, optimizer, baseline, lr_scheduler, epoch, rollout, tb_logger, opts):
    # lr_scheduler
    lr_scheduler.step(epoch)
    step = epoch * (opts.epoch_size // (opts.num_obj * opts.batch_size))
    print('\n\n')
    print("|", format(f" Training epoch {epoch} ", "*^60"), "|")
    print("Training with lr={:.3e} for run {}".format(optimizer.param_groups[0]['lr'], opts.run_name), flush=True)
    # if epoch % 10 == 0:
    # checkpoint = torch.load('pre.pt', map_location=opts.device)
    # sel_model.load_state_dict(checkpoint['selmodel'])
    # sel_baseline.load_state_dict(checkpoint['selbaseline'])
    # Put model in tra 2in mode!
    if opts.eval_only == True:
        eval_only(problem, model, None, opts)
    # start training
    pbar = tqdm(total=(opts.epoch_size // (opts.batch_size)) * (opts.T_train // opts.n_step),
                disable=opts.no_progress_bar, desc=f'training',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    for i in range(opts.epoch_size // (opts.batch_size)):
        batch = torch.rand((opts.batch_size, opts.num_obj, opts.graph_size, 2), dtype=torch.float32, device=opts.device)
        model.train()
        rollout = train_batch(problem, model, optimizer, baseline, epoch, rollout, step, batch, tb_logger, opts, pbar)
        step += 1
    pbar.close()

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'baseline': get_inner_model(baseline).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    return rollout


def train_batch(problem, model, optimizer, baseline, epoch, rollout, step, batch, tb_logger, opts, pbar):
    train_sel = False
    rollout_sel = opts.rollout_enable
    if problem.NAME == 'tsp':
        x = batch
    else:
        assert False, "Unsupported problem: {}".format(problem.NAME)
    x_input = move_to(x, opts.device)  # batch_size, graph_size, 2
    batch = move_to(batch, opts.device)  # batch_size, graph_size, 2
    pareto = [[0] for i in range(batch.size(0))]
    vec_best = None
    sol_now = move_to(problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
    initial_cost = problem.get_costs(batch, sol_now)
    solution = sol_now.unsqueeze(1)
    f = initial_cost.unsqueeze(1)
    for i in range(1, opts.stomatic_init_num):
        sol_now = move_to(problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
        cost = problem.get_costs(batch, sol_now)
        solution = torch.cat((sol_now.unsqueeze(1), solution), dim=1)
        f = torch.cat((cost.unsqueeze(1), f), dim=1)

    mv = Mean_vector(99, 2)
    v = move_to(torch.tensor(mv.get_mean_vectors()), opts.device)
    inp = torch.zeros((f.size(0), f.size(1)), dtype=torch.float32, device=opts.device)
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    t = 0
    bs = opts.batch_size
    ps = opts.stomatic_init_num
    vs = opts.vector_init_num
    batchv = v.t().unsqueeze(0).expand((batch.size(0), -1, -1))
    exchange = None
    vec_sel = None

    vec = torch.bmm(f, batchv)

    R2_best = v[None, :, None, :].expand(bs, -1, ps, -1) * f[:, None, ...].expand(-1, vs, -1, -1)
    R2_best = R2_best.max(-1)[0].min(-1)[0]
    sov_best = vec.min(1)[0]

    while t < T:

        inp = torch.zeros((f.size(0), f.size(1)), dtype=torch.float32, device=opts.device)
        for i in range(opts.vector_init_num):
            p = (f * v[i]).sum(-1)
            f_now = p
            inp[torch.arange(batch.size(0)), p.min(-1)[1]] = 1

        baseline_val = []
        baseline_val_detached = []
        log_likelihood = []
        reward = []
        t_s = t
        if rollout_sel:
            r = torch.clamp(sov_best - rollout, min=0)
            r = r / (r.max(-1)[0].unsqueeze(-1) + 0.001)
            r = torch.nn.functional.softmax(r, dim=-1)
            vec_sel = r.multinomial(1).squeeze(-1)
        else:
            vec_sel = move_to(torch.randint(opts.vector_init_num, (opts.batch_size, 1)).squeeze(-1), opts.device)
        v_sel = torch.gather(v, dim=0, index=vec_sel.unsqueeze(-1).expand(-1, v.size(-1)))
        p = (f * v_sel.unsqueeze(1).expand(-1, f.size(1), -1)).sum(-1)
        f_sel = p
        _ = f_sel.min(-1)[1]
        f_sel[inp > 0] = -1 * np.inf
        _place = f_sel.max(-1)[1]
        span = _.unsqueeze(-1).unsqueeze(-1)
        span2 = _place.unsqueeze(-1).unsqueeze(-1)
        f_select = torch.gather(f, dim=1, index=span.expand(-1, -1, f.size(-1))).squeeze(1)
        select = torch.gather(solution, dim=1, index=span.expand(-1, -1, solution.size(-1))).squeeze(1)

        f_vec = v_sel
        p = (f_select * f_vec).sum(dim=-1)
        best_so_far = p
        while t - t_s < n_step and not (t == T):
            bl_val_detached, bl_val = baseline.eval(x_input, select, v_sel)
            baseline_val_detached.append(bl_val_detached)
            baseline_val.append(bl_val)
            exchange, log_lh = model(x_input, select, exchange, v_sel, do_sample=True)
            log_likelihood.append(log_lh)
            select = problem.step(select, exchange)
            select = move_to(select, opts.device)
            cost = problem.get_costs(batch, select)
            f_select = cost
            p = (cost * f_vec).sum(dim=-1)
            best_for_now = torch.cat((best_so_far[None, :], p[None, :]), 0).min(0)[0]
            reward.append(best_so_far - best_for_now)
            best_so_far = best_for_now.clone()
            t = t + 1

        # Get discounted R
        Reward = []
        reward_reversed = reward[::-1]
        f = torch.scatter(input=f, dim=1, index=span2.expand(-1, -1, f.size(-1)), src=f_select.unsqueeze(1))
        solution = torch.scatter(input=solution, dim=1, index=span2.expand(-1, -1, solution.size(-1)), src=select.unsqueeze(1))
        next_return, oo = baseline.eval(x_input, select, v_sel)
        for r in range(len(reward_reversed)):
            R = next_return * gamma + reward_reversed[r]
            Reward.append(R)
            next_return = R

        vec = torch.bmm(f, batchv)
        Reward = torch.stack(Reward[::-1], 0)
        baseline_val = torch.stack(baseline_val, 0)
        baseline_val_detached = torch.stack(baseline_val_detached, 0)
        log_likelihood = torch.stack(log_likelihood, 0)

        '''
        R2_now = v[None, :, None, :].expand(bs, -1, ps, -1) * f[:, None, ...].expand(-1, vs, -1, -1)
        R2_now = R2_now.max(-1)[0].min(-1)[0]
        rew = torch.cat((R2_best[None, :], R2_now[None, :]), 0).min(0)[0]
        reward_sel = (R2_best - rew).mean(-1) + gamma * hv_next
        R2_best = rew.clone()
        '''

        sov_now = vec.min(1)[0]
        rew = torch.cat((sov_now[None, :], sov_best[None, :]), 0).min(0)[0]
        sov_best = rew.clone()

        criteria = torch.nn.MSELoss()
        # reward_sel = (sov_best - rew).mean(-1) + gamma * hv_next
        # if train_sel:
        #     sel_model_loss = -((reward_sel.detach() - hv_base_detach) * logsel).mean()
        #     sel_baseline_loss = criteria(reward_sel, hv_base)
        #     sel_baseline_loss.backward()
        #     sel_model_loss.backward()
        # calculate loss
        baseline_loss = criteria(Reward.detach(), baseline_val)
        reinforce_loss = -((Reward.detach() - baseline_val_detached) * log_likelihood).mean()

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)

        current_step = int(step * T / n_step + t // n_step)
        # Logging to tensorboard
        baseline_loss.backward()
        reinforce_loss.backward()

        if (current_step - 1) % 200 == 0:
            print(reinforce_loss.mean(0), baseline_loss.mean(0))
            # print(R2_best.mean(), R2_best.mean(0).max(0)[0], R2_best.mean(0).min(0)[0])
            print(vec.min(1)[0].mean(), vec.min(1)[0].mean(0).max(0)[0], vec.min(1)[0].mean(0).min(0)[0])

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step()
        pbar.update(1)
    hv, _, _ = problem.calhv(opts.graph_size, f)
    print(reinforce_loss.mean(0), baseline_loss.mean(0))
    # print(R2_best.mean(), R2_best.mean(0).max(0)[0], R2_best.mean(0).min(0)[0])
    print(vec.min(1)[0].mean(), vec.min(1)[0].mean(0).max(0)[0], vec.min(1)[0].mean(0).min(0)[0])
    print(hv.mean(0), f.size(1))
    return vec.min(1)[0]
