#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem

import numpy as np
import copy
import matplotlib.pyplot as plt
import time


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))
    if not opts.eval_only:
        os.makedirs(opts.save_dir)
        # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    opts.logger_list = []

    opts.w_list = []
    if opts.num_objs == 2:
        w1_list = np.linspace(opts.lower_bound, opts.upper_bound, opts.num_weights).tolist()
        for w1 in w1_list:
            w = torch.Tensor([w1, 1 - w1])
            temp_tb_logger = TbLogger(
                os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name,
                             '{:.2f}_{:.2f}'.format(*w))) if not opts.eval_only else None
            opts.w_list.append(w)
            opts.logger_list.append(temp_tb_logger)
    elif opts.num_objs == 3:
        ws = get_w(m=opts.num_objs, H=opts.H)
        opts.num_weights = len(ws)
        for w in ws:
            temp_tb_logger = TbLogger(
                os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name,
                             '{:.2f}_{:.2f}_{:.2f}'.format(*w))) if not opts.eval_only else None
            opts.w_list.append(torch.Tensor(w))
            opts.logger_list.append(temp_tb_logger)
    else:
        assert opts.num_objs <= 3, 'Only support 2 or 3 objs so far!'
    opts.num_top = round(0.2 * len(opts.w_list) / opts.num_objs)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        num_objs=opts.num_objs,
        mix_objs=opts.mix_objs
    ).to(opts.device)
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(total_params)
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        for i in range(opts.num_weights):
            opts.w_list[i] = torch.cat([opts.w_list[i], opts.w_list[i]], dim=-1)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    elif opts.baseline == 'NoBaseline':
        baseline = NoBaseline()
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
        correlation=opts.correlation,
        num_objs=opts.num_objs,
        mix_objs=opts.mix_objs
    )

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        with torch.no_grad():
            gs = opts.graph_size
            opts.reference_point = gs
            print(opts.num_weights)
            val_dataset.load_rand_data(gs, opts.val_size)
            # val_dataset.load_kroAB(gs, opts.val_size)
            opts.start_time = time.time()
            _, all_objs_list, NDS, HV, num_NDS = validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


def perm(sequence):
    l = sequence
    if (len(l) <= 1):
        return [l]
    r = []
    for i in range(len(l)):
        if i != 0 and sequence[i - 1] == sequence[i]:
            continue
        else:
            s = l[:i] + l[i + 1:]
            p = perm(s)
            for x in p:
                r.append(l[i:i + 1] + x)
    return r

def get_w(m, H):
    sequence = []
    for ii in range(H):
        sequence.append(0)
    for jj in range(m - 1):
        sequence.append(1)
    ws = []

    pe_seq = perm(sequence)
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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
    opts = get_options()
    opts.start_time = time.time()
    run(opts)

