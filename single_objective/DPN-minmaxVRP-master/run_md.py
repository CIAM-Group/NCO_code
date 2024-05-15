#!/usr/bin/env python

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pprint as pp

import torch
import torch.optim as optim
import torch.nn as nn
from options_md import get_options
from train import train_epoch, validate, get_inner_model
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

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
        'attention': AttentionModel
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
        ft = opts.ft
    ).to(opts.device)

    # if opts.use_cuda and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})


    if opts.ft == "N":
        # Initialize optimizer
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
        )
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.contextual_emb = nn.Sequential(nn.Linear(opts.embedding_dim, 8 * opts.embedding_dim, bias=False),
                nn.ReLU(),
                nn.Linear(8 * opts.embedding_dim, opts.embedding_dim, bias=False)
                )
        model = model.to(opts.device)
        optimizer = optim.Adam(
        [{'params': model.contextual_emb.parameters(), 'lr': opts.lr_model}]
    )
    
    # # Load optimizer state
    # if 'optimizer' in load_data:
    #     optimizer.load_state_dict(load_data['optimizer'])
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             # if isinstance(v, torch.Tensor):
    #             if torch.is_tensor(v):
    #                 state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    val_dataset2 = problem.make_dataset(
        size=opts.validate_size2, num_samples=opts.val_size, filename=opts.val_dataset2, distribution=opts.data_distribution)
    val_dataset3 = problem.make_dataset(
        size=opts.validate_size3, num_samples=opts.val_size, filename=opts.val_dataset3, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                val_dataset,
                val_dataset2,
                problem,
                opts
            )


if __name__ == "__main__":
    run(get_options())
