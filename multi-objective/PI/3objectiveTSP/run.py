import os
import json
import pprint
from options import get_options

import torch
import torch.optim as optim

from nets.critic_network import CriticNetwork
from train import train_epoch, validate
from nets.reinforce_baselines import CriticBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem, get_inner_model


def run(opts):
    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)(
        p_size=opts.graph_size,
        with_assert=not opts.no_assert)

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
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        hidden_dim=opts.hidden_dim,
        n_heads=opts.n_heads_encoder,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization,
        device=opts.device
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Load the validation datasets
    val_dataset = None #problem.make_dataset(size=opts.graph_size * opts.num_obj, num_samples=opts.val_size, filename=opts.val_dataset)

    baseline = CriticBaseline(
        CriticNetwork(
            problem=problem,
            embedding_dim=opts.embedding_dim,
            hidden_dim=opts.hidden_dim,
            n_heads=opts.n_heads_decoder,
            n_layers=opts.n_encode_layers,
            normalization=opts.normalization,
            device=opts.device
        ).to(opts.device)
    )

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0 else []
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

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1
    tag =0 # another way to resume
    if tag > 0:
        opts.epoch_start = tag
        checkpoint = torch.load('/home/zhengzhi/zzTSPL2H20/outputs/tsp_40/run_name_20211015T103707/epoch-18.pt', map_location=opts.device)
        model.load_state_dict(checkpoint['model'])
        baseline.load_state_dict(checkpoint['baseline'])
    rollout = torch.ones((opts.batch_size, opts.vector_init_num), dtype=torch.float32, device=opts.device)
    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(
            problem,
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            rollout,
            val_dataset,
            tb_logger,
            opts
        )


if __name__ == "__main__":
    import warnings

    # torch.set_default_tensor_type(torch.float)
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run(get_options())
