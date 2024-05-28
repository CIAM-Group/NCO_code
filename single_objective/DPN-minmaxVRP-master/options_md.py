import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='fmdvrp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=100, help="The size of the problem graph")
    parser.add_argument('--pomo_size', type=int, default=60, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=256, help='Number of instances per batch during training')  # 512
    parser.add_argument('--epoch_size', type=int, default=256000, help='Number of instances per epoch during training')  # 1280000
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default='data/mdvrp/mdvrp100_test_seed2.pkl', help='Dataset file to use for validation')
    parser.add_argument('--val_dataset2', type=str, default='data/mdvrp/mdvrp100_test_seed2.pkl', help='Dataset file to use for validation')
    parser.add_argument('--validate_size2', type=int, default=200, help="The size of the problem graph")
    parser.add_argument('--val_dataset3', type=str, default='data/mdvrp/mdvrp100_test_seed2.pkl', help='Dataset file to use for validation')
    parser.add_argument('--validate_size3', type=int, default=500, help="The size of the problem graph")
    parser.add_argument('--N_aug', type=int, default=1, help="The size of the problem graph")
    parser.add_argument('--subloss', type=int, default=0, help="The size of the problem graph")

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=6,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=50.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--agent_min', default=2, type=int, help="decide the number of agent")
    parser.add_argument('--agent_max', default=10, type=int, help="decide the number of robot")
    parser.add_argument('--depot_min', default=5, type=int, help="decide the number of agent")
    parser.add_argument('--depot_max', default=10, type=int, help="decide the number of robot")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=500, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=3333, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument('--eval_batch_size', type=int, default=20,
                        help="Batch size to use during evaluation")
    parser.add_argument('--aug_eval', type=int, default=8, help="Number of augmentations in evaluation")
    parser.add_argument('--agent_list', default=[5, 7, 10], help="Number of M in evaluation")
    parser.add_argument('--depot_eval', default=8, help="Number of D in evaluation")
    parser.add_argument('--r_eval', type=int, default=16, help="Number of permutations in evaluation")

    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from') # , default='pretrained/MDVRP-trained/FMDVRP-100-ft.pt'
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    # Finetuning
    parser.add_argument('--ft', default="N", type=str, help='Finetuning')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
