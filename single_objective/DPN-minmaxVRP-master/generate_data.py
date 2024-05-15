import argparse
import os
import random

import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_mdvrp_data(dataset_size, tsp_size):
    return list(zip(np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist(),  # Depot location
                    np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()
                    ))


def generate_mtsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_mcvrp_data(dataset_size, tsp_size):
    if tsp_size == 20:
        demand_scaler = 40
    elif tsp_size == 30:
        demand_scaler = 60
    elif tsp_size == 50:
        demand_scaler = 40
    elif tsp_size == 100:
        demand_scaler = 50
    a = np.concatenate([np.random.uniform(size=(dataset_size, tsp_size, 2)), np.random.randint(1, 10, size=(dataset_size, tsp_size, 1))], axis=-1)
    a[:, 0, -1] = 0
    demand_scaler = 0.4 * a[:, :, -1].sum()
    a[:, :, -1] /= demand_scaler
    return a.tolist()


def generate_pdp_data(dataset_size, pdp_size):
    return list(zip(np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
                    np.random.uniform(size=(dataset_size, pdp_size, 2)).tolist()
                    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='mtsp',
                        help="Problem, 'mtsp', 'mpdp', or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=1000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=2, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'mtsp': [None],
        'mcvrp': [None],
        'mpdp': [None],
        'mdvrp': [None],
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                random.seed(opts.seed)
                if problem == 'mtsp':
                    dataset = generate_mtsp_data(opts.dataset_size, graph_size)
                elif problem == 'mcvrp':
                    dataset = generate_mcvrp_data(opts.dataset_size, graph_size)
                elif problem == 'mpdp':
                    dataset = generate_pdp_data(opts.dataset_size, graph_size)
                elif problem == 'mdvrp':
                    dataset = generate_mdvrp_data(opts.dataset_size, graph_size)
                else:
                    assert False, "Unknown problem: {}".format(problem)

                # print(dataset[0])
                save_dataset(dataset, filename)
