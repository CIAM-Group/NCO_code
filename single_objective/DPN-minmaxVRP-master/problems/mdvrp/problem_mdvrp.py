from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.mdvrp.state_mdrvp import State_MDVRP
from utils.beam_search import beam_search
import numpy as np


class MDVRP(object):
    NAME = 'mdvrp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MDVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return State_MDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096, agent_num=5):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, agent_num=agent_num, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, *args = args
    # grid_size = 1
    # if len(args) > 0:
    #     depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float)
    }


class MDVRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(MDVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            if os.path.splitext(filename)[1] == '.tsp':
                problem = tsplib95.load(filename)
                max_val = np.array(list(problem.node_coords.values())).max()
                self.data = [torch.FloatTensor(list(problem.node_coords.values()))]
            else:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
                # print(self.data)
        else:
            # Sample points randomly in [0, 1] square
            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(size, 2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
