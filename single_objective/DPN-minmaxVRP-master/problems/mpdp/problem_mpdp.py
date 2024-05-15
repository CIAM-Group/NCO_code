from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.mpdp.state_mpdp import State_MPDP


class MPDP(object):
    NAME = 'mpdp'  # Capacitated Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi):  # pi:[batch_size, graph_size]
        assert (pi[:, 0] == 0).all(), "not starting at depot"
        assert (torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) == pi.data.sort(1)[0]).all(), "not visiting all nodes"

        visited_time = torch.argsort(pi, 1)  # pickup的index < 对应的delivery的index
        assert (visited_time[:, 1:pi.size(1) // 2 + 1] < visited_time[:, pi.size(1) // 2 + 1:]).all(), "deliverying without pick-up"
        # dataset['depot']: [batch_size, 2], dataset['loc']: [batch_size, graph_size, 2]
        dataset = torch.cat([dataset['depot'].reshape(-1, 1, 2), dataset['loc']], dim=1)  # [batch, graph_size+1, 2]
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))  # [batch, graph_size+1, 2]
        # d[:, :-1] do not include -1
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MPDPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return State_MPDP.initialize(*args, **kwargs)


def make_instance(args):
    depot, loc, *args = args
    # grid_size = 1
    # if len(args) > 0:
    #     depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float),
        'depot': torch.tensor(depot, dtype=torch.float)
    }


class MPDPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(MPDPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
