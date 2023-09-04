from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.motsp.state_motsp import StateMOTSP
from utils.beam_search import beam_search

import numpy as np
import matplotlib.pyplot as plt


class MOTSP(object):

    NAME = 'motsp'

    @staticmethod
    def get_costs(dataset, pi, w, num_objs):
        # Check that tours are valid, i.e. contain 0 to n -1

        def weighted_sum(dist1, dist2, crt_weight):
            return crt_weight[0] * dist1 + crt_weight[1] * dist2

        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        d = dataset.unsqueeze(1).expand(-1, w.size(0), -1, -1).reshape(-1, dataset.size(1), dataset.size(2))\
            .gather(1, pi.unsqueeze(-1).expand(-1, -1, dataset.size(-1)))
        if num_objs == 3:
            cor1 = d[..., :2]
            cor2 = d[..., 2:4]
            # cor3 = d[..., 4].unsqueeze(-1)
            cor3 = d[..., 4:]

            dist1 = (cor1[:, 1:] - cor1[:, :-1]).norm(p=2, dim=2).sum(1) + (cor1[:, 0] - cor1[:, -1]).norm(p=2, dim=1)
            dist2 = (cor2[:, 1:] - cor2[:, :-1]).norm(p=2, dim=2).sum(1) + (cor2[:, 0] - cor2[:, -1]).norm(p=2, dim=1)
            dist3 = (cor3[:, 1:] - cor3[:, :-1]).norm(p=2, dim=2).sum(1) + (cor3[:, 0] - cor3[:, -1]).norm(p=2, dim=1)

            dist1 = dist1.reshape(-1, w.size(0))
            dist2 = dist2.reshape(-1, w.size(0))
            dist3 = dist3.reshape(-1, w.size(0))

            w_rep = w.unsqueeze(0).expand(dist1.size(0), -1, -1)

            dist = (w_rep * torch.stack([dist1, dist2, dist3], dim=-1))
            # weighted sum
            # return dist.sum(-1).detach(), None, [dist1, dist2, dist3]
            # Tchebycheff
            return dist.max(-1)[0].detach(), None, [dist1, dist2, dist3]

        else:
            cor1 = d[..., :2]
            cor2 = d[..., 2:]
            dist1 = (cor1[:, 1:] - cor1[:, :-1]).norm(p=2, dim=2).sum(1) + (cor1[:, 0] - cor1[:, -1]).norm(p=2, dim=1)
            dist2 = (cor2[:, 1:] - cor2[:, :-1]).norm(p=2, dim=2).sum(1) + (cor2[:, 0] - cor2[:, -1]).norm(p=2, dim=1)

            dist1 = dist1.reshape(-1, w.size(0))
            dist2 = dist2.reshape(-1, w.size(0))
            w_rep = w.unsqueeze(0).expand(dist1.size(0), -1, -1)

            dist = (w_rep * torch.stack([dist1, dist2], dim=-1))
            # weighted sum
            return dist.sum(-1).detach(), None, [dist1, dist2]
            # Tchebycheff
            # return dist.max(-1)[0].detach(), None, [dist1, dist2]

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MOTSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMOTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = MOTSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class MOTSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, correlation=0, num_objs=2, mix_objs=0):
        super(MOTSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            self.data = torch.rand((num_samples, size, num_objs*2))
            if mix_objs > 0:
                self.data = self.data[:, :, :-mix_objs]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def load_kroAB(self, size):
        def read_tsp(path):
            cor_list = list()
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    info = line.split()
                    if info[0].isdigit():
                        cor_list.append(torch.tensor([int(info[1]), int(info[2])]))
            cor = torch.stack(cor_list, 1)
            return cor

        def kroAB(size):
            kroA = read_tsp('./data/kroA{}.tsp'.format(size))
            kroB = read_tsp('./data/kroB{}.tsp'.format(size))

            data = torch.stack(
                [(kroA[0] - kroA.min()) / (kroA.max() - kroA.min()),
                 (kroA[1] - kroA.min()) / (kroA.max() - kroA.min()),
                 (kroB[0] - kroB.min()) / (kroB.max() - kroB.min()),
                 (kroB[1] - kroB.min()) / (kroB.max() - kroB.min())],
                dim=1)
            return data

        self.data = [kroAB(size)]
        self.size = len(self.data)

    def load_rand_data(self, size, num_samples):
        path = './data/test200_instances_{}_mix3.pt'.format(size)
        if os.path.exists(path):
            pre_data = torch.load(path).permute(0, 2, 1)
            self.data = pre_data
            self.size = self.data.size(0)
            print(self.data.shape)
        else:
            print('Do not exist!', size, num_samples)
