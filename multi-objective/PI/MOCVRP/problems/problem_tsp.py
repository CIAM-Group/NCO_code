import random

from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle
import math


class TSP(object):
    NAME = 'tsp'

    def __init__(self, p_size=20, with_assert=False):

        self.size = p_size  # the number of nodes in tsp
        self.do_assert = with_assert
        print(f'TSP with {self.size} nodes.')

    def step(self, rec, exchange):

        device = rec.device

        exchange_num = exchange.clone().cpu().numpy()
        rec_num = rec.clone().cpu().numpy()

        for i in range(rec.size()[0]):

            loc_of_first = np.where(rec_num[i] == exchange_num[i][0])[0][0]
            loc_of_second = np.where(rec_num[i] == exchange_num[i][1])[0][0]

            if (loc_of_first < loc_of_second):
                rec_num[i][loc_of_first:loc_of_second + 1] = np.flip(
                    rec_num[i][loc_of_first:loc_of_second + 1])
            else:
                temp = rec_num[i][loc_of_first]
                rec_num[i][loc_of_first] = rec_num[i][loc_of_second]
                rec_num[i][loc_of_second] = temp

        return torch.tensor(rec_num, device=device)

    def calpareto(self,r,f):
        pareto = [[0] for i in range(f.size(0))]
        inp = torch.zeros((f.size(0), f.size(1)), dtype=torch.float32)
        for now in range(f.size(1)):
            for i in range(f.size(0)):
                tag = 1
                for j in range(len(pareto[i])):
                    if f[i][now][0] >= f[i][pareto[i][j]][0] and f[i][now][1] >= f[i][pareto[i][j]][1]:
                        tag = 0
                if tag == 1:
                    j = 0
                    while j < len(pareto[i]):
                        if f[i][now][0] < f[i][pareto[i][j]][0] and f[i][now][1] < f[i][pareto[i][j]][1]:
                            inp[i][pareto[i][j]] = 0
                            pareto[i].pop(j)
                        else:
                            j += 1
                    if (len(pareto[i]) == 0):
                        inp[i][now] = 1
                        pareto[i].insert(0, now)
                        continue
                    for j in range(len(pareto[i])):
                        if f[i][now][0] < f[i][pareto[i][j]][0]:
                            inp[i][now] = 1
                            pareto[i].insert(j, now)
                            tag = 0
                            break
                    if tag == 1:
                        inp[i][now] = 1
                        pareto[i].insert(len(pareto[i]), now)
        return pareto, inp

    def calhv(self, r, f):
        pareto = [[0] for i in range(f.size(0))]
        inp = torch.zeros((f.size(0), f.size(1)), dtype=torch.float32)
        for now in range(f.size(1)):
            for i in range(f.size(0)):
                tag = 1
                for j in range(len(pareto[i])):
                    if f[i][now][0] >= f[i][pareto[i][j]][0] and f[i][now][1] >= f[i][pareto[i][j]][1]:
                        tag = 0
                if tag == 1:
                    j = 0
                    while j < len(pareto[i]):
                        if f[i][now][0] < f[i][pareto[i][j]][0] and f[i][now][1] < f[i][pareto[i][j]][1]:
                            inp[i][pareto[i][j]] = 0
                            pareto[i].pop(j)
                        else:
                            j += 1
                    if (len(pareto[i]) == 0):
                        inp[i][now] = 1
                        pareto[i].insert(0, now)
                        continue
                    for j in range(len(pareto[i])):
                        if f[i][now][0] < f[i][pareto[i][j]][0]:
                            inp[i][now] = 1
                            pareto[i].insert(j, now)
                            tag = 0
                            break
                    if tag == 1:
                        inp[i][now] = 1
                        pareto[i].insert(len(pareto[i]), now)
        ans = torch.zeros(f.size(0), dtype=float)
        ans = ans.float()
        for i in range(f.size(0)):
            y_now = r
            for j in range(len(pareto[i])):
                ans[i] = ans[i] + (r - f[i][pareto[i][j]][0]) * (y_now - f[i][pareto[i][j]][1])
                y_now = f[i][pareto[i][j]][1]
        return ans, pareto, inp

    '''
    def calhv(self, r, pareto, f_val):
        ans = torch.zeros(f_val.size(0), dtype=float)
        ans = ans.float()
        for i in range(f_val.size(0)):
            mn0 = r[i][0]
            mx0 = r[i][1]
            mn1 = r[i][2]
            mx1 = r[i][3]
            y_now = 1.1
            for j in range(len(pareto[i])):
                ans[i] = ans[i] + (1.1 - (f_val[i][pareto[i][j]][0].item() - mn0) / (mx0 - mn0)) * \
                         (y_now - (f_val[i][pareto[i][j]][1].item() - mn1) / (mx1 - mn1))
                y_now = (f_val[i][pareto[i][j]][1].item() - mn1) / (mx1 - mn1)
        return ans

    '''

    def get_costs(self, dataset, rec):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param rec: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        if self.do_assert:
            assert (
                    torch.arange(rec.size(1), out=rec.data.new()).view(1, -1).expand_as(rec) ==
                    rec.data.sort(1)[0]
            ).all(), "Invalid tour"

        # Gather dataset in order of tour
        '''
        d = dataset.gather(2, rec.long().unsqueeze(1).expand(-1, dataset.size(1), -1).unsqueeze(-1).expand_as(dataset))
        length = torch.zeros((d.size(0), d.size(1)), dtype=float)
        for i in range(d.size(0)):
            for j in range(d.size(1)):
                for k in range(d.size(2) - 1):
                    length[i][j] += math.sqrt(
                        (d[i][j][k][0] - d[i][j][k + 1][0]) * (d[i][j][k][0] - d[i][j][k + 1][0]) + (d[i][j][k][1] - d[i][j][k + 1][1]) * (
                                d[i][j][k][1] - d[i][j][k + 1][1]))
                length[i][j] += math.sqrt(
                    (d[i][j][-1][0] - d[i][j][0][0]) * (d[i][j][-1][0] - d[i][j][0][0]) + (d[i][j][-1][1] - d[i][j][0][1]) * (d[i][j][-1][1] - d[i][j][0][1]))
        # length = (d[:, :, 1:] - d[:, :, :-1]).norm(p=2, dim=2).sum(1) + (d[:, :, 0] - d[:, :, -1]).norm(p=2, dim=2)
        '''
        # length = self.getcost(dataset[:, 0, ...], rec).unsqueeze(-1)
        # for i in range(1, dataset.size(1)):
        #    length = torch.cat((length, self.getcost(dataset[:, i, ...], rec).unsqueeze(-1)), dim=1)
        length = self.getcost2(dataset, rec)
        return length

    def getcost2(self, data, rec):
        d = data.gather(2, rec.long().unsqueeze(1).unsqueeze(-1).expand_as(data))
        length = (d[:, :, 1:] - d[:, :, :-1]).norm(p=2, dim=3).sum(2) + (d[:, :, 0] - d[:, :, -1]).norm(p=2, dim=2)
        return length

    def getcost(self, data, rec):
        d = data.gather(1, rec.long().unsqueeze(-1).expand_as(data))
        length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
        return length

    def get_swap_mask(self, rec):
        _, graph_size = rec.size()
        return torch.eye(graph_size).view(1, graph_size, graph_size)

    def get_initial_solutions(self, methods, batch):

        def seq_solutions(batch_size):
            graph_size = self.size
            solution = torch.linspace(0, graph_size - 1, steps=graph_size)
            for i in range(graph_size):
                a = random.randint(0, graph_size - 1)
                b = random.randint(0, graph_size - 1)
                p = solution[a].clone()
                solution[a] = solution[b]
                solution[b] = p

            return solution.expand(batch_size, graph_size).clone()

        batch_size = len(batch)

        if methods == 'seq':
            return seq_solutions(batch_size)

    def greddy(self, batch, n_vec, device):
        batch = batch.to(device)
        bs, os, ns, fs = batch.shape
        n_vec = n_vec.unsqueeze(0).unsqueeze(-1).expand(bs, -1, ns).to(device)
        visit = batch.clone()
        f = torch.zeros((bs, os), dtype=torch.float32).to(device)
        selected = torch.zeros((bs, 1), dtype=torch.long, device=device, requires_grad=False)
        solution = selected
        pre_info = batch[..., 0, :].clone().unsqueeze(2)
        pre_info0 = pre_info.clone()
        visit[..., 0, :] = 1e5
        pre_info.expand(-1, -1, ns, -1)
        while (visit < 1e5).any():
            dist = torch.norm(visit - pre_info, p=2, dim=-1)
            selected = (dist * n_vec).sum(1).squeeze(1).min(1)[1]

            solution = torch.cat((solution, selected.unsqueeze(-1)), dim=1)
            selected_rep = selected[:, None, None, None].expand(-1, os, -1, fs)
            pre_info = torch.gather(visit, 2, selected_rep).clone().expand(-1, -1, ns, -1)
            visit.scatter_(dim=2, index=selected_rep, src=1e5 * torch.ones_like(visit, device=device))
            f = f + torch.gather(dist, 2, selected[:, None, None].expand(-1, os, -1)).squeeze(2)
        dist = torch.norm(pre_info0 - pre_info, p=2, dim=-1)
        selected = (dist * n_vec).sum(1).squeeze(1).min(1)[1]
        f = f + torch.gather(dist, 2, selected[:, None, None].expand(-1, os, -1)).squeeze(2)
        return solution.unsqueeze(1), f.unsqueeze(1)

    def stomatic(self, batch, n_vec, device):
        batch = batch.to(device)
        bs, os, ns, fs = batch.shape
        n_vec = n_vec.unsqueeze(0).unsqueeze(-1).expand(bs, -1, ns).to(device)
        visit = batch.clone()
        f = torch.zeros((bs, os), dtype=torch.float32).to(device)
        selected = torch.zeros((bs, 1), dtype=torch.long, device=device, requires_grad=False)
        solution = selected
        pre_info = batch[..., 0, :].clone().unsqueeze(2)
        pre_info0 = pre_info.clone()
        visit[..., 0, :] = 1e5
        pre_info.expand(-1, -1, ns, -1)
        while (visit < 1e5).any():
            dist = torch.norm(visit - pre_info, p=2, dim=-1)
            selected = (dist * n_vec).sum(1).squeeze(1).min(1)[1]

            solution = torch.cat((solution, selected.unsqueeze(-1)), dim=1)
            selected_rep = selected[:, None, None, None].expand(-1, os, -1, fs)
            pre_info = torch.gather(visit, 2, selected_rep).clone().expand(-1, -1, ns, -1)
            visit.scatter_(dim=2, index=selected_rep, src=1e5 * torch.ones_like(visit, device=device))
            f = f + torch.gather(dist, 2, selected[:, None, None].expand(-1, os, -1)).squeeze(2)
        dist = torch.norm(pre_info0 - pre_info, p=2, dim=-1)
        selected = (dist * n_vec).sum(1).squeeze(1).min(1)[1]
        f = f + torch.gather(dist, 2, selected[:, None, None].expand(-1, os, -1)).squeeze(2)
        return solution.unsqueeze(1), f.unsqueeze(1)

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=20, num_obj=2, num_mix=0, num_samples=10000):

        super(TSPDataset, self).__init__()

        self.data_set = []
        if num_mix:
            self.data = [torch.cat((torch.FloatTensor(num_obj - num_mix, size, 2).uniform_(0, 1),
                                    torch.cat((torch.FloatTensor(num_mix, size, 1).uniform_(0, 1), torch.zeros(num_mix, size, 1)), dim=-1)), dim=0) for i in
                         range(num_samples)]
        else:
            self.data = [torch.FloatTensor(num_obj - num_mix, size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
