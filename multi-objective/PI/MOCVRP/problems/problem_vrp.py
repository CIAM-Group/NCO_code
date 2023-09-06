import torch
from torch.utils.data import Dataset
from itertools import combinations
import numpy as np

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class CVRPDataset(Dataset):

    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, DUMMY_RATE=1):
        super(CVRPDataset, self).__init__()

        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        capacity = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }
        max_demand = 9
        self.data = []
        self.size = int(np.ceil(size * (1 + DUMMY_RATE)))  # the number of real nodes plus dummy nodes in cvrp
        self.real_size = size  # the number of real nodes in cvrp

        if capacity[size] < max_demand:
            raise ValueError(':param max_load: must be > max_demand')
        seed = 1234
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.static = torch.rand(num_samples, 2, size + 1)
        demands = torch.randint(low=1, high=max_demand + 1, size=(num_samples, 1, size + 1)) / float(capacity[size])
        demands[:, 0, 0] = 0.
        self.demands = demands
        data = torch.cat([self.static, self.demands], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.static[idx], self.demands[idx]


class CVRP:
    NAME = 'cvrp'  # Capacitiated Vehicle Routing Problem

    def __init__(self):
        self.combination_list = None
        self.zero = None
        self.exchanged_idx_rep = None
        self.upper_tiangle_idx_rep = None
        self.triu_mask = None
        self.idx_mat = None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPDataset(*args, **kwargs)

    def input_feature_encoding(self, batch):
        return torch.cat((batch[0], batch[1]), 1)  # solution-independent features

    def greedy_solve2(self, maps):
        batch_size, feature_dim, map_size = maps.shape
        dynamic = maps[:, -1, :].clone()
        available_load = torch.ones((batch_size, map_size), device=device, requires_grad=False)
        solutions = list()
        selected = torch.zeros(batch_size, dtype=torch.long, device=device, requires_grad=False)
        solutions.append(selected)
        pre_info = maps[:, :, 0].unsqueeze(2).expand(-1, -1, map_size)
        disturb = torch.ones((batch_size, map_size), device=device, dtype=float, requires_grad=False) * 1e-4
        while self._unfinish(dynamic).any():
            capacity_mask = torch.lt(dynamic, available_load + disturb)
            demand_mask = torch.gt(dynamic, -1)
            depot_mask = torch.ones(capacity_mask.shape, device=device, dtype=torch.bool, requires_grad=False)
            is_depot = torch.eq(selected, 0)
            if is_depot.any():
                depot_mask[torch.nonzero(is_depot, as_tuple=False).squeeze(), 0] = 0

            mask = capacity_mask * demand_mask * depot_mask
            mask[torch.nonzero(~(mask.any(1)), as_tuple=False).squeeze(), 0] = 1
            mask_rep = mask.unsqueeze(1).expand(-1, feature_dim, -1)
            candidate_info = torch.where(mask_rep, maps, torch.tensor(1e5, device=device))
            dist = torch.norm(candidate_info[:, :2, :] - pre_info[:, :2, :], p=2, dim=1)
            priority = dist - (dynamic / 1e3)
            _, selected = torch.min(priority, 1)
            selected_rep = selected[:, None, None].expand(-1, feature_dim, -1)
            pre_info = torch.gather(maps, 2, selected_rep)
            solutions.append(selected)
            no_depot = torch.nonzero(selected, as_tuple=False).squeeze(1)
            available_load[no_depot, :] = available_load[no_depot, :] - dynamic[no_depot, selected[no_depot]][:,
                                                                        None].repeat(1, map_size)
            dynamic[no_depot, selected[no_depot]] = -1
            is_depot2 = torch.nonzero(torch.eq(selected, 0), as_tuple=False).squeeze()
            available_load[is_depot2, :] = 1.0
        solutions.append(torch.zeros(batch_size, device=device, dtype=torch.long).data)
        solutions = torch.stack(solutions, dim=1)
        depot_pad = torch.zeros(batch_size, 2 * map_size - solutions.shape[1] - 2, device=device, dtype=torch.long)
        solutions = torch.cat((solutions, depot_pad), dim=1)
        return solutions

    def random_solve0(self, maps):
        batch_size, feature_dim, map_size = maps.shape
        solutions = torch.cat((
            torch.zeros((map_size - 1, 1), dtype=torch.long, device=maps.device),
            torch.arange(map_size - 1, dtype=torch.long, device=maps.device)[:, None] + 1), dim=-1).view(-1)
        return solutions[None, :].expand((batch_size, -1))

    def random_solve1(self, maps):
        batch_size, feature_dim, map_size = maps.shape
        dynamic = maps[:, -1, :].clone()

        available_load = torch.ones((batch_size, map_size), device=device, requires_grad=False)
        solutions = list()
        selected = torch.zeros(batch_size, dtype=torch.long, device=device, requires_grad=False)
        solutions.append(selected)
        disturb = torch.ones((batch_size, map_size), dtype=float, device=device, requires_grad=False) * 1e-4
        while self._unfinish(dynamic).any():
            capacity_mask = torch.le(dynamic, available_load + disturb)
            demand_mask = torch.gt(dynamic, -1)
            depot_mask = torch.ones(capacity_mask.shape, device=device, dtype=torch.bool, requires_grad=False)
            is_depot = torch.eq(selected, 0)
            if is_depot.any():
                depot_mask[torch.nonzero(is_depot, as_tuple=False).squeeze(), 0] = 0
            mask = capacity_mask * demand_mask * depot_mask
            mask[torch.nonzero(~(mask.any(1)), as_tuple=False).squeeze(), 0] = 1
            probs = torch.einsum('bm , b -> bm', mask, (1 / torch.count_nonzero(mask, 1)))
            selected = probs.multinomial(1).squeeze()
            solutions.append(selected)
            no_depot = torch.nonzero(selected, as_tuple=False).squeeze(1)
            available_load[no_depot, :] = available_load[no_depot, :] - dynamic[no_depot, selected[no_depot]][:,
                                                                        None].repeat(1, map_size)
            dynamic[no_depot, selected[no_depot]] = -1
            is_depot2 = torch.nonzero(torch.eq(selected, 0), as_tuple=False).squeeze()
            available_load[is_depot2, :] = 1.0
        solutions.append(torch.zeros(batch_size, device=device, dtype=torch.long).data)

        solutions = torch.stack(solutions, dim=1)

        depot_pad = torch.zeros(batch_size, 2 * map_size - solutions.shape[1] - 2, device=device, dtype=torch.long)
        solutions = torch.cat((solutions, depot_pad), dim=1)
        return solutions

    def _unfinish(self, dynamic):
        dynamic = torch.clamp(dynamic, 0, 1)
        demands = torch.sum(dynamic, dim=1)
        unfinish = torch.gt(demands, 0)
        return unfinish

    def evaluate(self, solutions):
        num_vehicles = self._get_num_vehicles(solutions)
        dist = self._get_distance(solutions)
        return dist.detach(), num_vehicles.detach()

    def _get_num_vehicles(self, solutions):
        load_info = solutions[:, 2, :].cumsum(1)
        load_info1 = \
            torch.sort(torch.where(torch.eq(solutions[:, 2, :], 0), load_info, torch.zeros(1, device=load_info.device)),
                       dim=1)[0]
        load_info2 = torch.cat((load_info1[:, 1:], load_info1[:, -1][:, None]), dim=1)
        depot = (load_info2 - load_info1).gt(1e-2)
        num_v = torch.count_nonzero(depot, 1).float()
        return num_v

    def _get_distance(self, solutions):
        distances = torch.norm(solutions[:, :2, :-1] - solutions[:, :2, 1:], p=2, dim=1).sum(1)
        return distances

    def feasibility_check(self, tour_idx, tour_info):
        batch_size, tour_length = tour_idx.shape

        if self.combination_list is None:
            # guarantee the head and tail are depot
            index_list = list(range(1, tour_length - 1))
            self.combination_list = torch.tensor(list(combinations(index_list, 2)), dtype=torch.long, device=device, requires_grad=False)

        if self.exchanged_idx_rep is None:
            exchanged_idx_list = list()
            idx = torch.tensor([i for i in range(tour_length)])
            for pair in self.combination_list:
                idx_copy = idx.clone()
                idx_copy[pair[0]:pair[1] + 1] = torch.flip(idx_copy[pair[0]:pair[1] + 1], [0]).clone()
                exchanged_idx_list.append(idx_copy)

            self.exchanged_idx_rep = torch.stack(exchanged_idx_list, 1).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

        if self.triu_mask is None:
            self.triu_mask = torch.triu(torch.ones((tour_length, tour_length), device=device, dtype=torch.bool),
                                        diagonal=1).repeat(batch_size, 1, 1)
            self.triu_mask[:, 0, :] = 0
            self.triu_mask[:, :, -1] = 0

        num_pairs = self.exchanged_idx_rep.shape[2]
        tour_idx_rep = tour_idx.unsqueeze(2).expand(-1, -1, num_pairs)
        exchanged_tour_idx = torch.gather(tour_idx_rep, 1, self.exchanged_idx_rep)
        demands_rep = tour_info[:, -1, :].unsqueeze(2).expand(-1, -1, num_pairs)
        exchanged_demands = torch.gather(demands_rep, 1, exchanged_tour_idx)
        # demands
        load_info = exchanged_demands.cumsum(1)
        if self.zero is None:
            self.zero = torch.zeros(load_info.shape, device=device)
        load_info1 = torch.sort(torch.where(torch.eq(exchanged_demands, 0), load_info, self.zero), dim=1)[0]
        load_info2 = torch.cat((load_info1[:, 1:, :], load_info1[:, -1, :][:, None, :]), dim=1)
        feasibility_mask = ((load_info2 - load_info1) <= 1.0 + 1e-5).all(dim=1)
        mask = torch.zeros((batch_size, tour_length, tour_length), device=device, dtype=torch.bool)
        mask[self.triu_mask] = feasibility_mask.flatten()
        return mask

    def get_costs(self, batch, rec):
        batch_size, size = rec.size()
        # check feasibility
        batch = batch[0].clone().permute((0, 2, 1))
        # calculate obj value
        # first_row = torch.arange(size, device = rec.device).long().unsqueeze(0).expand(batch_size, size)
        d = batch.gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        tour_len = (d[:, :-1] - d[:, 1:]).norm(p=2, dim=2).sum(1)
        tours = torch.zeros(batch_size, device=d.device)
        best = torch.zeros(batch_size, device=d.device)
        last_node = batch[:, 0, :]
        for i in range(rec.size(1)):
            decoder_input = d[:, i, :]
            tours = tours + (decoder_input - last_node).norm(dim=1, p=2)
            last_node = decoder_input
            now = torch.zeros(batch_size, device=d.device)
            now[rec[:, i] == 0] = tours[rec[:, i] == 0]
            tours[rec[:, i] == 0] = 0
            best = torch.cat((best[None, :], now[None, :]), 0).max(0)[0]
        best = torch.cat((best[None, :], tours[None, :]), 0).max(0)[0]
        return torch.cat((tour_len[:, None], best[:, None]), dim=1)

    def opt2(self, tour_idx, idx):
        batch_size, tour_length = tour_idx.shape
        if self.idx_mat is None:
            triu_mask = torch.triu(torch.ones((tour_length, tour_length), device=device, dtype=torch.bool), diagonal=1)  # .repeat(batch_size, 1, 1)
            triu_mask[0, :] = 0
            triu_mask[:, -1] = 0
            self.idx_mat = torch.zeros(tour_length, tour_length, dtype=torch.long, device=device)
            self.idx_mat[triu_mask] = torch.tensor([i for i in range((tour_length - 3) * (tour_length - 2) // 2)], device=device)
            self.idx_mat = self.idx_mat.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size, -1)

        real_idx = torch.gather(self.idx_mat, 1, idx)
        real_idx_rep = real_idx.unsqueeze(1).repeat(1, tour_length, 1)
        pair_selected = torch.gather(self.exchanged_idx_rep, 2, real_idx_rep)

        tour_idx = torch.gather(tour_idx.unsqueeze(2), 1, pair_selected).squeeze(2)
        return tour_idx

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
