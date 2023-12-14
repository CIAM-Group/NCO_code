from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search


class CVRP(object):
    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def _get_travel_distance(dataset, pi):
        # 逐步构建
        # 从t=1时间步开始
        # self.selected_node_list # shape: (batch, pomo, selected_node_list_length)
        # self.problem # shape: (batch, problem, 2)

        # 先获取当前时间节点的index
        # 然后获取上个时间节点的index
        torch.set_printoptions(profile="full")
        data = torch.cat((dataset['depot'].unsqueeze(1), dataset['loc']), dim=1)
        m1 = (pi == data.size(1))  # regret的那一步
        m2 = (m1.roll(dims=-1, shifts=-1) | m1)  # regret的那一步和后悔掉的那一步 路径长度都不加
        m3 = m1.roll(dims=-1, shifts=1)  # regret后重新选择的那一步 加相对前面3个t步的节点的距离
        m4 = ~(m2 | m3)  # 加相对前面1个t步的节点的距离
        m5 = ~ m2

        selected_node_list_right = pi.roll(dims=-1, shifts=1)
        selected_node_list_right2 = pi.roll(dims=-1, shifts=3)
        nf = torch.gather(data, 1, pi[:, 0].unsqueeze(-1).unsqueeze(-1).expand((-1, -1, 2))).squeeze()
        nl = torch.gather(data, 1, pi[:, -1].unsqueeze(-1).unsqueeze(-1).expand((-1, -1, 2))).squeeze()
        travel_distances = torch.zeros(data.size(0), device=pi.device)
        load = torch.zeros(data.size(0), device=pi.device)
        demand = torch.cat((torch.zeros(data.size(0), device=pi.device).unsqueeze(-1), dataset['demand']), dim=1)
        for t in range(pi.shape[1]):
            depot_index = torch.nonzero((pi[:, t] == 0))
            add1_index = torch.nonzero(m4[:, t].unsqueeze(-1))
            add3_index = torch.nonzero(m3[:, t].unsqueeze(-1))
            addl_index = torch.nonzero(m5[:, t].unsqueeze(-1))
            load[addl_index[:, 0]] += demand[addl_index[:, 0], pi[addl_index[:, 0], t]]
            assert (load <= 1.0001).all()
            load[depot_index[:, 0]] = 0
            travel_distances[add1_index[:, 0]] = travel_distances[add1_index[:, 0]].clone() + (
                    (data[add1_index[:, 0], pi[add1_index[:, 0], t], :] - data[add1_index[:, 0], selected_node_list_right[add1_index[:, 0], t],
                                                                          :]) ** 2).sum(1).sqrt()
            travel_distances[add3_index[:, 0]] = travel_distances[add3_index[:, 0]].clone() + (
                    (data[add3_index[:, 0], pi[add3_index[:, 0], t], :] - data[add3_index[:, 0], selected_node_list_right2[add3_index[:, 0], t],
                                                                          :]) ** 2).sum(1).sqrt()

        return travel_distances, None

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
                       torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                       sorted_pi[:, -graph_size:]
               ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
                       (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                       + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                       + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
               ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class SDVRP(object):
    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
                       (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                       + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                       + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
               ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
