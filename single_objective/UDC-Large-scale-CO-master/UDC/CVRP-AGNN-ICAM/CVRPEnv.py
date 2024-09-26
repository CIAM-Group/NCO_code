import math
import pickle
from dataclasses import dataclass
import torch
import numpy as np
from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    dist: torch.Tensor = None
    # shape: (batch, problem, problem)
    log_scale: float = None
    flag_return: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    left: torch.Tensor = None
    solution_flag: torch.Tensor = None
    solution_list: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size_low = env_params['problem_size_low']
        self.problem_size_high = env_params['problem_size_high']
        self.problem_size = env_params['sub_size']
        self.sample_size = env_params['sample_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.node_count = None
        self.solution_list = None
        self.solution_flag = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        self.demand_last = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None, nodes_demands=None):
        if nodes_coords is not None:
            self.raw_depot_node_xy = nodes_coords[episode:episode + batch_size]
            self.raw_depot_node_demand = nodes_demands[episode:episode + batch_size]
            self.raw_problems = torch.cat((self.raw_depot_node_xy, self.raw_depot_node_demand[:, :, None]), dim=-1)
        else:
            self.raw_problem_size = np.random.randint(self.problem_size_low // self.problem_size, self.problem_size_high // self.problem_size + 1) * self.problem_size
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.raw_problem_size)
            self.raw_depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
            # shape: (batch, problem+1, 2)
            depot_demand = torch.zeros(size=(batch_size, 1))
            # shape: (batch, 1)
            self.raw_depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
            self.raw_problems = torch.cat((self.raw_depot_node_xy, self.raw_depot_node_demand[:, :, None]), dim=-1)

    def load_problems(self, batch_size, depot_xy, node_xy, node_demand, flag, aug_factor=1):
        self.batch_size = batch_size
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        self.dist = torch.cdist(self.depot_node_xy, self.depot_node_xy, p=2)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.flag = flag
        self.demand_last = node_demand[:, -1]
        self.demand_last[self.flag] = 0

        self.reset_state.flag_return = flag
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def make_dataset_pickle2(self, filename, episode):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        raw_data_nodes = []
        raw_data_demand = []
        raw_data_capacity = data[0][3]
        for i in range(episode):
            raw_data_nodes.append(data[i][0] + data[i][1])
            raw_data_demand.append([0] + data[i][2])
        raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
        # shape (B )
        raw_data_demand = (torch.tensor(raw_data_demand, requires_grad=False) / raw_data_capacity)
        # shape (B,V,2)
        return raw_data_nodes, raw_data_demand, None, None

    def make_dataset_pickle(self, filename, episode):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        raw_data_nodes = []
        raw_data_demand = []
        raw_data_capacity = data[0][3]
        for i in range(episode):
            raw_data_nodes.append([data[i][0]] + data[i][1])
            raw_data_demand.append([0] + data[i][2])
        raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
        # shape (B )
        raw_data_demand = (torch.tensor(raw_data_demand, requires_grad=False) / raw_data_capacity)
        # shape (B,V,2)
        return raw_data_nodes, raw_data_demand, None, None

    def make_dataset(self, filename, episode):
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag

        raw_data_nodes = []
        raw_data_capacity = []
        raw_data_demand = []
        raw_data_cost = []
        raw_data_node_flag = []
        for line in open(filename, "r").readlines()[0:episode]:
            line = line.split(",")
            depot_index = int(line.index('depot'))
            customer_index = int(line.index('customer'))
            capacity_index = int(line.index('capacity'))
            demand_index = int(line.index('demand'))
            cost_index = int(line.index('cost'))
            node_flag_index = int(line.index('node_flag'))
            depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
            customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]
            loc = depot + customer
            capacity = int(float(line[capacity_index + 1]))
            demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
            cost = float(line[cost_index + 1])
            node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
            node_flag = tow_col_nodeflag(node_flag)
            raw_data_nodes.append(loc)
            raw_data_capacity.append(capacity)
            raw_data_demand.append(demand)
            raw_data_cost.append(cost)
            raw_data_node_flag.append(node_flag)

        raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
        # shape (B )
        raw_data_demand = torch.tensor(raw_data_demand, requires_grad=False) / torch.tensor(raw_data_capacity, requires_grad=False)[:, None]
        # shape (B,V+1) customer num + depot
        raw_data_cost = torch.tensor(raw_data_cost, requires_grad=False)
        # shape (B )
        raw_data_node_flag = torch.tensor(raw_data_node_flag, requires_grad=False)
        # shape (B,V,2)
        return raw_data_nodes, raw_data_demand, raw_data_cost, raw_data_node_flag


    def make_dataset_lib(self, filename):
        raw_data_nodes = []
        raw_data_capacity = []
        raw_data_demand = []
        raw_data_cost = []
        for line in open(filename, "r").readlines():
            line = line.replace(' ','')
            line = line.replace("'",'')
            line = line.split(",")
            depot_index = int(line.index('depot'))
            customer_index = int(line.index('customer'))
            demand_index = int(line.index('demand'))
            capacity_index = int(line.index('capacity'))
            cost_index = int(line.index('cost'))
            depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
            customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, demand_index, 2)]
            loc = depot + customer
            capacity = int(float(line[capacity_index + 1]))
            demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]
            cost = float(line[cost_index + 1])
            raw_data_nodes.append(torch.tensor(loc, requires_grad=False))
            raw_data_capacity.append(torch.tensor(capacity, requires_grad=False))
            raw_data_demand.append(torch.tensor(demand, requires_grad=False) / capacity)
            raw_data_cost.append(cost)
            print(cost)

        return raw_data_nodes, raw_data_demand, raw_data_cost

    def reset(self, capacity_now, capacity_end):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        self.solution_list = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.long)
        self.solution_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.long)
        self.node_count = -1 * torch.ones((self.batch_size, self.pomo_size, 1), dtype=torch.long)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        first = torch.ones(size=(self.batch_size, self.pomo_size)) * capacity_now
        last = torch.ones(size=(self.batch_size, self.pomo_size)) * capacity_end
        self.load = first
        self.left = last
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        self.last_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        self.last_mask[:, :, -1][(self.flag == 0)[:, None].expand(-1, self.pomo_size)] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

        self.reset_state.log_scale = math.log2(self.problem_size)
        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.left = self.left
        self.step_state.solution_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size), dtype=torch.long)
        self.step_state.solution_list = torch.zeros((self.batch_size, self.pomo_size, self.problem_size), dtype=torch.long)
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)
        if self.selected_count > 1:
            self.solution_flag = self.solution_flag.scatter_add(dim=-1, index=self.node_count, src=(selected[:, :, None] == 0).long())
        self.node_count[selected[:, :, None] != 0] += 1
        self.solution_list = self.solution_list.scatter_add(dim=-1, index=self.node_count, src=selected[:, :, None])
        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        round_error_epsilon = 0.0001
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        condition_mask = ((self.visited_ninf_flag[:, :, 1:] == float('-inf')).sum(-1) < self.problem_size - 1) | (
                1 - self.load + self.demand_last[:, None] > self.left + round_error_epsilon)
        self.ninf_mask[condition_mask[:, :, None].expand_as(self.ninf_mask)] += self.last_mask[condition_mask[:, :, None].expand_as(self.ninf_mask)]
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list

        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag[:, :, 1:] == float('-inf')).all(dim=2) & (1 - self.load < self.left + round_error_epsilon) & ~self.finished
        # shape: (batch, pomo)
        self.step_state.solution_list[newly_finished] = self.solution_list[:, :, :-1][newly_finished]
        self.step_state.solution_flag[newly_finished] = self.solution_flag[:, :, :-1][newly_finished]
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        return self.step_state, None, done

    def cal_open_length(self, problems, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def cal_leagal_2dim(self, demand, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        demand_ = demand.clone().squeeze()
        for i in range(order_node_.size(0)):
            for k in range(order_node_.size(1)):
                list_d = []
                now = 0
                for j in range(order_node_.size(2)):
                    now += demand_[i, order_node_[i, k, j]]
                    if order_flag_[i, k, j] == 1:
                        list_d.append(now)
                        now = 0
                list_demand = torch.stack(list_d, 0)
                list_demand[0] += now
                if (list_demand > 1 + 1e-5).any():
                    print('illeagal')
                # else:
                #    print('leagal')

    def cal_leagal(self, demand, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        demand_ = demand.clone().squeeze()
        for i in range(order_node_.size(0)):
            list_d = []
            now = 0
            for j in range(order_node_.size(1)):
                now += demand_[order_node_[i, j]]
                if order_flag_[i, j] == 1:
                    list_d.append(now)
                    now = 0
            list_demand = torch.stack(list_d, 0)
            list_demand[0] += now
            if (list_demand > 1 + 1e-5).any():
                print('illeagal')
            # else:
            #     print('leagal')

    def cal_length_total(self, problems, order_node, order_flag):
        order_node_ = order_node[None, :, :].clone()
        order_flag_ = order_flag[None, :, :].clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def cal_length_total2(self, problems, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def get_local_feature(self):
        if self.current_node is None:
            return None
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.problem_size + 1)
        cur_dist = self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, self.problem_size + 1).gather(2, current_node).squeeze(2)
        return cur_dist
