import pickle
from dataclasses import dataclass
import torch
import tqdm
import numpy as np
from MTSPProblemDef import get_random_problems, augment_xy_data_by_8_fold
import math


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    dist: torch.Tensor
    # shape: (batch, problem, problem)
    log_scale: float
    route_num: float
    flag_return: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    lengths: torch.Tensor = None
    last_length: torch.Tensor = None
    agent_id: torch.Tensor = None
    max_dis: torch.Tensor = None
    remain_max_dis: torch.Tensor = None
    depot_num: torch.Tensor = None
    route_cnt: torch.Tensor = None
    city_num: torch.Tensor = None
    left_city: torch.Tensor = None


class MTSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size_low = env_params['problem_size_low']
        self.problem_size_high = env_params['problem_size_high']
        self.problem_size = env_params['sub_size']
        self.sample_size = env_params['sample_size']
        self.pomo_size = env_params['pomo_size']
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        self.tour = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None):
        if nodes_coords is not None:
            self.raw_problems = nodes_coords[episode:episode + batch_size]
        else:
            self.raw_problem_size = np.random.randint(self.problem_size_low // self.problem_size, self.problem_size_high // self.problem_size + 1) * self.problem_size
            self.raw_problems, self.M_number = get_random_problems(batch_size, self.raw_problem_size)

    def load_problems(self, batch_size, subp, route_num, flag, first_route, last_route, aug_factor=1):
        self.batch_size = batch_size

        self.problems = subp
        self.decision_length = subp.size(1)
        self.dist = torch.cdist(subp, subp, p=2)
        self.loc = subp.clone()[:, None, :, :].repeat(1, self.pomo_size, 1, 1)
        self.route_num = route_num
        self.cur_coord = subp[:, self.route_num.max(-1)[0]][:, None, None, :].repeat(1, self.pomo_size, 1, 1)
        self.flag = flag
        self.solution_list = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.long)
        self.solution_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.long)
        self.node_count = -1 * torch.ones((self.batch_size, self.pomo_size, 1), dtype=torch.long)
        self.lengths = torch.zeros((self.batch_size, self.pomo_size, route_num.max(-1)[0]), dtype=torch.float32)
        self.lengths[:, :, 0] = first_route[:, None].repeat(1, self.pomo_size)
        self.lengths = self.lengths.scatter_add(-1, (route_num - 1)[:, None, None].expand(-1, self.pomo_size, -1), last_route[:, None, None].repeat(1, self.pomo_size, 1))
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)
        self.left_city = torch.ones((self.batch_size, self.pomo_size), dtype=torch.long) * self.problem_size
        self.selected_depot = torch.zeros((self.batch_size, self.pomo_size, 1), dtype=torch.long)
        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.decision_length))
        self.depot_dis = self.dist[:, :, 0][:, None, :].repeat(1, self.pomo_size, 1)
        self.remain_dis = self.depot_dis.clone()
        self.max_dis = self.depot_dis.clone().max(dim=-1, keepdim=True)[0]
        log_scale = math.log2(self.problem_size)
        # shape: (batch, pomo, problem)
        self.agent_per = torch.arange(self.route_num.max(-1)[0]).cuda()[None, None, :].repeat(self.batch_size, self.pomo_size, 1)
        if (self.pomo_size > 1):
            for i in range(20):
                a = torch.randint(0, self.route_num.max(-1)[0], (self.pomo_size, 1)).cuda()[None, :] % self.route_num[:, None, None]
                b = torch.randint(0, self.route_num.max(-1)[0], (self.pomo_size, 1)).cuda()[None, :] % self.route_num[:, None, None]
                p = self.agent_per.gather(-1, a).clone()
                q = self.agent_per.gather(-1, b).clone()
                self.agent_per = self.agent_per.scatter(dim=-1, index=b, src=p)
                self.agent_per = self.agent_per.scatter(dim=-1, index=a, src=q)
        self.agent_per[:, 0, :] = torch.arange(self.route_num.max(-1)[0]).cuda()[None, :].repeat(self.batch_size, 1)
        self.agent_idx = self.agent_per[:, :, 0:1]
        self.depot_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length))
        self.depot_mask[self.BATCH_IDX, self.POMO_IDX, :self.lengths.size(-1)] = float('-inf')
        self.node_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length))
        self.node_mask[self.BATCH_IDX, self.POMO_IDX, self.lengths.size(-1):] = float('-inf')
        self.visited = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length))
        self.visited_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length))
        self.last = (torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.long) * (self.decision_length - 1)) * self.flag[:, None]
        self.last_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length))
        self.last_mask[:, :, -1][(self.flag == 0)[:, None].expand(-1, self.pomo_size)] = float('-inf')
        last_depot = self.agent_per.gather(-1, self.route_num[:, None, None].expand(-1, self.pomo_size, -1) - 1)
        last_depot_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length)).scatter(-1, last_depot, float('-inf'))
        self.last_mask[(self.flag == 1)[:, None, None].expand_as(self.last_mask)] = last_depot_mask[(self.flag == 1)[:, None, None].expand_as(self.last_mask)]

        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, :self.lengths.size(-1)] = float('-inf')
        self.step_state.ninf_mask = self.step_state.ninf_mask.scatter(-1, self.agent_idx, 0)
        reward = None
        done = False
        self.step_state.depot_num = self.route_num
        self.step_state.route_cnt = self.selected_depot
        self.step_state.left_city = self.left_city
        self.step_state.city_num = self.problem_size
        self.step_state.max_dis = self.max_dis
        self.step_state.remain_max_dis = self.remain_dis
        return Reset_State(problems=self.problems, dist=self.dist, log_scale=log_scale, flag_return=self.flag, route_num=self.route_num), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        is_city = selected >= self.lengths.size(2)
        self.left_city[is_city] -= 1
        self.depot_dis = self.depot_dis.scatter(-1, selected[:, :, None], 0)
        self.remain_dis = self.depot_dis.max(dim=-1, keepdim=True)[0].clone()
        cur_coord = self.loc.gather(2, selected[..., None, None].expand(-1, -1, -1, 2))
        path_lengths = (cur_coord - self.cur_coord).norm(p=2, dim=-1)
        self.lengths = self.lengths.scatter_add(-1, self.selected_depot, path_lengths)
        self.cur_coord = cur_coord.clone()
        self.visited[self.BATCH_IDX, self.POMO_IDX, self.current_node] = 1

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        # shape: (batch, pomo, node)
        self.finish = self.visited.sum(-1) == (self.problem_size + self.route_num[:, None] - 1)
        selected_mask = (selected[:, :, None] == self.agent_idx)
        selected_mask[self.finish[:, :, None]] = 0
        self.selected_depot[selected_mask] += torch.ones(self.selected_depot[selected_mask].shape, dtype=torch.int64, device=self.selected_depot.device)
        mask_current = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length)).scatter(-1, self.current_node[:, :, None], 1).bool()
        mask_current[self.finish[:, :, None].expand_as(mask_current)] = 0
        self.visited_mask[mask_current] = float('-inf')

        left_depot = self.route_num[:, None] - self.selected_depot.squeeze(-1) - 1
        mask_nodes = (self.left_city <= left_depot)
        mask_depot = (selected < self.route_num[:, None]) | (self.route_num[:, None] <= self.selected_depot.squeeze(-1))

        self.step_state.ninf_mask = self.visited_mask.clone()
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, :self.lengths.size(-1)] = float('-inf')
        mask_current_depot = torch.zeros(size=(self.batch_size, self.pomo_size, self.decision_length)).scatter(-1, self.agent_idx, 1).bool()
        mask_current_depot[left_depot[:, :, None].expand_as(mask_current_depot) == 0] = 0
        self.step_state.ninf_mask[mask_current_depot] = 0

        self.step_state.ninf_mask[mask_depot & ~self.finish] += self.depot_mask[mask_depot & ~self.finish]
        self.step_state.ninf_mask[mask_nodes & ~self.finish] += self.node_mask[mask_nodes & ~self.finish]
        condition_mask = (self.visited.sum(-1) < (self.problem_size + self.route_num[:, None] - 2)) & ~self.finish
        self.step_state.ninf_mask[condition_mask[:, :, None].expand_as(self.step_state.ninf_mask)] += self.last_mask[
            condition_mask[:, :, None].expand_as(self.step_state.ninf_mask)]
        self.agent_idx = self.agent_per.gather(2, self.selected_depot)
        if self.selected_count > 1:
            self.solution_flag = self.solution_flag.scatter_add(dim=-1, index=self.node_count, src=(selected[:, :, None] < self.route_num.max(-1)[0]).long())
        self.node_count[(selected[:, :, None] >= self.route_num.max(-1)[0]) & (self.node_count < self.solution_flag.size(-1) - 1)] += 1
        self.solution_list = self.solution_list.scatter_add(dim=-1, index=self.node_count, src=(selected[:, :, None] - self.route_num.max(-1)[0]).clamp_min_(0))
        self.step_state.route_cnt = self.selected_depot
        self.step_state.depot_id = self.agent_idx
        self.step_state.left_city = self.left_city
        self.step_state.remain_max_dis = self.remain_dis
        self.step_state.lengths = self.lengths
        # returning values
        done = self.finish.all()
        if done:
            reward = -1 * self.lengths.max(-1)[0]  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def make_dataset(self, filename, episode):
        nodes_coords = []
        tour = []

        # print('\nLoading from {}...'.format(filename))
        # print(filename)

        for line in open(filename, "r").readlines()[0:episode]:
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes_coords.append(
                [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
            )

            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]  # [:-1]
            tour.append(tour_nodes)

        return torch.tensor(nodes_coords), torch.tensor(tour)

    def make_dataset_pickle(self, filename, episode):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        raw_data_nodes = []
        for i in range(episode):
            raw_data_nodes.append(torch.tensor(data[i], requires_grad=False))
        raw_data_nodes = torch.stack(raw_data_nodes, dim=0)
        # shape (B )
        # shape (B,V,2)
        return raw_data_nodes

    def cal_info(self, problems, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=2).view(order_node_.size(0), -1)
        solution_flag = torch.stack((torch.zeros_like(index_bigger.long()), index_bigger.long()), dim=2).view(order_node_.size(0), -1).long()
        batch_size = solution.size(0)

        gathering_index = solution.unsqueeze(2).expand(batch_size, -1, 2)
        # shape: (batch, problem, 2)

        ordered_seq = problems.repeat(batch_size, 1, 1).gather(dim=1, index=gathering_index)
        # shape: (batch, problem, 2)

        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(-1).sqrt()
        # shape: (batch, problem)
        segment_sum = torch.cumsum(segment_lengths, dim=-1)
        double_segment_sum = torch.cumsum(torch.cat((segment_lengths, segment_lengths), dim=-1).clone(), dim=-1)
        double_solution_flag = torch.cat((solution_flag, solution_flag), dim=-1).clone()
        solution_start = torch.cummax(double_solution_flag.roll(dims=-1, shifts=1) * torch.arange(solution.size(-1) * 2)[None, :], dim=-1)[0]
        before = double_segment_sum.roll(dims=-1, shifts=1) - double_segment_sum.gather(-1, (solution_start - 2).clamp_min_(0))
        before = before[:, solution.size(-1):]
        before = before.view(-1, order_node.size(-1), 2)[:, :, 0]
        solution_end = 2 * solution.size(-1) - 1 - torch.flip(torch.cummax(torch.flip(double_solution_flag, dims=[-1]) * torch.arange(solution.size(-1) * 2)[None, :], dim=-1)[0],
                                                              dims=[-1])
        end = double_segment_sum.gather(-1, solution_end) - double_segment_sum.roll(dims=-1, shifts=-1)
        end = end[:, :solution.size(-1)]
        end = end.view(-1, order_node.size(-1), 2)[:, :, 0]
        # shape: (batch, pomo)
        return before, end

    def cal_info2(self, problems, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        solution_flag = torch.stack((torch.zeros_like(index_bigger.clone().long()), index_bigger.clone().long()), dim=3).view(order_node_.size(0), order_node_.size(1), -1).long()
        batch_size = solution.size(1)

        gathering_index = solution.unsqueeze(-1).expand(-1, -1, -1, 2)
        # shape: (batch, problem, 2)

        ordered_seq = problems[:, None, :].repeat(1, batch_size, 1, 1).gather(dim=2, index=gathering_index)
        # shape: (batch, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(-1).sqrt()
        # shape: (batch, problem)
        segment_sum = torch.cumsum(segment_lengths, dim=-1)
        double_segment_sum = torch.cumsum(torch.cat((segment_lengths, segment_lengths), dim=-1).clone(), dim=-1)
        double_solution_flag = torch.cat((solution_flag, solution_flag), dim=-1).clone()
        solution_start = torch.cummax(double_solution_flag.roll(dims=-1, shifts=1).clone() * torch.arange(solution.size(-1) * 2)[None, None, :], dim=-1)[0]
        before = double_segment_sum.roll(dims=-1, shifts=1).clone() - double_segment_sum.clone().gather(-1, (solution_start - 2).clamp_min_(0))
        before = before[:, :, solution.size(-1):]
        before = before.view(order_node.size(0), order_node.size(1), order_node.size(-1), 2)[:, :, :, 0]
        solution_end = 2 * solution.size(-1) - 1 - torch.flip(
            torch.cummax(torch.flip(double_solution_flag, dims=[-1]) * torch.arange(solution.size(-1) * 2)[None, None, :], dim=-1)[0],
            dims=[-1])
        end = double_segment_sum.clone().gather(-1, solution_end) - double_segment_sum.clone().roll(dims=-1, shifts=-1)
        end = end[:, :, :solution.size(-1)]
        end = end.view(order_node.size(0), order_node.size(1), order_node.size(-1), 2)[:, :, :, 0]
        # shape: (batch, pomo)
        return before, end

    def get_open_travel_distance(self, problems, order_node, order_flag, front_depot_length, next_depot_length):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        solution_flag = torch.stack((torch.zeros_like(index_bigger.long()), index_bigger.long()), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
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
        segment_lengths[:, :, -1] = 0
        segment_lengths[:, :, -2] += next_depot_length[:, None].repeat(1, segment_lengths.size(-2))
        solution[:, :, -1] = 0
        solution = torch.cat((torch.zeros_like(solution[:, :, 0:1]), solution), dim=-1)
        solution_flag = torch.cat((torch.zeros_like(solution_flag[:, :, 0:1]), solution_flag), dim=-1)
        segment_lengths = torch.cat((torch.zeros_like(segment_lengths[:, :, 0:1]), segment_lengths), dim=-1)
        segment_lengths[:, :, 0] += front_depot_length[:, None].repeat(1, segment_lengths.size(-2))
        solution_start = torch.cummax(solution_flag.roll(dims=-1, shifts=1) * torch.arange(solution.size(-1))[None, :], dim=-1)[0]
        solution_start = (solution_start - 1).clamp_min(0)
        pre0 = torch.cumsum(segment_lengths, dim=-1).roll(dims=-1, shifts=1)
        pre0[:, :, 0] = 0
        segment_sum = torch.cumsum(segment_lengths, dim=-1).roll(dims=-1, shifts=1) - pre0.gather(-1, solution_start)
        segment_sum *= solution_flag
        segment_sum[:, :, 0] = 0
        # shape: (batch, pomo)
        return segment_sum.max(-1)[0]

    def _get_travel_distance(self, problems, order_node, order_flag):
        order_node_ = order_node[None, :].clone()
        order_flag_ = order_flag[None, :].clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        solution_flag = torch.stack((torch.zeros_like(index_bigger.long()), index_bigger.long()), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)

        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        double_lengths = torch.cat((segment_lengths, segment_lengths), dim=-1)
        # shape: (batch, pomo, problem)
        solution_start = torch.cummax(torch.cat((solution_flag, solution_flag), dim=-1).roll(dims=-1, shifts=1) * torch.arange(solution.size(-1) * 2)[None, :], dim=-1)[0]
        segment_sum = torch.cumsum(double_lengths, dim=-1).roll(dims=2, shifts=1) - torch.cumsum(double_lengths, dim=-1).gather(-1, (solution_start - 2).clamp_min_(0))
        segment_sum = segment_sum[:, :, 2 * order_node_.size(-1):] * solution_flag
        # shape: (batch, pomo)
        return segment_sum.max(-1)[0]

    def _get_travel_distance2(self, problems, order_node, order_flag):
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        solution_flag = torch.stack((torch.zeros_like(index_bigger.long()), index_bigger.long()), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)

        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        double_lengths = torch.cat((segment_lengths, segment_lengths), dim=-1)
        # shape: (batch, pomo, problem)
        solution_start = torch.cummax(torch.cat((solution_flag, solution_flag), dim=-1).roll(dims=-1, shifts=1) * torch.arange(solution.size(-1) * 2)[None, :], dim=-1)[0]
        segment_sum = torch.cumsum(double_lengths, dim=-1).roll(dims=2, shifts=1) - torch.cumsum(double_lengths, dim=-1).gather(-1, (solution_start - 2).clamp_min_(0))
        segment_sum = segment_sum[:, :, 2 * order_node_.size(-1):] * solution_flag
        # shape: (batch, pomo)
        return segment_sum.max(-1)[0]

    def get_local_feature(self):
        if self.current_node is None:
            return None
        real_ps = self.dist.size(-1)
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, real_ps)
        '''
        cur_dist = torch.take_along_dim(
            self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.pomo_size, self.pomo_size),
            current_node, dim=2).squeeze(2)
        # shape: (batch, pomo, problem)'''
        cur_dist = self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, real_ps, real_ps).gather(2, current_node).squeeze(2)
        return cur_dist
