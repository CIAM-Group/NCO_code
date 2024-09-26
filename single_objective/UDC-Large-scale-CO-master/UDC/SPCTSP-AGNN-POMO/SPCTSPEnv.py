import pickle
from dataclasses import dataclass
import torch
import tqdm
import numpy as np
from SPCTSProblemDef import get_random_problems, augment_xy_data_by_8_fold
import math


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    dist: torch.Tensor
    # shape: (batch, problem, problem)
    log_scale: float
    first: torch.Tensor
    last: torch.Tensor
    depot_tag: torch.Tensor


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    load = None


class SPCTSPEnv:
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
        self.first = None
        self.last = None
        self.collected = None
        self.last_collected = None
        self.finished = None
        self.allow = None
        self.depot_flag = None
        self.depot_tag = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None):
        if nodes_coords is not None:
            self.raw_problems = nodes_coords[episode:episode + batch_size]
        else:
            self.raw_problem_size = np.random.randint(self.problem_size_low // self.problem_size, self.problem_size_high // self.problem_size + 1) * self.problem_size
            self.raw_problems = get_random_problems(batch_size, self.raw_problem_size)

    def load_problems(self, batch_size, subp, real_prize, first, last, collected, depot_flag, aug_factor=1):
        self.batch_size = batch_size

        self.problems = subp
        self.problems[:, :, 2] /= collected[:, None]
        self.rp = real_prize
        self.first = first
        self.last = last
        self.collected = collected
        self.load = torch.zeros(self.batch_size, self.pomo_size, device=last.device, dtype=torch.float32)
        self.dist = torch.cdist(subp[:, :, :2], subp[:, :, :2], p=2)
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
        self.pomo_last = torch.cat((self.last[:, None].repeat(1, self.pomo_size // 2), self.first[:, None].repeat(1, self.pomo_size // 2)), dim=1)
        self.last_collected = self.rp[:, None, :].expand(-1, self.pomo_size, -1).gather(-1, self.pomo_last[:, :, None]).squeeze()
        self.depot_tag = depot_flag.clone()
        self.depot_flag = depot_flag.clone()[:, None, :].repeat(1, self.pomo_size, 1)
        self.depot_flag[self.BATCH_IDX, self.POMO_IDX, self.pomo_last] = 0

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        self.dist = (self.problems[:, :, None, :2] - self.problems[:, None, :, :2]).norm(p=2, dim=-1)
        log_scale = math.log2(self.problem_size)
        self.step_state.ninf_mask[torch.arange(self.batch_size), :self.pomo_size // 2, self.last] = float('-inf')
        self.step_state.ninf_mask[torch.arange(self.batch_size), self.pomo_size // 2:, self.first] = float('-inf')
        self.step_state.load = 0
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(problems=self.problems, dist=self.dist, log_scale=log_scale, last=self.last, first=self.first, depot_tag=self.depot_tag), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        self.depot_flag[self.BATCH_IDX, self.POMO_IDX, self.current_node] = 0
        self.load += self.rp[:, None, :].expand(-1, self.pomo_size, -1).gather(-1, selected[:, :, None]).squeeze()
        # shape: (batch, pomo, node)
        allow = (self.load + self.last_collected + 1e-5 > self.collected[:, None]) & (self.depot_flag.sum(-1) == 0)
        last_hot = torch.zeros_like(self.step_state.ninf_mask)
        last_hot = last_hot.scatter(-1, self.pomo_last.unsqueeze(-1), 1).bool()
        self.step_state.ninf_mask[(allow[:, :, None] & last_hot)] = 0.
        self.finished = (self.current_node == self.pomo_last)
        self.step_state.ninf_mask[self.finished] = float('-inf')
        self.step_state.ninf_mask[(self.finished[:, :, None] & last_hot)] = 0
        self.step_state.load = self.load
        # returning values
        done = self.finished.all()
        if done and self.selected_count == self.problem_size:
            reward = -self._get_open_travel_distance()  # note the minus sign!
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
        raw_data_sto_prize = []
        raw_data_prize = []
        raw_data_penalty = []
        for i in range(episode):
            raw_data_nodes.append([data[i][0]] + data[i][1])
            raw_data_penalty.append([0] + data[i][2])
            raw_data_prize.append([0] + data[i][3])
            raw_data_sto_prize.append([0] + data[i][4])
        raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
        # shape (B )
        raw_data_prize = (torch.tensor(raw_data_prize, requires_grad=False))
        raw_data_sto_prize = (torch.tensor(raw_data_sto_prize, requires_grad=False))
        raw_data_penalty = (torch.tensor(raw_data_penalty, requires_grad=False))
        # shape (B,V,2)
        raw_data = torch.cat((raw_data_nodes, raw_data_sto_prize.unsqueeze(-1), raw_data_prize.unsqueeze(-1), raw_data_penalty.unsqueeze(-1)), dim=-1)
        return raw_data

    def _get_open_travel_distance(self):
        solution = self.selected_node_list.clone()
        visited = torch.ones((solution.size(0), solution.size(1), self.problems.size(-2)))
        visited = visited.scatter(-1, solution, 0)
        penalty = (visited * self.problems[:, :, -1][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :2].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)
        travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return travel_distances + penalty

    def get_open_travel_distance(self, problems, solution):
        solution = solution[:, None, :]
        visited = torch.ones_like(solution)
        visited = visited.scatter(-1, solution, 0)
        penalty = (visited * problems[:, :, -1][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :2].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return travel_distances + penalty

    def _get_travel_distance(self, problems, solution):
        solution = solution[None, :]
        visited = torch.ones_like(solution)
        visited = visited.scatter(-1, solution, 0)
        penalty = (visited * problems[:, :, -1][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :2].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances + penalty

    def _get_travel_distance2(self, problems, solution):
        solution = solution.clone()
        visited = torch.ones((solution.size(0), solution.size(1), problems.size(-2)))
        visited = visited.scatter(-1, solution, 0)
        penalty = (visited * problems[:, :, -1][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :2].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances + penalty

    def get_local_feature(self):
        if self.current_node is None:
            return None
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.problem_size)
        '''
        cur_dist = torch.take_along_dim(
            self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.pomo_size, self.pomo_size),
            current_node, dim=2).squeeze(2)
        # shape: (batch, pomo, problem)'''
        cur_dist = self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, self.problem_size).gather(2, current_node).squeeze(2)
        return cur_dist
