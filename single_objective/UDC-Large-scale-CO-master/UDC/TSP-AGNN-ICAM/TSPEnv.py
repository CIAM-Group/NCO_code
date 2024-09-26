import pickle
from dataclasses import dataclass
import torch
import tqdm
import numpy as np
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold
import math


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    dist: torch.Tensor
    # shape: (batch, problem, problem)
    log_scale: float


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
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
            self.raw_problems = get_random_problems(batch_size, self.raw_problem_size)

    def load_problems(self, batch_size, subp, aug_factor=1):
        self.batch_size = batch_size

        self.problems = subp
        self.dist = torch.cdist(subp, subp, p=2)
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

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        self.dist = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p=2, dim=-1)
        log_scale = math.log2(self.problem_size)
        self.step_state.ninf_mask[:, :self.pomo_size // 2, -1] = float('-inf')
        self.step_state.ninf_mask[:, self.pomo_size // 2:, 0] = float('-inf')
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(problems=self.problems, dist=self.dist, log_scale=log_scale), reward, done

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
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count >= self.problem_size - 1)
        if done and self.selected_node_list.size(-1) == self.problem_size:
            reward = -self._get_open_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def make_dataset2(self, filename, episode):
        nodes_coords = []
        tour = []

        # print('\nLoading from {}...'.format(filename))
        # print(filename)
        data = torch.load(filename)['node_xy'][0:episode]
        return data, None

    def make_dataset3(self, filename, episode):
        nodes_coords = []
        tour = []

        # print('\nLoading from {}...'.format(filename))
        # print(filename)
        raw_data_nodes = []
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        for i in range(episode):
            raw_data_nodes.append(data[i])
        raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
        # shape (B,V,2)
        return raw_data_nodes, None

    def make_solution(self, filename, episode):

        tour = []

        # print('\nLoading from {}...'.format(filename))
        # print(filename)

        for line in open(filename, "r").readlines()[0:episode]:
            line = line.split(" ")
            num_nodes = int(line.index('\n'))
            tour.append(
                [int(line[idx + 1]) for idx in range(2, num_nodes-1, 1)]
            )

        return torch.tensor(tour)

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

    def make_tsplib_data(self, filename):
        instance_data = []
        cost = []
        instance_name = []
        for line in open(filename, "r").readlines():
            line = line.rstrip("\n")
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('\'', '')
            line = line.split(sep=',')

            line_data = torch.from_numpy(np.array(line[2:], dtype=float).reshape(-1, 2)).cuda().float()
            instance_data.append(line_data)
            cost.append(float(line[1]))
            instance_name.append(line[0])  # 每一行的数据表示一个instance，每一个instance的size不一样
        cost = torch.tensor(cost)
        # print(instance_data.shape)

        return instance_data, cost, instance_name

    def _get_open_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        open_travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return open_travel_distances

    def get_open_travel_distance(self, problems, solution):
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

    def _get_travel_distance(self, problems, solution):
        solution = solution[None, :]
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

    def _get_travel_distance2(self, problems, solution):
        solution = solution.clone()
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

    def get_local_feature(self):
        if self.current_node is None:
            return None
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.problem_size)
        cur_dist = self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, self.problem_size).gather(2, current_node).squeeze(2)
        return cur_dist
