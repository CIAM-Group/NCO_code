from dataclasses import dataclass
import torch
import tqdm
import numpy as np
from ATSProblemDef import get_random_problems, augment_xy_data_by_8_fold
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


class ATSPEnv:
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
            self.raw_problems = get_random_problems(batch_size, self.raw_problem_size, self.env_params['problem_gen_params'])

    def load_problems(self, batch_size, subp, aug_factor=1):
        self.batch_size = batch_size

        self.problems = subp
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
        self.dist = self.problems
        log_scale = math.log2(self.problem_size)
        self.step_state.ninf_mask[:, :self.pomo_size, -1] = float('-inf')
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

    def _get_open_travel_distance(self):
        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size)
        # shape: (batch, pomo, node)

        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost[:, :, :-1].sum(2)
        # shape: (batch, pomo)

        return total_distance

    def get_open_travel_distance(self, problems, solution):
        node_from = solution
        # shape: (batch, pomo, node)
        node_to = solution.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)

        BATCH_IDX = torch.arange(problems.size(0))[:, None].expand(problems.size(0), solution.size(1))
        batch_index = BATCH_IDX[:, :, None].expand(problems.size(0), solution.size(1), solution.size(-1))
        # shape: (batch, pomo, node)

        selected_cost = problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost[:, :, :-1].sum(2)
        return total_distance

    def _get_travel_distance(self, problems, solution):
        solution = solution.clone()[None, :]
        node_from = solution
        # shape: (batch, pomo, node)
        node_to = solution.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)

        BATCH_IDX = torch.arange(problems.size(0))[:, None].expand(problems.size(0), solution.size(1))
        batch_index = BATCH_IDX[:, :, None].expand(problems.size(0), solution.size(1), solution.size(-1))
        # shape: (batch, pomo, node)

        selected_cost = problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        return total_distance

    def _get_travel_distance2(self, problems, solution):
        node_from = solution
        # shape: (batch, pomo, node)
        node_to = solution.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)

        BATCH_IDX = torch.arange(problems.size(0))[:, None].expand(problems.size(0), solution.size(1))
        batch_index = BATCH_IDX[:, :, None].expand(problems.size(0), solution.size(1), solution.size(-1))
        # shape: (batch, pomo, node)

        selected_cost = problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        return total_distance

    def get_local_feature(self):
        if self.current_node is None:
            return None
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.problem_size)

        cur_dist = self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, self.problem_size).gather(2, current_node).squeeze(2)
        return cur_dist
