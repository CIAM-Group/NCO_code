
from dataclasses import dataclass
import torch

from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold

import numpy as np
import zipfile
import pickle

from torch.utils.data import Dataset, DataLoader
import itertools

import math

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


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
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        self.solutions = None
        self.solutions_flipped = None
        self.offset = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)
        data = self.env_params['data']
        batch_size = self.env_params['batch_size']
        if data is None:
            self.dataset = TSPDataSet(self.problem_size)
            dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)
            self.dataloader = itertools.cycle(enumerate(dataloader))
            self.vali = False
        else:
            dataset = data
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            self.dataloader = itertools.cycle(enumerate(dataloader))
            self.vali = True

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.offset is None:
            triu = torch.triu(torch.ones((self.problem_size,  self.problem_size), dtype=torch.bool))
            tril = torch.tril(torch.ones((self.problem_size,  self.problem_size), dtype=torch.bool), diagonal=-1)
            mask = torch.cat([triu, tril], dim=1)
            self.offset = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # self.problems = get_random_problems(batch_size, self.problem_size)
        if not self.vali:
            _, (problems, solutions) = self.dataloader.__next__()
            if self.env_params['data_aug']:
                self.problems = data_aug(problems[:batch_size].float())
            else:
                self.problems = problems[:batch_size].float()
            solutions = torch.stack(solutions, dim=1)[:batch_size]
            self.solutions = self.equal_solutions(solutions, batch_size)
            self.solutions_flipped = self.equal_solutions(torch.flip(solutions, dims=[1]), batch_size)

        else:
            _, self.problems = self.dataloader.__next__()

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

    def equal_solutions(self, ori_solutions, batch_size):
        all_pomo = self.problem_size
        solutions_rep = torch.cat([ori_solutions, ori_solutions], dim=1).unsqueeze(1).expand(-1, all_pomo, -1)
        solutions_rep = solutions_rep[self.offset[:batch_size]].view(batch_size, all_pomo, self.problem_size)

        idx = solutions_rep[:, :, 0].sort(1)[1]
        idx = idx.sort(1)[1]
        idx_rep = idx.unsqueeze(-1).expand_as(solutions_rep)
        empty = - torch.ones_like(solutions_rep)
        equal_solution = torch.scatter(empty, 1, idx_rep, solutions_rep)[:, :self.pomo_size]
        return equal_solution

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

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
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


def read_instance_data_tsp(problem_size,
                           nb_instances,  # 100000
                           offset=0):
    instance_file = './data/tsp_{}_instances.zip'.format(problem_size)
    solution_file = './data/tsp_{}_solutions.zip'.format(problem_size)
    instances = []
    solutions = []
    with zipfile.ZipFile(instance_file) as instance_zip:
        with zipfile.ZipFile(solution_file) as solution_zip:
            instances_list = instance_zip.namelist()
            solutions_list = solution_zip.namelist()
            assert len(instances_list) == len(solutions_list)
            instances_list.sort()
            solutions_list.sort()
            i = offset
            while len(instances) < nb_instances:
                if instances_list[i].endswith('/'):
                    i += 1
                    continue

                #Read instance data
                f = instance_zip.open(instances_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                instance = np.zeros((problem_size, 2))
                ii = 0
                while not lines[ii].startswith("NODE_COORD_SECTION"):
                    ii += 1
                ii += 1
                header_lines = ii
                while ii < len(lines):
                    line = lines[ii]
                    if line == 'EOF':
                        break
                    line = line.replace('\t', " ").split(" ")
                    x = line[1]
                    y = line[2]
                    instance[ii-header_lines] = [x, y]
                    ii += 1

                instance = np.array(instance) / 1000000
                instances.append(instance)

                #Read solution data
                f = solution_zip.open(solutions_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                tour = [int(l) for ll in lines[1:] for l in ll.split(' ')]

                solutions.append(tour)
                i += 1
    return instances, solutions


class TSPDataSet(Dataset):

    def __init__(self, problem_size, nb_instances=50000):
        # 10-fold expand
        instances, solutions = read_instance_data_tsp(problem_size, nb_instances)
        # instances, solutions = read_instance_data_tsp(problem_size, 10)
        self.x = instances * 2
        self.y = solutions * 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


def data_aug(problems):
    batch_size = problems.shape[0]

    problems = problems - 0.5
    theta = torch.atan2(problems[:, :, 1], problems[:, :, 0])
    rho = torch.linalg.norm(problems, dim=2)
    rotation = torch.rand(batch_size) * 2 * math.pi
    # rotation
    theta = theta + rotation.unsqueeze(-1).expand_as(theta)

    # symm
    symmetry = torch.rand(batch_size).unsqueeze(-1).expand_as(theta) > 0.5
    theta[symmetry] = -theta[symmetry]

    # shrink
    rho = rho * (torch.rand_like(problems[:, 0, 0])[:, None].expand_as(rho) * 0.6 + 0.7)

    # recover
    x = rho * torch.cos(theta) + 0.5
    y = rho * torch.sin(theta) + 0.5
    problems = torch.stack([x, y], dim=-1)

    # noise
    problems1 = problems.unsqueeze(1).expand(-1, problems.size(1), -1, -1)
    problems2 = problems.unsqueeze(2).expand(-1, -1, problems.size(1), -1)
    dist = torch.linalg.norm(problems1 - problems2, dim=3)
    # mask 0 in diagonal
    dist[dist == 0] = 500.
    noise_threshold = dist.min(-1)[0].min(-1)[0].unsqueeze(-1).expand_as(x)

    theta_ = torch.rand_like(x) * 2 * math.pi
    rho_ = torch.rand_like(x)
    x_noise = rho_ * torch.cos(theta_)
    y_noise = rho_ * torch.sin(theta_)

    problems = torch.stack([
        x + x_noise * noise_threshold,
        y + y_noise * noise_threshold,
    ], dim=-1)
    return problems