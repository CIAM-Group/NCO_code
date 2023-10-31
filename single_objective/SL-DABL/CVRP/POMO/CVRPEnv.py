
from dataclasses import dataclass
import torch

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold

from torch.utils.data import Dataset, DataLoader
import itertools

import numpy as np
import zipfile

import math

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
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
        self.problem_size = env_params['problem_size']
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
        # shape: (batch, pomo)

        self.solutions = None
        data = self.env_params['data']
        batch_size = self.env_params['batch_size']
        if data is None:
            dataset = CVRPDataSet(self.problem_size)
            self.dataloader0 = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            self.dataloader = itertools.cycle(enumerate(self.dataloader0))
            self.vali = False
        else:
            dataset = data
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            self.dataloader = enumerate(dataloader)
            self.vali = True
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

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.vali:
            _, problems = self.dataloader.__next__()
            problems = problems.cuda()
            depot_node_xy, depot_node_demand = problems[:, :, :2], problems[:, :, 2]
            # self.depot_node_xy, self.depot_node_demand = problems[:, :, :2], problems[:, :, 2]
        else:
            _, (problems, solutions, self.label_cost) = self.dataloader.__next__()
            problems = problems[:batch_size].float().cuda()
            depot_node_xy, depot_node_demand = problems[:, :, :2], problems[:, :, 2]
            if self.env_params['data_aug']:
                depot_node_xy = data_aug(depot_node_xy)
            # self.depot_node_xy = depot_node_xy
            # self.depot_node_demand = depot_node_demand
            self.solutions = solutions[:batch_size].cuda()
            self.demands_sum = depot_node_demand.sum(-1, keepdim=True).expand(-1, self.pomo_size)
            # shape: (batch, pomo)

        depot_xy, node_xy = depot_node_xy[:, 0].unsqueeze(1), depot_node_xy[:, 1:]
        node_demand = depot_node_demand[:, 1:]

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
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
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
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
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done, self.selected_node_list if done else None

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

def read_instance_data_cvrp(problem_size,
                            nb_instances=100000,
                            offset=0):
    instance_file = './data/cvrp_{}_instances.zip'.format(problem_size)
    solution_file = './data/cvrp_{}_solutions.zip'.format(problem_size)
    instances = []
    solutions = []
    all_num_vehicles = []
    label_cost = []
    with zipfile.ZipFile(instance_file) as instance_zip:
        with zipfile.ZipFile(solution_file) as solution_zip:
            # instances_list = instance_zip.namelist()[:-1]
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

                # Read instance data
                f = instance_zip.open(instances_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                ii = 0

                while ii < len(lines):
                    line = lines[ii]
                    if line.startswith("DIMENSION"):
                        dimension = int(line.split(':')[1])
                    elif line.startswith("CAPACITY"):
                        capacity = int(line.split(':')[1])
                    elif line.startswith('NODE_COORD_SECTION'):
                        locations = np.loadtxt(lines[ii + 1:ii + 1 + dimension], dtype=float)
                        ii += dimension
                    elif line.startswith('DEMAND_SECTION'):
                        demand = np.loadtxt(lines[ii + 1:ii + 1 + dimension], dtype=float)
                        ii += dimension
                    elif line.startswith('NAME'):
                        instance_name = int(line.split(':')[1].split('_')[-1])
                    ii += 1

                locations = locations[:, 1:] / 1000000
                demand = demand[:, 1:] / capacity
                instance = torch.from_numpy(np.concatenate((locations, demand), axis=1))
                instances.append(instance)

                # Read solution data
                f = solution_zip.open(solutions_list[i], "r")
                solution = []
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                ii = 0
                while ii < len(lines):
                    line = lines[ii]
                    ii += 1
                    if not line.startswith("Route"):
                        continue
                    line = line.split(':')[1]
                    tour = [int(l) for l in line[1:].split(' ')]
                    # solution.append(tour)
                    solution += [0]
                    solution += tour

                num_vehicles = len(solution) - problem_size
                solution = torch.tensor(solution).cpu()
                solutions.append(solution)
                all_num_vehicles.append(num_vehicles)

                ordered_seq = torch.index_select(instance[:, :2], 0, solution)
                rolled_seq = ordered_seq.roll(dims=0, shifts=-1)
                label_cost.append(torch.linalg.norm(rolled_seq-ordered_seq, dim=1).sum())

                i += 1

    max_num_vehicles = max(all_num_vehicles)
    for i in range(len(solutions)):
        if all_num_vehicles[i] < max(all_num_vehicles):
            solutions[i] = torch.cat([solutions[i],
                                      torch.zeros(max_num_vehicles - all_num_vehicles[i],
                                                  device=torch.device("cpu"),
                                                  dtype=solutions[i].dtype)],
                                     dim=-1)
    return instances, solutions, label_cost

class CVRPDataSet(Dataset):
    def __init__(self, problem_size):
        instances, solutions, label_cost = read_instance_data_cvrp(problem_size, 50000)
        self.instances = instances * 2
        self.solutions = solutions * 2
        self.label_cost = label_cost * 2

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return (self.instances[index],
                self.solutions[index],
                self.label_cost[index])

def data_aug(problems):
    batch_size = problems.shape[0]

    problems = problems - 0.5
    theta = torch.atan2(problems[:, :, 1], problems[:, :, 0])
    rho = torch.linalg.norm(problems, dim=2)
    rotation = torch.rand(batch_size) * 2 * math.pi
    # rotation
    theta = theta + rotation.unsqueeze(-1).expand_as(theta)

    # flip
    symmetry = torch.rand(batch_size).unsqueeze(-1).expand_as(theta) > 0.5
    theta[symmetry] = -theta[symmetry]

    # shrink
    rho = rho * (torch.rand_like(problems[:, 0, 0])[:, None].expand_as(rho) * 0.6 + 0.7)

    # recover
    x = rho * torch.cos(theta) + 0.5
    y = rho * torch.sin(theta) + 0.5
    problems = torch.stack([x, y], dim=-1)

    return problems