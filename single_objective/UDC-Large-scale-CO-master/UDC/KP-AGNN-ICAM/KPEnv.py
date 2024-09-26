from dataclasses import dataclass
import torch
import tqdm
import numpy as np
from KProblemDef import get_random_problems, augment_xy_data_by_8_fold
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


class KPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size_low = env_params['problem_size_low']
        self.problem_size_high = env_params['problem_size_high']
        self.problem_size = env_params['sub_size']
        self.sample_size = env_params['sample_size']
        self.pomo_size = env_params['pomo_size']
        self.capacity = None
        # Const @Load_Problem
        ####################################
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, 2)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)
        self.finished = None
        self.capacity = None
        self.accumulated_value_obj = None
        self.items_and_a_dummy = None
        self.item_data = None

        self.ninf_mask_w_dummy = None
        self.ninf_mask = None
        self.fit_ninf_mask = None

        self.dist = None

        self.FLAG__use_saved_problems = False
        self.saved_problems = None
        self.saved_index = 0
        self.optimal = 0
        self.device = None

        self.step_state = None
        self.reset_state = None

        self.problems_values = None
        self.problems_weights = None
        self.problems_unit_values = None

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None, capacity=None):
        if nodes_coords is not None:
            self.raw_problems = nodes_coords[episode:episode + batch_size]
            self.capacitys = capacity[episode:episode + batch_size]
        else:
            self.raw_problem_size = np.random.randint(self.problem_size_low // self.problem_size, self.problem_size_high // self.problem_size + 1) * self.problem_size
            self.raw_problems, self.capacitys = get_random_problems(batch_size, self.raw_problem_size)

    def load_problems(self, batch_size, subp, capacity, aug_factor=1):
        self.batch_size = batch_size

        self.problems = subp
        self.capacity = capacity[:, None].repeat(1, self.pomo_size)
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

        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        # KP
        ###################################

        self.items_and_a_dummy = torch.zeros(self.batch_size, self.problem_size + 1, 2)  # the last item is a dummy item
        self.items_and_a_dummy[:, :self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, :self.problem_size, :]  # shape: (batch, problem, 2)

        self.accumulated_value_obj = torch.zeros(self.batch_size, self.pomo_size)
        # shape: (batch, pomo)

        self.ninf_mask_w_dummy = torch.zeros(self.batch_size, self.pomo_size, self.problem_size + 1)  # the last item is a dummy item
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :self.problem_size]
        # shape: (batch, pomo, problem)

        self.fit_ninf_mask = None
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.current_node = self.current_node  # default None
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        unfit_bool = (self.capacity[:, :, None] - self.item_data[:, None, :, 0]) + 1e-5 < 0
        self.fit_ninf_mask = self.ninf_mask.clone()
        self.fit_ninf_mask[unfit_bool] = float('-inf')
        self.finished = (self.fit_ninf_mask == float('-inf')).all(dim=2)

        self.fit_ninf_mask[self.finished[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size)] = 0
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.fit_ninf_mask
        self.step_state.capacity = self.capacity
        self.step_state.finished = self.finished

        # Additional variables for ICAM
        ####################################
        log_scale = math.log2(self.problem_size)
        # calculate the sum between the unit values of each pair of items
        # i.e., (v_i + v_j) / (w_i + w_j)
        problems_weights = self.problems[:, :, 0]
        # shape: (batch, problem)
        problems_values = self.problems[:, :, 1]
        # shape: (batch, problem)
        self.problems_unit_values = (problems_values / (problems_weights + 1e-10)).unsqueeze(-1)
        # shape: (batch, problem, 1)
        value_addition = problems_values[:, :, None] + problems_values[:, None, :]
        # value_addition = problems_values[:, None, :] + problems_values[:, :, None] #another way to calculate the sum of values
        weight_addition = (problems_weights[:, :, None] + problems_weights[:, None, :]) + 1e-10  # add a small value to avoid division by zero
        # shape: (batch, problem, problem)
        dist_original = value_addition / weight_addition  # shape: (batch, problem, problem)
        max_dist = dist_original.max(-1)[0].unsqueeze(-1)  # shape: (batch, problem, 1)
        dist = (dist_original / max_dist) - 1  # change the range from [0, 1] to [-1, 0]
        # shape: (batch, problem, problem)
        dummy_dist = torch.zeros(size=(self.batch_size, 1, self.problem_size), dtype=torch.float32)
        self.dist = torch.cat((dist, dummy_dist), dim=1)  # shape: (batch, problem+1, problem)

        self.reset_state = Reset_State(problems=self.problems, dist=self.dist, log_scale=log_scale)
        ####################################
        # end of ICAM
        ####################################

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~max_num)

        # Status
        ####################################
        items_mat = self.items_and_a_dummy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, 2)
        # shape: (batch, pomo, problem+1, 2)
        gathering_index = selected[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 2)
        # shape: (batch, pomo, 1, 2)
        selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # selected_item.shape: (batch, pomo, 2) weight and value

        # index 0: weight, index 1: value
        self.accumulated_value_obj += selected_item[:, :, 1]
        self.capacity -= selected_item[:, :, 0]

        assert (self.capacity + 1e-5 > 0).all(), "The sum of the selected items' weight should not exceed the capacity"

        self.ninf_mask_w_dummy[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :self.problem_size]

        # 判断是否存在节点重量大于背包容量,存在则将其置位负无穷
        unfit_bool = (self.capacity[:, :, None] - self.item_data[:, None, :, 0]) + 1e-5 < 0
        self.fit_ninf_mask = self.ninf_mask.clone()
        self.fit_ninf_mask[unfit_bool] = float('-inf')
        # shape: (batch, pomo, problem)

        self.finished = (self.fit_ninf_mask == float('-inf')).all(dim=2)

        self.fit_ninf_mask[self.finished[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size)] = 0
        # do not mask finished episode

        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.fit_ninf_mask
        self.step_state.capacity = self.capacity
        self.step_state.finished = self.finished

        reward = None
        done = self.finished.all()
        if done:
            reward = self.accumulated_value_obj
            # shape: (batch, pomo)
            # self._check_solution_validity()

        return self.step_state, reward, done

    def get_local_feature(self):
        # dist.shape: (batch, problem, problem)
        # current_node.shape: (batch, pomo)
        if self.current_node is None:
            return torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size), dtype=torch.float32)

        current_node = self.current_node[:, :, None, None].expand(-1, -1, 1, self.problem_size)
        # shape: (batch, pomo, 1, problem)

        dist_expand = self.dist[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, problem)

        cur_dist = dist_expand.gather(dim=2, index=current_node).squeeze(2)
        # shape: (batch, pomo, problem)

        return cur_dist

    def make_dataset(self, filename, episode):

        loaded_dict = torch.load(filename, map_location=self.device)
        saved_problems = loaded_dict['problems'][:episode].cuda()
        # shape: (batch, problem, 2)
        # self.optimal = loaded_dict['optimal'][:episode]
        batch_size = saved_problems.size(0)
        if saved_problems.size(1) == 500:
            capacity = 50
        if saved_problems.size(1) == 1000:
            capacity = 100
        if saved_problems.size(1) == 2000:
            capacity = 200
        if saved_problems.size(1) == 5000:
            capacity = 500
        if saved_problems.size(1) == 10000:
            capacity = 1000
        capacity = torch.ones(batch_size, saved_problems.size(1), device=self.device) * capacity
        # shape: (batch, pomo)

        return saved_problems, capacity

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
        return (problems[:, :, 1] * solution).sum(1)

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
