from dataclasses import dataclass
import torch
import tqdm
import numpy as np
from MISProblemDef import get_random_problems, augment_xy_data_by_8_fold
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


class MISEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size_low = env_params['problem_size_low']
        self.problem_size_high = env_params['problem_size_high']
        self.problem_size = env_params['sub_size']
        self.p_value = env_params['p_value']
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

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None):
        if nodes_coords is not None:
            self.raw_problems = nodes_coords[episode]
        else:
            self.raw_problem_size = np.random.randint(self.problem_size_low, self.problem_size_high + 1)
            self.raw_problems = get_random_problems(batch_size, self.raw_problem_size, p=self.p_value)

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

