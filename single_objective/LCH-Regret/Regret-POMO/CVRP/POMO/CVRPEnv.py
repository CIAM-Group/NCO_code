
from dataclasses import dataclass
import torch

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


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

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        # regret
        ####################################
        self.mode = None
        self.last_current_node = None
        self.last_load = None
        self.regret_count = None

        self.regret_mask_matrix = None
        self.add_mask_matrix = None

        self.time_step=0

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

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
        self.selected_count = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.long)
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+2))
        self.visited_ninf_flag[:, :, self.problem_size+1] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+2))
        self.ninf_mask[:, :, self.problem_size+1] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        self.regret_count = torch.zeros((self.batch_size, self.pomo_size))
        self.mode = torch.full((self.batch_size, self.pomo_size), 0)
        self.last_current_node = None
        self.last_load = None
        self.time_step=0

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = 0
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        if self.time_step<4:


            self.time_step=self.time_step+1
            self.selected_count = self.selected_count+1
            self.at_the_depot = (selected == 0)
            if self.time_step==3:
                self.last_current_node = self.current_node.clone()
                self.last_load = self.load.clone()
            if self.time_step == 4:
                self.last_current_node = self.current_node.clone()
                self.last_load = self.load.clone()
                self.visited_ninf_flag[:, :, self.problem_size+1][(~self.at_the_depot)&(self.last_current_node!=0)] = 0

            self.current_node = selected
            self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

            demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
            gathering_index = selected[:, :, None]
            selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
            self.load -= selected_demand
            self.load[self.at_the_depot] = 1  # refill loaded at the depot

            self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
            self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

            self.ninf_mask = self.visited_ninf_flag.clone()
            round_error_epsilon = 0.00001
            demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
            _2=torch.full((demand_too_large.shape[0],demand_too_large.shape[1],1),False)
            demand_too_large = torch.cat((demand_too_large, _2), dim=2)
            self.ninf_mask[demand_too_large] = float('-inf')

            self.step_state.selected_count = self.time_step
            self.step_state.load = self.load
            self.step_state.current_node = self.current_node
            self.step_state.ninf_mask = self.ninf_mask



        else:
            action0_bool_index = ((self.mode == 0) & (selected != self.problem_size + 1))
            action1_bool_index = ((self.mode == 0) & (selected == self.problem_size + 1))  # regret
            action2_bool_index = self.mode == 1
            action3_bool_index = self.mode == 2

            action1_index = torch.nonzero(action1_bool_index)
            action2_index = torch.nonzero(action2_bool_index)

            action4_index = torch.nonzero((action3_bool_index & (self.current_node != 0)))


            self.selected_count = self.selected_count+1
            self.selected_count[action1_bool_index] = self.selected_count[action1_bool_index] - 2

            self.last_is_depot = (self.last_current_node == 0)

            _ = self.last_current_node[action1_index[:, 0], action1_index[:, 1]].clone()
            temp_last_current_node_action2 = self.last_current_node[action2_index[:, 0], action2_index[:, 1]].clone()
            self.last_current_node = self.current_node.clone()
            self.current_node = selected.clone()
            self.current_node[action1_index[:, 0], action1_index[:, 1]] = _.clone()

            self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)

            self.at_the_depot = (selected == 0)
            demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
            # shape: (batch, pomo, problem+1)
            _3 = torch.full((demand_list.shape[0], demand_list.shape[1], 1), 0)
            demand_list = torch.cat((demand_list, _3), dim=2)
            gathering_index = selected[:, :, None]
            # shape: (batch, pomo, 1)
            selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
            _1 = self.last_load[action1_index[:, 0], action1_index[:, 1]].clone()
            self.last_load= self.load.clone()
            # shape: (batch, pomo)
            self.load -= selected_demand
            self.load[action1_index[:, 0], action1_index[:, 1]] = _1.clone()
            self.load[self.at_the_depot] = 1  # refill loaded at the depot

            self.visited_ninf_flag[:, :, self.problem_size+1][self.last_is_depot] = 0
            self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
            self.visited_ninf_flag[action2_index[:, 0], action2_index[:, 1], temp_last_current_node_action2] = float(0)
            self.visited_ninf_flag[action4_index[:, 0], action4_index[:, 1], self.problem_size + 1] = float(0)
            self.visited_ninf_flag[:, :, self.problem_size+1][self.at_the_depot] = float('-inf')
            self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0



            self.ninf_mask = self.visited_ninf_flag.clone()
            round_error_epsilon = 0.00001
            demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
            # shape: (batch, pomo, problem+1)
            self.ninf_mask[demand_too_large] = float('-inf')

            newly_finished = (self.visited_ninf_flag == float('-inf'))[:,:,:self.problem_size+1].all(dim=2)
            # shape: (batch, pomo)
            self.finished = self.finished + newly_finished
            # shape: (batch, pomo)

            self.mode[action1_bool_index] = 1
            self.mode[action2_bool_index] = 2
            self.mode[action3_bool_index] = 0
            self.mode[self.finished] = 4


            self.ninf_mask[:, :, 0][self.finished] = 0
            self.ninf_mask[:, :, self.problem_size+1][self.finished] = float('-inf')

            self.step_state.selected_count = self.time_step
            self.step_state.load = self.load
            self.step_state.current_node = self.current_node
            self.step_state.ninf_mask = self.ninf_mask



        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):

        m1 = (self.selected_node_list==self.problem_size+1)
        m2 = (m1.roll(dims=2, shifts=-1) | m1)
        m3 = m1.roll(dims=2, shifts=1)
        m4 = ~(m2|m3)

        selected_node_list_right = self.selected_node_list.roll(dims=2, shifts=1)
        selected_node_list_right2 = self.selected_node_list.roll(dims=2, shifts=3)

        self.regret_mask_matrix = m1
        self.add_mask_matrix = (~m2)

        travel_distances = torch.zeros((self.batch_size, self.pomo_size))

        for t in range(self.selected_node_list.shape[2]):
            add1_index = (m4[:,:,t].unsqueeze(2)).nonzero()
            add3_index = (m3[:,:,t].unsqueeze(2)).nonzero()

            travel_distances[add1_index[:,0],add1_index[:,1]] = travel_distances[add1_index[:,0],add1_index[:,1]].clone()+((self.depot_node_xy[add1_index[:,0],self.selected_node_list[add1_index[:,0],add1_index[:,1],t],:]-self.depot_node_xy[add1_index[:,0],selected_node_list_right[add1_index[:,0],add1_index[:,1],t],:])**2).sum(1).sqrt()

            travel_distances[add3_index[:,0],add3_index[:,1]] = travel_distances[add3_index[:,0],add3_index[:,1]].clone()+((self.depot_node_xy[add3_index[:,0],self.selected_node_list[add3_index[:,0],add3_index[:,1],t],:]-self.depot_node_xy[add3_index[:,0],selected_node_list_right2[add3_index[:,0],add3_index[:,1],t],:])**2).sum(1).sqrt()



        return travel_distances


