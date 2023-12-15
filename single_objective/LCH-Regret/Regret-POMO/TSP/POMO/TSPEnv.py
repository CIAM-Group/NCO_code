
from dataclasses import dataclass
import torch
import pickle

from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold


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
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        self.last_current_node = None
        self.mode=None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        self.regret_mask_matrix=None
        self.add_mask_matrix=None

        self.use_load_data=False
        self.data=None
        self.offset=0

        if 'test_data_load' in env_params:
            if env_params['test_data_load']['enable']:
                self.use_load_data=True
                filename = env_params['test_data_load']['filename']
                with open(filename, 'rb') as f:
                    self.data = pickle.load(f)
                self.data = self.data.cuda(0)
                self.offset = 0


    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.use_load_data:
            self.problems = self.data[self.offset:self.offset + batch_size]
            self.offset += batch_size
        else:
            self.problems = get_random_problems(batch_size, self.problem_size)
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
        self.selected_count = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.long)
        self.mode=torch.full((self.batch_size, self.pomo_size),1)
        self.current_node = None
        self.last_current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size+1))
        self.step_state.ninf_mask[:, :,self.problem_size] = float('-inf')
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

        action0_bool_index=((self.mode==0) & (selected!=self.problem_size))
        action1_bool_index=((self.mode==0) & (selected==self.problem_size)) # regret
        action2_bool_index=self.mode==1
        action3_bool_index=self.mode==2

        action0_index = torch.nonzero(action0_bool_index)
        action1_index = torch.nonzero(action1_bool_index)
        action2_index = torch.nonzero(action2_bool_index)
        action3_index = torch.nonzero(action3_bool_index)

        first_step=self.current_node is None
        second_step = (self.current_node is not None) and (self.last_current_node is None)

        # 1 change self.current_node and self.last_current_node
        if first_step:
            self.current_node = selected
        elif second_step:
            self.last_current_node=self.current_node.clone()
            self.current_node = selected
        else:
            _ = self.last_current_node[action1_index[:, 0], action1_index[:, 1]].clone()
            temp_last_current_node_action2 = self.last_current_node[action2_index[:, 0], action2_index[:, 1]].clone()
            self.last_current_node=self.current_node.clone()
            self.current_node = selected.clone()
            self.current_node[action1_index[:, 0], action1_index[:, 1]] = _.clone()


        # 2 change self.step_state.ninf_mask
        # action0
        self.step_state.ninf_mask[action0_index[:, 0], action0_index[:, 1], self.current_node[action0_index[:, 0], action0_index[:, 1]]] = float('-inf')
        # action1
        self.step_state.ninf_mask[action1_index[:, 0], action1_index[:, 1], selected[action1_index[:, 0], action1_index[:, 1]]] = float('-inf')
        # action2
        self.step_state.ninf_mask[action2_index[:, 0], action2_index[:, 1], self.current_node[action2_index[:, 0], action2_index[:, 1]]] = float('-inf')
        if not (first_step or second_step):
            self.step_state.ninf_mask[action2_index[:, 0], action2_index[:, 1], temp_last_current_node_action2] = float(0)
        # action3
        self.step_state.ninf_mask[action3_index[:, 0], action3_index[:, 1], self.current_node[action3_index[:, 0], action3_index[:, 1]]] = float('-inf')
        self.step_state.ninf_mask[action3_index[:, 0], action3_index[:, 1], self.problem_size] = float(0)

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)

        # 4 change self.selected_count
        self.selected_count[action0_bool_index | action2_bool_index | action3_bool_index] = self.selected_count[action0_bool_index | action2_bool_index | action3_bool_index] +1
        self.selected_count[action1_bool_index] = self.selected_count[action1_bool_index] -1

        ####################
        # done: shape: (batch, pomo)
        done = (self.selected_count == self.problem_size)

        # 5 change self.mode
        self.mode[action1_bool_index] = 1
        self.mode[action2_bool_index] = 2
        self.mode[action3_bool_index] = 0
        self.mode[done] = 3

        done_idex= torch.nonzero(done)

        self.step_state.ninf_mask[done_idex[:, 0], done_idex[:, 1], self.current_node[done_idex[:, 0], done_idex[:, 1]]] = float(0)
        self.step_state.ninf_mask[done_idex[:, 0], done_idex[:, 1], self.problem_size] = float("-inf")

        self.step_state.current_node = self.current_node


        # returning values
        if done.all():
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done.all()

    def _get_travel_distance(self):

        m1 = (self.selected_node_list==self.problem_size)
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

            travel_distances[add1_index[:,0],add1_index[:,1]] = travel_distances[add1_index[:,0],add1_index[:,1]].clone()+((self.problems[add1_index[:,0],self.selected_node_list[add1_index[:,0],add1_index[:,1],t],:]-self.problems[add1_index[:,0],selected_node_list_right[add1_index[:,0],add1_index[:,1],t],:])**2).sum(1).sqrt()

            travel_distances[add3_index[:,0],add3_index[:,1]] = travel_distances[add3_index[:,0],add3_index[:,1]].clone()+((self.problems[add3_index[:,0],self.selected_node_list[add3_index[:,0],add3_index[:,1],t],:]-self.problems[add3_index[:,0],selected_node_list_right2[add3_index[:,0],add3_index[:,1],t],:])**2).sum(1).sqrt()


        return travel_distances


