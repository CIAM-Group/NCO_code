import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    last: torch.Tensor
    selected_count: torch.Tensor
    regret_count: torch.Tensor
    mode: torch.Tensor
    done: torch.Tensor

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        batch_size, n_loc, _ = loc.size()
        mask = torch.zeros(batch_size, 1, n_loc + 2, dtype=torch.uint8, device=loc.device)
        mask[:, :, -1] = 1
        return StateCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                mask
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps

            # 就下面这部分和mask不一样
            # TODO：可能要改，一开始应该是NONE，不过应该不影响
            last=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            selected_count=torch.zeros((batch_size, 1), dtype=torch.long, device=loc.device),
            regret_count=torch.zeros((batch_size, 1), dtype=torch.long, device=loc.device),
            # TODO：已改，从1改为0
            mode=torch.full((batch_size, 1), 0, device=loc.device),  # Vector with length num_steps,
            # TODO：已改，从long类型改成bool类型
            done=torch.zeros((batch_size, 1), dtype=torch.bool, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"
        problem_size = self.demand.size(1)
        n_loc = self.demand.size(-1)
        # Update the state

        action0_bool_index = ((self.mode == 0) & (selected[:, None] != problem_size + 1))  # 正常选择
        action1_bool_index = ((self.mode == 0) & (selected[:, None] == problem_size + 1))  # regret
        action2_bool_index = self.mode == 1  # regret 后一步 释放last last node mask
        action3_bool_index = self.mode == 2  # regret 后两步 释放regret mask
        # Add dimension for step
        action0_index = torch.nonzero(action0_bool_index)
        action1_index = torch.nonzero(action1_bool_index)
        action2_index = torch.nonzero(action2_bool_index)
        action3_index = torch.nonzero(action3_bool_index)

        selected_count = self.selected_count + 1
        selected_count[action1_bool_index] = selected_count[action1_bool_index] - 2

        regret_count = self.regret_count
        regret_count[action1_bool_index] = regret_count[action1_bool_index] + 1

        last_depot = self.last == 0
        last_index = torch.nonzero(last_depot)

        select_nondepot = selected != 0
        select_ndindex = torch.nonzero(select_nondepot)

        select_depot = selected == 0
        select_index = torch.nonzero(select_depot)

        curr_depot = self.prev_a == 0
        curr_index = torch.nonzero(curr_depot)

        # time step
        # selected_node_list
        # load
        # mask
        # last/current_node
        # finish
        # mode

        #############################################################################
        # load
        selected_demand = self.demand[self.ids, torch.clamp(selected[:, None] - 1, 0, n_loc - 1)]
        if self.i > 0:
            regret_load = self.demand[action1_index[:, 0], self.prev_a[action1_index[:, 0], action1_index[:, 1]]-1]
            selected_demand[action1_index[:, 0], 0] = -1 * regret_load
        used_capacity = (self.used_capacity + selected_demand) * (selected[:, None] != 0).float()

        #############################################################################
        # mask
        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value

            visited_ = self.visited_.scatter(-1, selected[:, None][:, :, None], 1)
            if self.i > 1:
                visited_[action2_index[:, 0], 0, self.last[action2_index[:, 0], action2_index[:, 1]]] = 0
            visited_[action3_index[:, 0], 0, -1] = 0
            if self.i == 1:
                visited_[:, 0, -1] = 0
            if self.i > 1:
                visited_[last_index[:, 0], 0, -1] = 0
            visited_[select_index[:, 0], 0, -1] = 1
            if self.i > 0:
                visited_[curr_index[:, 0], 0, -1] = 1
            visited_[select_ndindex[:, 0], 0, 0] = 0

        # TODO:超出capacity的mask
        # TODO:只有两个选择的时候

        #############################################################################
        # last/current_node
        prev_a = selected[:, None].clone()
        prev_a[action1_index[:, 0], action1_index[:, 1]] = self.last[action1_index[:, 0], action1_index[:, 1]].clone()


        #############################################################################
        # finish
        done = self.done + visited_[:, :, :-1].all(-1)
        done_index = torch.nonzero(done)
        visited_[done_index[:, 0], 0, 0] = 0
        visited_[done_index[:, 0], 0, -1] = 1

        #############################################################################
        # mode
        self.mode[action1_bool_index] = 1
        self.mode[action2_bool_index] = 2
        self.mode[action3_bool_index] = 0
        self.mode[done_index[:, 0], done_index[:, 1]] = 4

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_, last=self.prev_a, i=self.i + 1, selected_count=selected_count,
            regret_count=regret_count, mode=self.mode, done=done.bool()
        )

    def all_finished(self):
        return self.done.bool().all()

    def get_finished(self):
        return self.done

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        cap_regret = torch.zeros_like(exceeds_cap[..., -1]).unsqueeze(-1)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | torch.cat((exceeds_cap, cap_regret), dim=-1)

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        mask = torch.cat((mask_depot[:, :, None], mask_loc), -1)

        only = mask[:, :, 1:-1].all(-1)
        only_index = torch.nonzero(only)
        mask[only_index[:, 0], 0, -1] = 1
        return mask

    def construct_solutions(self, actions):
        return actions
