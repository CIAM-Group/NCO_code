import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    last: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    selected_count: torch.Tensor
    regret_count: torch.Tensor
    mode: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        mask = torch.zeros(batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device)
        mask[:, :, -1] = 1
        return StateTSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            last=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                mask  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            selected_count=torch.zeros((batch_size, 1), dtype=torch.long, device=loc.device),
            regret_count=torch.zeros((batch_size, 1), dtype=torch.long, device=loc.device),
            mode=torch.full((batch_size, 1), 1, device=loc.device)
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        problem_size = self.loc.size(1)
        # Update the state
        action0_bool_index = ((self.mode == 0) & (selected[:, None] != problem_size))
        action1_bool_index = ((self.mode == 0) & (selected[:, None] == problem_size))  # regret
        action2_bool_index = self.mode == 1
        action3_bool_index = self.mode == 2
        # Add dimension for step
        action0_index = torch.nonzero(action0_bool_index)
        action1_index = torch.nonzero(action1_bool_index)
        action2_index = torch.nonzero(action2_bool_index)
        action3_index = torch.nonzero(action3_bool_index)

        self.selected_count[action0_bool_index | action2_bool_index | action3_bool_index] = self.selected_count[
                                                                                                action0_bool_index | action2_bool_index | action3_bool_index] + 1
        self.selected_count[action1_bool_index] = self.selected_count[action1_bool_index] - 1
        self.regret_count[action1_bool_index] = self.regret_count[action1_bool_index] + 1
        done = (self.selected_count == self.loc.size(1))
        self.mode[action1_bool_index] = 1
        self.mode[action2_bool_index] = 2
        self.mode[action3_bool_index] = 0
        self.mode[done] = 3
        done_idex = torch.nonzero(done)

        if self.i <= 1:
            prev_a = selected[:, None]
        else:
            _ = self.last[action1_index[:, 0], action1_index[:, 1]].clone()
            temp_last_current_node_action2 = self.last[action2_index[:, 0], action2_index[:, 1]].clone()
            prev_a = selected[:, None].clone()
            prev_a[action1_index[:, 0], action1_index[:, 1]] = _.clone()

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_
            visited_[action0_index[:, 0], 0, prev_a[action0_index[:, 0], 0]] = 1
            visited_[action1_index[:, 0], 0, selected[:, None][action1_index[:, 0], 0]] = 1
            visited_[action2_index[:, 0], 0, prev_a[action2_index[:, 0], 0]] = 1
            if self.i > 1:
                visited_[action2_index[:, 0], 0, temp_last_current_node_action2] = 0
            visited_[action3_index[:, 0], 0, prev_a[action3_index[:, 0], 0]] = 1
            visited_[action3_index[:, 0], 0, -1] = 0
            visited_[done_idex[:, 0], 0, prev_a[done_idex[:, 0], 0]] = 0
            visited_[done_idex[:, 0], 0, -1] = 1

        # visited_[:, 0, -1] = 1
        return self._replace(first_a=first_a, prev_a=prev_a, last=self.prev_a, visited_=visited_,
                             i=self.i + 1, selected_count=self.selected_count,
                             regret_count=self.regret_count, mode=self.mode)

    def all_finished(self):
        # Exactly n steps
        return (self.selected_count == self.loc.size(1)).all()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
                self.dist[
                    self.ids,
                    self.prev_a
                ] +
                self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
