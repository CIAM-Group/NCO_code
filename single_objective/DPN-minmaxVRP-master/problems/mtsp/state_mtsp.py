import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class State_MTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    agent_idx: torch.Tensor  # order of agents are currently moving
    agent_per: torch.Tensor  # order of agents are currently moving
    prev_a: torch.Tensor  # Previous actions
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor  # Keeps track of tour lengths corresponding to each agent
    cur_coord: torch.Tensor  # Keeps track of current coordinates
    i: torch.Tensor  # Keeps track of step
    count_depot: torch.Tensor  # Number of depot visited
    left_city: torch.Tensor  # Number of left cities
    depot_distance: torch.Tensor  # Distance from depot to all cities
    remain_max_distance: torch.Tensor  # Max distance from depot among left cities
    max_distance: torch.Tensor  # Max distance from depot among all cities

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(loc, agent_num, agent_per, visited_dtype=torch.uint8):

        # In mtsp the cities is start from index 1
        left_city = loc[:, 1:, :].size(1)
        loc = torch.cat((loc[:, :1, :].repeat(1, agent_num-1, 1), loc), 1)
        batch_size, n_loc, _ = loc.size()
        pomo_size = agent_per.size(0)
        loc = loc[:, None, :, :].expand(-1, pomo_size, -1, -1)
        prev_a = agent_per[None, :, 0:1].expand(batch_size, -1, -1)

        depot_distance = torch.cdist(loc, loc, p=2)
        depot_distance = depot_distance[:, :, 0, :]
        max_distance = depot_distance.max(dim=-1, keepdim=True)[0]

        return State_MTSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            agent_idx=agent_per[None, :, 0:1].expand(batch_size, -1, -1),
            agent_per=agent_per,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, pomo_size, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
            ),
            lengths=torch.zeros(batch_size, pomo_size, agent_num, device=loc.device),
            cur_coord=loc[:, :, 0, :][:, :, None, :],
            count_depot=torch.zeros(batch_size, pomo_size, 1, dtype=torch.int64, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            left_city=left_city * torch.ones(batch_size, pomo_size, 1, dtype=torch.long, device=loc.device),
            remain_max_distance=max_distance,
            max_distance=max_distance,
            depot_distance=depot_distance
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths

    def update(self, selected):

        # Update the state

        prev_a = selected[:, :, None]  # Add dimension for step

        # City idices starts from agent_num + 1
        is_city = prev_a >= self.lengths.size(2)

        self.left_city[is_city] -= 1

        # If agent move to other city, then, the distance between visited city and depot is 0
        depot_distance = self.depot_distance.scatter(-1, prev_a, 0)
        remain_max_distance = self.depot_distance.max(dim=-1, keepdim=True)[0]

        cur_coord = self.loc.gather(2, prev_a[..., None].expand(-1, -1, -1, 2))

        path_lengths = (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        lengths = self.lengths.scatter_add(-1, self.count_depot, path_lengths)

        # Current agent comes back to depot when it selects its own index
        self.count_depot[selected[:, :, None] == self.agent_idx] += torch.ones(self.count_depot[selected[:, :, None] == self.agent_idx].shape, dtype=torch.int64,
                                                                               device=self.count_depot.device)
        agent_idx = self.agent_idx
        # agent_idx is added by 1 if the current agent comes back to depot
        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a, 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        if ((is_city == False).all() and (self.count_depot == self.agent_per.size(1)).all()):
            return self._replace(agent_idx=agent_idx, prev_a=prev_a, visited_=visited_,
                                 lengths=lengths, cur_coord=cur_coord, i=self.i + 1, depot_distance=depot_distance, remain_max_distance=remain_max_distance)

        agent_idx = self.agent_per[None, :, :].expand(self.count_depot.size(0), -1, -1).gather(2, self.count_depot)

        return self._replace(agent_idx=agent_idx, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1, depot_distance=depot_distance, remain_max_distance=remain_max_distance)

    def all_finished(self):
        return self.visited.all()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):

        visited_loc = self.visited_
        agent_num = self.lengths.size(2)  # number of agent
        mask_loc = self.visited_.clone()
        mask_loc[:, :, :agent_num] = 1
        agent_idx = self.agent_idx
        mask_loc = mask_loc.scatter_(-1, agent_idx, 0)
        condition = ((self.count_depot == agent_num - 1).squeeze(-1) & ((visited_loc[:, :, agent_num:] == 0).sum(dim=-1) != 0))
        src = torch.ones_like(condition, dtype=torch.uint8) * condition
        mask_loc = mask_loc.scatter_(-1, agent_idx, src[:, :, None])

        return mask_loc > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
