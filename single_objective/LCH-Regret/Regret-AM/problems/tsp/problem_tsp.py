from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):
    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def _get_travel_distance(dataset, pi):
        # 逐步构建
        # 从t=1时间步开始
        # self.selected_node_list # shape: (batch, pomo, selected_node_list_length)
        # self.problem # shape: (batch, problem, 2)

        # 先获取当前时间节点的index
        # 然后获取上个时间节点的index
        torch.set_printoptions(profile="full")

        m1 = (pi == dataset.size(1))  # regret的那一步
        m2 = (m1.roll(dims=-1, shifts=-1) | m1)  # regret的那一步和后悔掉的那一步 路径长度都不加
        m3 = m1.roll(dims=-1, shifts=1)  # regret后重新选择的那一步 加相对前面3个t步的节点的距离
        m4 = ~(m2 | m3)  # 加相对前面1个t步的节点的距离

        selected_node_list_right = pi.roll(dims=-1, shifts=1)
        selected_node_list_right2 = pi.roll(dims=-1, shifts=3)

        travel_distances = torch.zeros(dataset.size(0), device=pi.device)
        for t in range(pi.shape[1]):
            add1_index = torch.nonzero(m4[:, t].unsqueeze(-1))
            add3_index = torch.nonzero(m3[:, t].unsqueeze(-1))
            travel_distances[add1_index[:, 0]] = travel_distances[add1_index[:, 0]].clone() + (
                    (dataset[add1_index[:, 0], pi[add1_index[:, 0], t], :] - dataset[add1_index[:, 0], selected_node_list_right[add1_index[:, 0], t],
                                                                             :]) ** 2).sum(1).sqrt()
            travel_distances[add3_index[:, 0]] = travel_distances[add3_index[:, 0]].clone() + (
                    (dataset[add3_index[:, 0], pi[add3_index[:, 0], t], :] - dataset[add3_index[:, 0], selected_node_list_right2[add3_index[:, 0], t],
                                                                             :]) ** 2).sum(1).sqrt()

        return travel_distances, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
