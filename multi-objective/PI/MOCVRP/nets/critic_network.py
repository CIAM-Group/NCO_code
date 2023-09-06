from torch import nn
from nets.graph_layers import MultiHeadAttentionLayer, EmbeddingNet
import torch


def represent_idx(tour_info, tour_idx):
    tour_idx_rep = tour_idx.unsqueeze(1).repeat(1, tour_info.shape[1], 1)
    mid = torch.gather(tour_info, 2, tour_idx_rep)

    pre = torch.cat((mid[:, :-1, 0][:, :, None], mid[:, :-1, :-1]), dim=2)
    post = torch.cat((mid[:, :-1, 1:], (mid[:, :, -1][:, :-1, None])), dim=2)

    demands = mid[:, -1, :][:, None, :]

    represented = torch.cat((pre, mid[:, :-1, :], post, demands), dim=1).permute((0, 2, 1))
    return represented, mid


class CriticNetwork(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 device,
                 best_incumbent=False
                 ):

        super(CriticNetwork, self).__init__()

        self.best_incumbent = best_incumbent

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.device = device

        # Problem specific placeholders
        if self.problem.NAME == 'tsp':
            self.node_dim = 2  # x, y
        elif self.problem.NAME == 'cvrp':
            self.node_dim = 7  # x, y
        else:
            assert False, "Unsupported problem: {}".format(self.problem.NAME)

        # networks
        self.embedder = EmbeddingNet(
            self.node_dim,
            self.embedding_dim,
            self.device)

        self.encoder = nn.Sequential(*(
            MultiHeadAttentionLayer(self.n_heads,
                                    self.embedding_dim,
                                    self.hidden_dim,
                                    self.normalization)
            for _ in range(self.n_layers)))

        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, solutions, n_vec):
        """
        :param inputs: (x, graph_size, input_dim)
        :return:
        """
        # pass through embedder
        bs, gs = solutions.size()
        x_r = represent_idx(x, solutions)[0].unsqueeze(1).expand(-1, 2, -1, -1)
        x_tmp = self.embedder(x_r, n_vec[..., None, None].expand(-1, 2, gs, 1), solutions)
        bs, os, gs, em_d = x_tmp.size()
        x_embed = x_tmp.contiguous().view(-1, gs, em_d)
        t_em = self.encoder(x_embed)
        h_em = t_em.contiguous().view(-1, os, gs, em_d).sum(1)
        max_pooling = h_em.max(1)[0]  # max Pooling
        graph_feature = self.project_graph(max_pooling)[:, None, :]
        node_feature = self.project_node(h_em)
        fusion = node_feature + graph_feature.expand_as(node_feature)  # torch.Size([2, 50, 128])
        value = self.value_head(fusion.mean(dim=1))

        return value
