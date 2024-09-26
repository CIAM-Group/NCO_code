import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn


class RotatePostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(RotatePostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
        # (output_dim//2)
        ids = torch.arange(0, d_model // 2, dtype=torch.float)
        theta = torch.pow(1000, -2 * ids / d_model)

        # (max_len, output_dim//2)
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        self.cos_embeddings = torch.sin(embeddings)
        self.sin_embeddings = torch.cos(embeddings)

    def forward(self, input):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, emb_size = input.size()
        cos_pos = self.cos_embeddings[None, :seq_len, :].repeat_interleave(2, dim=-1).to(input.device)
        sin_pos = self.sin_embeddings[None, :seq_len, :].repeat_interleave(2, dim=-1).to(input.device)

        # q,k: (bs, head, max_len, output_dim)
        input2 = torch.stack([-input[..., 1::2], input[..., ::2]], dim=-1)
        input2 = input2.reshape(input.shape)

        output = input * cos_pos + input2 * sin_pos
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
        return output


# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=1, units=48, act_fn='silu', agg_fn='mean'):  # TODO feats=1
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.RoPE = RotatePostionalEncoding(self.units, 10000)
        self.pos_emb_proj = nn.Sequential(nn.Linear(self.units, self.units, bias=False))
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_linR = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, nodes, edge_index, edge_attr, M):
        x = nodes.clone()
        w = edge_attr
        x = self.v_lin0(x)
        place_embedding = self.v_linR(nodes[:M, :])
        positional_embedding = self.RoPE(place_embedding[None, :])[0]
        positional_embedding = self.alpha * self.pos_emb_proj(positional_embedding)
        x[:M, :] += positional_embedding
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        self.lin_f = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        self.lin_l = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        self.lin_v = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, node, x, solution, visited, k_sparse):
        sample_size = solution.size(0)
        x_first = node.clone()[None, :].repeat(solution.size(0), 1, 1).gather(1, solution[:, 0][:, None, None].expand(-1, -1, node.size(-1)))
        x_last = node.clone()[None, :].repeat(solution.size(0), 1, 1).gather(1, solution[:, 1][:, None, None].expand(-1, -1, node.size(-1)))
        x_visited = node.clone()
        for i in range(self.depth):
            x = self.lins[i](x)
            x_first = self.lin_f[i](x_first)
            x_last = self.lin_l[i](x_last)
            x_visited = self.lin_v[i](x_visited)
            if i < self.depth - 1:
                x = self.act_fn(x)
                x_first = self.act_fn(x_first)
                x_last = self.act_fn(x_last)
                x_visited = self.act_fn(x_visited)
            else:
                q = x_visited[None, :].repeat(solution.size(0), 1, 1) * visited[:, :, None].expand(-1, -1, node.size(-1))
                q = q.sum(1) / visited[:, :, None].expand(-1, -1, node.size(-1)).sum(1)
                q += x_first.squeeze(1) + x_last.squeeze(1)
                x = torch.mm(q, x.T).reshape(sample_size, -1, k_sparse)
                x = torch.softmax(x, dim=-1)
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, k_sparse, depth=3, units=48, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        self.k_sparse = k_sparse
        super().__init__([self.units] * depth, act_fn)

    def forward(self, x_emb, e_emb, solution, visited):
        return super().forward(x_emb, e_emb, solution, visited, self.k_sparse).squeeze(dim=-1)


class PartitionModel(nn.Module):
    def __init__(self, units, feats, k_sparse, edge_feats=1, depth=12):
        super().__init__()
        self.emb_net = EmbNet(depth=depth, units=units, feats=feats, edge_feats=edge_feats)
        self.par_net_heu = ParNet(units=units, k_sparse=k_sparse)
        self.x_emb = None

    def pre(self, pyg, M):
        '''
        Args:
            pyg: torch_geometric.data.Data instance with x, edge_index, and edge attr
        Returns:
            heu: heuristic vector [n_nodes * k_sparsification,]
        '''
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        self.x_emb, self.emb = self.emb_net(x, edge_index, edge_attr, M)

    def forward(self, solution=None, visited=None):
        '''
        Args:
            pyg: torch_geometric.data.Data instance with x, edge_index, and edge attr
        Returns:
            heu: heuristic vector [n_nodes * k_sparsification,]
        '''
        solution_cat = torch.cat((solution[:, 0].unsqueeze(-1), solution[:, -1].unsqueeze(-1)), dim=-1)
        heu = self.par_net_heu(self.x_emb, self.emb, solution_cat, visited)
        return self.x_emb, heu

    @staticmethod
    def reshape(pyg, vector):
        '''Turn heu vector into matrix with zero padding
        '''
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(vector.size(0), n_nodes, n_nodes), device=device)
        idx = torch.repeat_interleave(torch.arange(vector.size(0)).to(device), repeats=pyg.edge_index[0].shape[0])
        idx0 = pyg.edge_index[0].repeat(vector.size(0))
        idx1 = pyg.edge_index[1].repeat(vector.size(0))
        matrix[idx, idx0, idx1] = vector.view(-1)
        try:
            assert (matrix.sum(dim=2) >= 0.99).all()
        except:
            torch.save(matrix, './error_reshape.pt')
        return matrix
