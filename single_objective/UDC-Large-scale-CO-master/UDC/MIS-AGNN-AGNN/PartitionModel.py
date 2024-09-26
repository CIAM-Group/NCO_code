import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn


# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=1, units=48, act_fn='silu', agg_fn='mean'):  # TODO feats=1
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
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

    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
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
        return x


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
        self.lin_o = nn.Linear(self.units_list[-1], 1)

    def forward(self, node):
        for i in range(self.depth):
            x = self.lins[i](node)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.softmax(self.lin_o(x).reshape(-1), dim=-1)
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, k_sparse, depth=3, units=48, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        self.k_sparse = k_sparse
        super().__init__([self.units] * depth, act_fn)

    def forward(self, x_emb):
        return super().forward(x_emb)


class PartitionModel(nn.Module):
    def __init__(self, units, feats, k_sparse, edge_feats=1, depth=12):
        super().__init__()
        self.emb_net = EmbNet(depth=depth, units=units, feats=feats, edge_feats=edge_feats)
        self.par_net_heu = ParNet(units=units, k_sparse=k_sparse)

    def forward(self, pyg):
        '''
        Args:
            pyg: torch_geometric.data.Data instance with x, edge_index, and edge attr
        Returns:
            heu: heuristic vector [n_nodes * k_sparsification,]
        '''
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        x_emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(x_emb)
        return heu

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
