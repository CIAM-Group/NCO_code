import torch
from torch import nn
import torch.nn.functional as F
from nets.graph_layers import MultiHeadAttentionLayer, MultiHeadSelDecoder


class EmbeddingNet(nn.Module):

    def __init__(self, node_dim, embedding_dim, device):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.embedder = nn.Linear(node_dim, embedding_dim)

    def forward(self, x):
        ans = self.embedder(x)
        return ans


class Sel_baseline(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_heads, n_layers, num_obj, normalization, device):
        super(Sel_baseline, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.normalization = normalization
        self.device = device
        self.population_size = 100
        self.node_dim = num_obj
        self.encoder = nn.Sequential(
            nn.Linear(self.population_size, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.graph_embedder = nn.Sequential(
            nn.Linear(2, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.embedderf = nn.Sequential(
            nn.Linear(self.node_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.embedder = nn.Linear(self.node_dim, embedding_dim)
        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, batch, f, vec):
        bs, num, os = f.size()
        outdim = vec.size(-1)
        f_act = vec.clone().transpose(1, -1)
        graph_embed = self.graph_embedder(batch).sum(1).sum(1).unsqueeze(1).repeat(1, outdim, 1)
        f_embed = self.embedderf(f).sum(1).unsqueeze(1).repeat(1, outdim, 1)
        h_em = self.encoder(f_act)
        max_pooling = h_em.max(1)[0]
        graph_feature = self.project_graph(max_pooling)[:, None, :]
        node_feature = self.project_node(h_em)
        fusion = node_feature + graph_feature.expand_as(node_feature) + graph_embed + f_embed
        ans = self.decoder(fusion.mean(1))
        return ans.detach().squeeze(), ans.squeeze()
