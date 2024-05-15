import torch
import numpy as np
from torch import nn
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

from nets.positional_encoding import RotatePostionalEncoding

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, sharp=False):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    if sharp == False:
        score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    else:
        score_scaled = score

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat


'''
class Add_And_Normalization_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        embedding_dim = dim
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans
'''


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0]))

    def forward(self, input1, input2):
        return input1 + input2 * self.alpha


class Feed_Forward_Module(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        embedding_dim = emb_dim
        ff_hidden_dim = ff_dim

        self.alpha = nn.Parameter(torch.Tensor([0]))
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return input1 + self.W2(F.relu(self.W1(input1))) * self.alpha


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            hidden_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.Wq_s = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk_s = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv_s = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.multi_head_combine_s = nn.Linear(n_heads * val_dim, embed_dim)
        self.addAndNormalizations = Add_And_Normalization_Module(input_dim)
        self.feedForwards = Feed_Forward_Module(input_dim, hidden_dim)
        self.RoPE = RotatePostionalEncoding(embed_dim, 10000)

        self.Wq_dn = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk_dn = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv_dn = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wq2_dn = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk2_dn = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv2_dn = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.multi_head_combine_dn = nn.Linear(n_heads * val_dim, embed_dim)
        self.multi_head_combine2_dn = nn.Linear(n_heads * val_dim, embed_dim)
        self.addAndNormalization1_dn = Add_And_Normalization_Module(input_dim)
        self.feedForward1_dn = Feed_Forward_Module(input_dim, hidden_dim)
        self.addAndNormalization2_dn = Add_And_Normalization_Module(input_dim)
        self.feedForward2_dn = Feed_Forward_Module(input_dim, hidden_dim)

        self.Wq_ad = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk_ad = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv_ad = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wq2_ad = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk2_ad = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv2_ad = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.multi_head_combine_ad = nn.Linear(n_heads * val_dim, embed_dim)
        self.multi_head_combine2_ad = nn.Linear(n_heads * val_dim, embed_dim)
        self.addAndNormalization1_ad = Add_And_Normalization_Module(input_dim)
        self.feedForward1_ad = Feed_Forward_Module(input_dim, hidden_dim)
        self.addAndNormalization2_ad = Add_And_Normalization_Module(input_dim)
        self.feedForward2_ad = Feed_Forward_Module(input_dim, hidden_dim)

        self.Wq_an = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk_an = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv_an = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wq2_an = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk2_an = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv2_an = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.multi_head_combine_an = nn.Linear(n_heads * val_dim, embed_dim)
        self.multi_head_combine2_an = nn.Linear(n_heads * val_dim, embed_dim)
        self.addAndNormalization1_an = Add_And_Normalization_Module(input_dim)
        self.feedForward1_an = Feed_Forward_Module(input_dim, hidden_dim)
        self.addAndNormalization2_an = Add_And_Normalization_Module(input_dim)
        self.feedForward2_an = Feed_Forward_Module(input_dim, hidden_dim)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, agent_emb, depot_emb, node_emb):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        head_num = self.n_heads
        # h should be (batch_size, graph_size, input_dim)
        batch_size, agent_num, input_dim = agent_emb.size()
        batch_size, depot_size, input_dim = depot_emb.size()
        batch_size, graph_size, input_dim = node_emb.size()

        #self
        q_s = reshape_by_heads(self.Wq_s(node_emb), head_num=head_num)
        k_s = reshape_by_heads(self.Wk_s(node_emb), head_num=head_num)
        v_s = reshape_by_heads(self.Wv_s(node_emb), head_num=head_num)
        out_concat_s = multi_head_attention(q_s, k_s, v_s, sharp=False)  # shape: (B, n, head_num*key_dim)
        multi_head_out_s = self.multi_head_combine_s(out_concat_s)  # shape: (B, n, embedding_dim)
        node_hidden_2 = self.addAndNormalizations(node_emb, multi_head_out_s)
        node_out_2 = self.feedForwards(node_hidden_2)


        #dn
        q_dn = reshape_by_heads(self.Wq_dn(depot_emb), head_num=head_num)
        k_dn = reshape_by_heads(self.Wk_dn(node_out_2), head_num=head_num)
        v_dn = reshape_by_heads(self.Wv_dn(node_out_2), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
        out_concat_dn = multi_head_attention(q_dn, k_dn, v_dn)  # shape: (B, n, head_num*key_dim)
        multi_head_out1_dn = self.multi_head_combine_dn(out_concat_dn)  # shape: (B, n, embedding_dim)
        depot_hidden = self.addAndNormalization1_dn(depot_emb, multi_head_out1_dn)
        depot_out_ = self.feedForward1_dn(depot_hidden)
        q2_dn = reshape_by_heads(self.Wq2_dn(node_out_2), head_num=head_num)
        k2_dn = reshape_by_heads(self.Wk2_dn(depot_out_), head_num=head_num)
        v2_dn = reshape_by_heads(self.Wv2_dn(depot_out_), head_num=head_num)
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat2_dn = multi_head_attention(q2_dn, k2_dn, v2_dn, sharp=True)  # shape: (B, n, head_num*key_dim)
        multi_head_out2_dn = self.multi_head_combine2_dn(out_concat2_dn)  # shape: (B, n, embedding_dim)
        node_hidden = self.addAndNormalization2_dn(node_out_2, multi_head_out2_dn)
        node_out_ = self.feedForward2_dn(node_hidden)


        q_ad = reshape_by_heads(self.RoPE(self.Wq_ad(agent_emb)), head_num=head_num)
        k_ad = reshape_by_heads(self.Wk_ad(depot_out_), head_num=head_num)
        v_ad = reshape_by_heads(self.Wv_ad(depot_out_), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
        out_concat_ad = multi_head_attention(q_ad, k_ad, v_ad, sharp=True)  # shape: (B, n, head_num*key_dim)
        multi_head_out1_ad = self.multi_head_combine_ad(out_concat_ad)  # shape: (B, n, embedding_dim)
        agent_hidden = self.addAndNormalization1_ad(agent_emb, multi_head_out1_ad)
        agent_out_ = self.feedForward1_ad(agent_hidden)
        q2_ad = reshape_by_heads(self.Wq2_ad(depot_out_), head_num=head_num)
        k2_ad = reshape_by_heads(self.RoPE(self.Wk2_ad(agent_out_)), head_num=head_num)
        v2_ad = reshape_by_heads(self.RoPE(self.Wv2_ad(agent_out_)), head_num=head_num)
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat2_ad = multi_head_attention(q2_ad, k2_ad, v2_ad)  # shape: (B, n, head_num*key_dim)
        multi_head_out2_ad = self.multi_head_combine2_ad(out_concat2_ad)  # shape: (B, n, embedding_dim)
        depot_hidden = self.addAndNormalization2_ad(depot_out_, multi_head_out2_ad)
        depot_out = self.feedForward2_ad(depot_hidden)

        q_an = reshape_by_heads(self.RoPE(self.Wq_an(agent_out_)), head_num=head_num)
        k_an = reshape_by_heads(self.Wk_an(node_out_), head_num=head_num)
        v_an = reshape_by_heads(self.Wv_an(node_out_), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
        out_concat_an = multi_head_attention(q_an, k_an, v_an)  # shape: (B, n, head_num*key_dim)
        multi_head_out1_an = self.multi_head_combine_an(out_concat_an)  # shape: (B, n, embedding_dim)
        agent_hidden = self.addAndNormalization1_an(agent_out_, multi_head_out1_an)
        agent_out = self.feedForward1_an(agent_hidden)
        q2_an = reshape_by_heads(self.Wq2_an(node_out_), head_num=head_num)
        k2_an = reshape_by_heads(self.RoPE(self.Wk2_an(agent_out)), head_num=head_num)
        v2_an = reshape_by_heads(self.RoPE(self.Wv2_an(agent_out)), head_num=head_num)
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat2_an = multi_head_attention(q2_an, k2_an, v2_an, sharp=True)  # shape: (B, n, head_num*key_dim)
        multi_head_out2_an = self.multi_head_combine2_an(out_concat2_an)  # shape: (B, n, embedding_dim)
        node_hidden = self.addAndNormalization2_an(node_out_, multi_head_out2_an)
        node_out = self.feedForward2_an(node_hidden)

        return agent_out, depot_out, node_out


class GraphMAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphMAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    def forward(self, agents, depots, nodes, agent_num=3, mask=None):
        assert mask is None, "TODO mask not yet supported!"
        # Batch multiply to get initial embeddings of nodes

        for layer in self.layers:
            agents, depots, nodes = layer(agents, depots, nodes)

        h = torch.cat((depots, nodes), dim=1)
        return (
            agents,  # (batch_size, graph_size, embed_dim)
            h  # average to get embedding of graph, (batch_size, embed_dim)
        )
