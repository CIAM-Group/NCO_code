import torch
import numpy as np
from torch import nn
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F


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

        self.Wq = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, n_heads * val_dim, bias=False)

        self.Wq_2 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk_2 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv_2 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)

        self.Wq_3 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk_3 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv_3 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        # self.alpha = nn.Parameter(torch.Tensor(n_heads, 1, 1))
        self.multi_head_combine = nn.Linear(n_heads * val_dim, embed_dim)
        self.multi_head_combine_2 = nn.Linear(n_heads * val_dim, embed_dim)
        self.multi_head_combine_3 = nn.Linear(n_heads * val_dim, embed_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(input_dim)
        self.feedForward1 = Feed_Forward_Module(input_dim, hidden_dim)

        self.addAndNormalization2 = Add_And_Normalization_Module(input_dim)
        self.feedForward2 = Feed_Forward_Module(input_dim, hidden_dim)

        self.addAndNormalization3 = Add_And_Normalization_Module(input_dim)
        self.feedForward3 = Feed_Forward_Module(input_dim, hidden_dim)
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, agent_emb, node_emb):
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
        batch_size, graph_size, input_dim = node_emb.size()

        q_3 = reshape_by_heads(self.Wq_3(node_emb), head_num=head_num)
        k_3 = reshape_by_heads(self.Wk_3(node_emb), head_num=head_num)
        v_3 = reshape_by_heads(self.Wv_3(node_emb), head_num=head_num)
        out_concat_3 = multi_head_attention(q_3, k_3, v_3, sharp=False)  # shape: (B, n, head_num*key_dim)
        multi_head_out_3 = self.multi_head_combine_3(out_concat_3)  # shape: (B, n, embedding_dim)
        node_hidden_2 = self.addAndNormalization3(node_emb, multi_head_out_3)
        node_out_2 = self.feedForward3(node_hidden_2)

        q = reshape_by_heads(self.Wq(agent_emb), head_num=head_num)
        k = reshape_by_heads(self.Wk(node_out_2), head_num=head_num)
        v = reshape_by_heads(self.Wv(node_out_2), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)

        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)
        agent_hidden = self.addAndNormalization1(agent_emb, multi_head_out)
        agent_out = self.feedForward1(agent_hidden)
        # out3 = multi_head_out
        ################################################################
        ################################################################
        q_2 = reshape_by_heads(self.Wq_2(node_out_2), head_num=head_num)
        k_2 = reshape_by_heads(self.Wk_2(agent_out), head_num=head_num)
        v_2 = reshape_by_heads(self.Wv_2(agent_out), head_num=head_num)
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat_2 = multi_head_attention(q_2, k_2, v_2, sharp=True)  # shape: (B, n, head_num*key_dim)
        multi_head_out_2 = self.multi_head_combine_2(out_concat_2)  # shape: (B, n, embedding_dim)
        node_hidden = self.addAndNormalization2(node_out_2, multi_head_out_2)
        node_out = self.feedForward2(node_hidden)

        return agent_out, node_out


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    def forward(self, agents, nodes, agent_num=3, mask=None):
        assert mask is None, "TODO mask not yet supported!"
        # Batch multiply to get initial embeddings of nodes

        for layer in self.layers:
            agents, nodes = layer(agents, nodes)

        h = torch.cat((agents, nodes), dim=1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
