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


class MultiHeadHAAttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            hidden_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadHAAttentionLayer, self).__init__()

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
        self.Wq_2_pickup = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wq_2_delivery = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wk_2 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)
        self.Wv_2 = nn.Linear(embed_dim, n_heads * val_dim, bias=False)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W2_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W3_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        # delivery
        self.W4_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W5_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W6_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        # self.alpha = nn.Parameter(torch.Tensor(n_heads, 1, 1))
        self.multi_head_combine_3 = nn.Linear(n_heads * val_dim, embed_dim)
        self.multi_head_combine = nn.Linear(n_heads * val_dim, embed_dim)
        self.multi_head_combine_2 = nn.Linear(n_heads * val_dim, embed_dim)

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

    def ha_encoder(self, q):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"
        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]
        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        # pickup -> its delivery attention
        n_pick = (graph_size) // 2
        shp_delivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.n_heads, batch_size, n_pick, -1)
        # pickup -> all pickups attention
        shp_allpick = (self.n_heads, batch_size, n_pick, -1)
        shp_q_allpick = (self.n_heads, batch_size, n_pick, -1)
        # pickup -> all pickups attention
        shp_alldelivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_alldelivery = (self.n_heads, batch_size, n_pick, -1)
        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        # pickup -> its delivery
        pick_flat = h[:, :n_pick, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        delivery_flat = h[:, n_pick:, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        # pickup -> its delivery attention
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(shp_delivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(shp_delivery)  # (n_heads, batch_size, n_pick, key/val_size)
        # pickup -> all pickups attention
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(shp_q_allpick)  # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        # pickup -> all delivery
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(shp_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)
        # pickup -> its delivery
        V_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device)
        ], 2)
        # delivery -> its pickup attention
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(shp_delivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(shp_q_pick)  # (n_heads, batch_size, n_pick, key/val_size)
        # delivery -> all delivery attention
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        # delivery -> all pickup
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(shp_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(shp_q_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)
        # delivery -> its pick up
        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            V_pick  # [n_heads, batch_size, n_pick, key/val_size]
        ], 2)
        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        ##Pick up
        # ??pair???attention??
        compatibility_pick_delivery = self.norm_factor * torch.sum(Q_pick * K_delivery, -1)  # element_wise, [n_heads, batch_size, n_pick]
        # [n_heads, batch_size, n_pick, n_pick]
        compatibility_pick_allpick = self.norm_factor * torch.matmul(Q_pick_allpick, K_allpick.transpose(2, 3))  # [n_heads, batch_size, n_pick, n_pick]
        compatibility_pick_alldelivery = self.norm_factor * torch.matmul(Q_pick_alldelivery, K_alldelivery.transpose(2, 3))  # [n_heads, batch_size, n_pick, n_pick]

        ##Deliver
        compatibility_delivery_pick = self.norm_factor * torch.sum(Q_delivery * K_pick, -1)  # element_wise, [n_heads, batch_size, n_pick]
        compatibility_delivery_alldelivery = self.norm_factor * torch.matmul(Q_delivery_alldelivery, K_alldelivery2.transpose(2, 3))  # [n_heads, batch_size, n_pick, n_pick]
        compatibility_delivery_allpick = self.norm_factor * torch.matmul(Q_delivery_allpickup, K_allpickup2.transpose(2, 3))  # [n_heads, batch_size, n_pick, n_pick]
        ##Pick up->
        # compatibility_additional?pickup????delivery????attention(size 1),1:n_pick+1??attention,depot?delivery??
        compatibility_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            compatibility_pick_delivery,  # [n_heads, batch_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype, device=compatibility.device)
        ], -1).view(self.n_heads, batch_size, graph_size, 1)
        compatibility_additional_allpick = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            compatibility_pick_allpick,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device)
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)
        compatibility_additional_alldelivery = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            compatibility_pick_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device)
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)
        # [n_heads, batch_size, n_query, graph_size+1+n_pick+n_pick]
        ##Delivery->
        compatibility_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_delivery_pick  # [n_heads, batch_size, n_pick]
        ], -1).view(self.n_heads, batch_size, graph_size, 1)
        compatibility_additional_alldelivery2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_delivery_alldelivery  # [n_heads, batch_size, n_pick, n_pick]
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_allpick2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_delivery_allpick  # [n_heads, batch_size, n_pick, n_pick]
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility = torch.cat([compatibility, compatibility_additional_delivery, compatibility_additional_allpick, compatibility_additional_alldelivery,
                                   compatibility_additional_pick, compatibility_additional_alldelivery2, compatibility_additional_allpick2], dim=-1)
        # Optionally apply mask to prevent attention
        attn = torch.softmax(compatibility, dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)
        # heads: [n_heads, batrch_size, n_query, val_size], attn????pick?deliver?attn
        heads = torch.matmul(attn[:, :, :, :graph_size], V)  # V: (self.n_heads, batch_size, graph_size, val_size)
        # heads??pick -> its delivery
        heads = heads + attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size, 1) * V_additional_delivery  # V_addi:[n_heads, batch_size, graph_size, key_size]
        # heads??pick -> otherpick, V_allpick: # [n_heads, batch_size, n_pick, key_size]
        # heads: [n_heads, batch_size, graph_size, key_size]
        heads = heads + torch.matmul(attn[:, :, :, graph_size + 1:graph_size + 1 + n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_allpick)
        # V_alldelivery: # (n_heads, batch_size, n_pick, key/val_size)
        heads = heads + torch.matmul(attn[:, :, :, graph_size + 1 + n_pick:graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_alldelivery)
        # delivery
        heads = heads + attn[:, :, :, graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, 1) * V_additional_pick
        heads = heads + torch.matmul(attn[:, :, :, graph_size + 1 + 2 * n_pick + 1:graph_size + 1 + 3 * n_pick + 1].view(self.n_heads, batch_size, graph_size, n_pick),
                                     V_alldelivery2)
        heads = heads + torch.matmul(attn[:, :, :, graph_size + 1 + 3 * n_pick + 1:].view(self.n_heads, batch_size, graph_size, n_pick), V_allpickup2)
        out = heads.permute(1, 2, 0, 3).reshape(batch_size, n_query, self.n_heads * self.val_dim) # shape: (B, n, head_num, key_dim)
        return out

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

        multi_head_out_3 = self.multi_head_combine_3(self.ha_encoder(node_emb))  # shape: (B, n, embedding_dim)
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
        pickup = node_out_2[:, :node_out_2.size(1) // 2, :]
        delivery = node_out_2[:, node_out_2.size(1) // 2:, :]
        q_2 = reshape_by_heads(self.Wq_2(node_out_2), head_num=head_num)
        q_2_p = reshape_by_heads(self.Wq_2_pickup(pickup), head_num=head_num)
        q_2_d = reshape_by_heads(self.Wq_2_delivery(delivery), head_num=head_num)
        q_2 += torch.cat((q_2_d, q_2_p), dim=2)
        k_2 = reshape_by_heads(self.Wk_2(agent_out), head_num=head_num)
        v_2 = reshape_by_heads(self.Wv_2(agent_out), head_num=head_num)
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat_2 = multi_head_attention(q_2, k_2, v_2, sharp=True)  # shape: (B, n, head_num*key_dim)
        multi_head_out_2 = self.multi_head_combine_2(out_concat_2)  # shape: (B, n, embedding_dim)
        node_hidden = self.addAndNormalization2(node_out_2, multi_head_out_2)
        node_out = self.feedForward2(node_hidden)

        return agent_out, node_out


class GraphHAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphHAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadHAAttentionLayer(n_heads, embed_dim, embed_dim, feed_forward_hidden)
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
