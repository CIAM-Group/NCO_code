import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class MTSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):

        self.encoded_nodes = self.encoder(reset_state.problems, reset_state.route_num, reset_state.flag_return)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)
        self.log_scale = reset_state.log_scale

    def forward(self, state, cur_dist, flag):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.ones(pomo_size)[None, :].expand(batch_size, pomo_size).long() * state.depot_num.max(-1)[0]
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_node = _get_encoding(self.encoded_nodes,
                                               (state.ninf_mask.size(-1) - torch.ones(pomo_size)[None, :].expand(batch_size, pomo_size).long()) * (1 - flag)[:, None])
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_agent = _get_encoding(self.encoded_nodes, state.depot_id.squeeze(-1))
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, encoded_last_agent, cur_dist, state, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embed_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        head_num = self.model_params['head_num']
        self.RoPE = RotatePostionalEncoding(embed_dim, 10000)
        self.embedding_depot = nn.Linear(2, embed_dim)
        self.embedding_place = nn.Linear(2, embed_dim)
        self.embedding_node = nn.Linear(2, embed_dim)
        self.pos_emb_proj = nn.Linear(embed_dim, embed_dim)
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.embedding_out = nn.Linear(embed_dim, embed_dim)
        self.embedding_in = nn.Linear(embed_dim, embed_dim)
        feed_forward_hidden = self.model_params['ff_hidden_dim']

        self.layers = nn.Sequential(*(EncoderLayer(**model_params) for _ in range(encoder_layer_num)))

    def forward(self, coords, route_num, flag):
        num_depot = route_num.max(-1)[0]
        depot_coords = coords[:, :num_depot, :]
        nodes_coords = coords[:, num_depot:, :]
        embedded_depot = self.embedding_depot(depot_coords)
        place_embedding = self.embedding_place(depot_coords)
        positional_embedding = self.RoPE(place_embedding)
        positional_embedding = self.alpha * self.pos_emb_proj(positional_embedding)
        depot_embedding = embedded_depot + positional_embedding
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(nodes_coords)
        # shape: (batch, problem, embedding)
        embedded_node[:, 0, :] = self.embedding_in(embedded_node[:, 0, :].clone())
        depot_embedding[flag == 0] = self.embedding_out(depot_embedding[flag == 0])
        last = self.embedding_out(embedded_node[:, -1, :].clone())
        embedded_node[:, -1, :][flag == 1] = last[flag == 1]
        for layer in self.layers:
            depot_embedding, embedded_node = layer(depot_embedding, embedded_node, route_num)

        h = torch.cat((depot_embedding, embedded_node), dim=1)
        return h


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super(EncoderLayer, self).__init__()
        self.model_params = model_params
        n_heads = self.model_params['head_num']
        input_dim = self.model_params['embedding_dim']
        embed_dim = self.model_params['embedding_dim']
        val_dim = self.model_params['qkv_dim']
        key_dim = self.model_params['qkv_dim']
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

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward1 = Feed_Forward_Module(**model_params)

        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
        self.feedForward2 = Feed_Forward_Module(**model_params)

        self.addAndNormalization3 = Add_And_Normalization_Module(**model_params)
        self.feedForward3 = Feed_Forward_Module(**model_params)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, agent_emb, node_emb, route_num):
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
        out_concat_2 = multi_head_attention(q_2, k_2, v_2, route_num=route_num, sharp=True)  # shape: (B, n, head_num*key_dim)
        multi_head_out_2 = self.multi_head_combine_2(out_concat_2)  # shape: (B, n, embedding_dim)
        node_hidden = self.addAndNormalization2(node_out_2, multi_head_out_2)
        node_out = self.feedForward2(node_hidden)

        return agent_out, node_out


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim * 2 + 2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.dis_emb = nn.Sequential(nn.Linear(4, head_num * qkv_dim, bias=False))

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, encoded_last_agents, cur_dist, state, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']
        route_num = state.depot_num[:, None, None].expand_as(state.route_cnt) - 1
        q_info = self.dis_emb(torch.cat((state.lengths.gather(-1, state.route_cnt), state.lengths.gather(-1, route_num), state.max_dis, state.remain_max_dis), -1))

        q_last = torch.cat((encoded_last_node, encoded_last_agents, 1.0 - (state.route_cnt + 1) / (route_num + 1), state.left_city[:, :, None] / state.city_num), 2)
        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(q_last), head_num=head_num)
        q_info = reshape_by_heads(q_info, head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last + q_info
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        max_dist = cur_dist.max(-1)[0].unsqueeze(-1)  # shape: (batch, pomo, 1)
        cur_dist_out = cur_dist / max_dist  # shape: (batch, pomo, problem)
        cur_dist_out = -1 * cur_dist_out
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled + self.dist_alpha_1 * cur_dist_out)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, route_num=None, sharp=False):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    if sharp == False:
        score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    else:
        score_scaled = score
    if route_num is not None:
        route_mask = (torch.arange(input_s)[None, None, :].repeat(1, n, 1) >= route_num[:, None, None]).float()
        route_mask[route_mask == 1] = float('-inf')
        score_scaled = score_scaled + route_mask[:, None, :, :].expand(-1, head_num, -1, -1)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
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


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
