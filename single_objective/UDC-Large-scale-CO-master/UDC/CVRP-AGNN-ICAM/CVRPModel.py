import torch
import torch.nn as nn
import torch.nn.functional as F


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        dist = reset_state.dist  # shape: (batch, problem+1, problem+1)

        self.log_scale = reset_state.log_scale  # it is a scalar and used for influence of distance
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand, dist, self.log_scale, reset_state.flag_return)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, cur_dist, flag):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.ones(pomo_size)[None, :].expand(batch_size, pomo_size).long()
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_node = _get_encoding(self.encoded_nodes,
                                               (state.ninf_mask.size(-1) - torch.ones(pomo_size)[None, :].expand(batch_size, pomo_size).long()) * (1 - flag)[:, None])
            self.decoder.set_q1(encoded_first_node)
            encoded_depot = _get_encoding(self.encoded_nodes, torch.zeros(pomo_size)[None, :].expand(batch_size, pomo_size).long())
            self.decoder.set_q2(encoded_depot)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.load, state.left, cur_dist, self.log_scale, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = (probs.reshape(batch_size * pomo_size, -1) + 1e-10).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

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

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.embedding_out = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_in = nn.Linear(embedding_dim, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand, dist, log_scale, flag):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        id = ((1 - flag) * embedded_node.size(1))[:, None, None].expand(-1, -1, embedded_node.size(-1))
        # shape: (batch, problem, embedding)
        embedded_node[:, 0, :] = self.embedding_in(embedded_node[:, 0, :].clone())

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        last = self.embedding_out(out.gather(1, id).clone())
        out = out.scatter(1, id, last)
        # shape: (batch, problem+1, embedding)
        negative_dist = -1 * dist  # -1 * dist represents negative distance

        for layer in self.layers:
            out = layer(out, negative_dist, log_scale)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        self.AFT_dist_alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, input1, negative_dist, log_scale):
        # input1.shape: (batch, problem+1, embedding)

        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)
        # qkv shape: (batch, problem+1, embedding)

        ##################################################################
        # AFT core code
        # paper: An Attention Free Transformer
        # https://arxiv.org/pdf/2105.14103.pdf
        ##################################################################
        sigmoid_q = torch.sigmoid(q)
        # shape: (batch, problem+1, embedding)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha * negative_dist
        # shape: (batch, problem+1, problem)
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(k), v)
        # shape: (batch, problem+1, embedding)

        # ref: https://pytorch.org/docs/1.10/generated/torch.nan_to_num.html
        # prevent nan and inf
        weighted = torch.nan_to_num_(bias) / torch.nan_to_num_(torch.exp(alpha_dist_bias_scale) @ torch.exp(k))
        # shape: (batch, problem+1, embedding)
        AFT_out = torch.mul(sigmoid_q, weighted)
        # shape: (batch, problem+1, embedding)
        ##################################################################
        # AFT core code end.
        ##################################################################

        out1 = self.add_n_normalization_1(input1, AFT_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim + 2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention
        self.q1 = None  # saved q1, for multi-head attention
        self.q2 = None  # saved q2, for multi-head attention

        self.probs_dist_alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.AFT_dist_alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)

        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        # shape: (batch, problem+1, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = self.Wq_1(encoded_q1)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = self.Wq_2(encoded_q2)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, left, cur_dist, log_scale, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None], left[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = self.Wq_last(input_cat)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        # q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        cur_dist_out = -1 * cur_dist
        # shape: (batch, pomo, problem+1)

        #  AFT
        # https://arxiv.org/pdf/2105.14103.pdf
        #######################################################
        sigmoid_q = torch.sigmoid(q)
        # shape: (batch, pomo, embedding)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha * cur_dist_out
        alpha_dist_bias_scale = alpha_dist_bias_scale + ninf_mask
        # shape: (batch, pomo, problem)
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(self.k), self.v)
        # shape: (batch, problem, embedding)

        # ref: https://pytorch.org/docs/1.10/generated/torch.nan_to_num.html
        # prevent nan and inf
        weighted = torch.nan_to_num_(bias) / (torch.nan_to_num_(torch.exp(alpha_dist_bias_scale) @ torch.exp(self.k)) + 1e-20)
        # shape: (batch, problem, embedding)
        AFT_out = torch.mul(sigmoid_q, weighted)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(AFT_out, self.single_head_key)
        # shape: (batch, pomo, problem+1)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem+1)
        # cur_dist_out is -1 * cur_dist, so this is a minus sign
        score_scaled = score_scaled + log_scale * self.probs_dist_alpha * cur_dist_out
        # shape: (batch, pomo, problem+1)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem+1)

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


class AddAndInstanceNormalization(nn.Module):
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


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
