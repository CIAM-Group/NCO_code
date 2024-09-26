import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):

        dist = reset_state.dist  # shape: (batch, problem, problem)

        self.encoded_nodes = self.encoder(reset_state.problems, dist, reset_state.log_scale)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)
        self.log_scale = reset_state.log_scale

    def forward(self, state, cur_dist):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.cat((torch.zeros(pomo_size // 2)[None, :].expand(batch_size, pomo_size // 2),
                                  state.ninf_mask.size(-1) - torch.ones(pomo_size // 2)[None, :].expand(batch_size, pomo_size // 2)), dim=-1).long()
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, torch.cat((state.ninf_mask.size(-1) - torch.ones(pomo_size // 2)[None, :].expand(batch_size, pomo_size // 2),
                                                                              torch.zeros(pomo_size // 2)[None, :].expand(batch_size, pomo_size // 2)), dim=-1).long())
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, cur_dist, self.log_scale, ninf_mask=state.ninf_mask)
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
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.embedding_out = nn.Linear(embedding_dim, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, dist, log_scale):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)
        embedded_input[:, -1, :] = self.embedding_out(embedded_input[:, -1, :].clone())
        embedded_input[:, 0, :] = self.embedding_out(embedded_input[:, 0, :].clone())

        out = embedded_input
        negative_dist = -1 * dist  # -1 * dist represents negative distance
        for layer in self.layers:
            out = layer(out, negative_dist, log_scale)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.AFT_dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, input1, negative_dist, log_scale):
        # input.shape: (batch, problem, embedding_dim)
        # dist.shape: (batch, problem, problem)
        # scale.shape: (1,)

        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)
        # shape: (batch, problem, embedding_dim)

        ##################################################################
        # AFT core code
        # paper: An Attention Free Transformer
        # https://arxiv.org/pdf/2105.14103.pdf
        ##################################################################

        sigmoid_q = torch.sigmoid(q)
        # shape: (batch, problem, embedding_dim)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha_1 * negative_dist
        # shape: (batch, problem, problem)
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(k), v)
        # shape: (batch, problem, embedding_dim)

        weighted = bias / (torch.exp(alpha_dist_bias_scale) @ torch.exp(k))
        # shape: (batch, problem, embedding_dim)
        out = torch.mul(sigmoid_q, weighted)
        # shape: (batch, problem, embedding_dim)
        ##################################################################
        # AFT core code end.
        ##################################################################

        out1 = self.addAndNormalization1(input1, out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq_first = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.AFT_dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        # shape: (batch, problem, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo

        self.q_first = self.Wq_first(encoded_q1)
        # shape: (batch, problem, embedding)

    def forward(self, encoded_last_node, cur_dist, log_scale, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        # cur_dist.shape: (batch, pomo, problem)

        q_last = self.Wq_last(encoded_last_node)
        # shape: (batch, pomo, embedding_dim)

        q = self.q_first + q_last
        # shape: (batch, pomo, embedding_dim)

        cur_dist_out = -1 * cur_dist
        # shape: (batch, pomo, problem)

        #  AFT
        #######################################################
        sigmoid_q = torch.sigmoid(q)
        # shape: (batch, pomo, embedding_dim)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha_1 * cur_dist_out
        alpha_dist_bias_scale = alpha_dist_bias_scale + ninf_mask
        # shape: (batch, pomo, problem)
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(self.k), self.v)
        # shape: (batch, problem, embedding_dim)

        weighted = bias / (torch.exp(alpha_dist_bias_scale) @ torch.exp(self.k))
        # shape: (batch, problem, embedding_dim)
        AFT_out = torch.mul(sigmoid_q, weighted)
        # shape: (batch, pomo, embedding_dim)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(AFT_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        # cur_dist_out is -1 * cur_dist
        score_scaled = score_scaled + log_scale * self.dist_alpha_1 * cur_dist_out
        # shape: (batch, pomo, problem)

        logit_clipping = self.model_params['logit_clipping']
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        # shape: (batch, pomo, problem)

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
