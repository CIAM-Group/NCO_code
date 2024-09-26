import torch
import torch.nn as nn
import torch.nn.functional as F


class KPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = KP_Encoder(**model_params)
        self.decoder = KP_Decoder(**model_params)
        self.encoded_nodes = None
        self.encoded_graph_mean = None

        self.log_scale = None
        # shape: (batch, problem, embedding_dim)

    def pre_forward(self, reset_state):

        dist = reset_state.dist[:, :-1, :]  # shape: (batch, problem, problem)
        self.log_scale = reset_state.log_scale

        self.encoded_nodes = self.encoder(reset_state.problems, dist, reset_state.log_scale)
        # shape: (batch, problem, embedding_dim)
        self.decoder.set_kv(self.encoded_nodes)

        self.encoded_graph_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
        # shape: (batch, 1, embedding_dim)

    def set_decoder_type(self, decoder_type):
        self.model_params['eval_type'] = decoder_type

    def forward(self, state, cur_dist):
        # 解决方案没有顺序（结果是一组项），因此不需要添加来源和目的地的标记
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        encoded_graph_mean_pomo = self.encoded_graph_mean.expand(batch_size, pomo_size, -1)
        # shape: (batch, pomo, embedding_dim)
        probs = self.decoder(encoded_graph_mean_pomo, state.capacity, cur_dist, self.log_scale, ninf_mask=state.ninf_mask)
        # shape: (batch, pomo, problem)

        if self.training or self.model_params['eval_type'] == 'softmax':
            while True:
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                # shape: (batch, pomo)
                if (prob != 0).all():
                    break
        else:
            selected = probs.argmax(dim=2)
            # shape: (batch, pomo)
            prob = None

        return selected, prob


# def _get_encoding(encoded_nodes, node_index_to_pick):
#     # encoded_nodes.shape: (batch, problem, embedding)
#     # node_index_to_pick.shape: (batch, pomo)
#
#     batch_size = node_index_to_pick.size(0)
#     pomo_size = node_index_to_pick.size(1)
#     embedding_dim = encoded_nodes.size(2)
#
#     gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
#     # shape: (batch, pomo, embedding)
#
#     picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
#     # shape: (batch, pomo, embedding)
#
#     return picked_nodes


########################################
# ENCODER
########################################

class KP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, dist, log_scale):
        # data.shape: (batch, problem, 2) weight and value

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        negative_dist = dist  # dist is preprocessed in env.py
        for layer in self.layers:
            out = layer(out, negative_dist, log_scale)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        alpha_default = self.model_params['alpha_default']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.AFT_dist_alpha = nn.Parameter(torch.Tensor([alpha_default]), requires_grad=True)

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
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha * negative_dist
        # shape: (batch, problem, problem)
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(k), v)
        # shape: (batch, problem, embedding_dim)

        weighted = bias / (torch.exp(alpha_dist_bias_scale) @ torch.exp(k))
        # prevent nan in weighted, if nan, use torch.nan_to_num_ to solve it.
        if torch.isnan(weighted).any():
            weighted = torch.nan_to_num_(bias) / torch.nan_to_num_(torch.exp(alpha_dist_bias_scale) @ torch.exp(k))

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

class KP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        alpha_default = self.model_params['alpha_default']

        self.Wq_last = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        self.probs_dist_alpha = nn.Parameter(torch.Tensor([alpha_default]), requires_grad=True)
        self.AFT_dist_alpha = nn.Parameter(torch.Tensor([alpha_default]), requires_grad=True)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        # shape: (batch, problem, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def forward(self, encoded_graph_mean_pomo, capacity, cur_dist, log_scale, ninf_mask):
        # encoded_graph_mean_pomo.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        # cur_dist.shape: (batch, pomo, problem)
        # capacity.shape: (batch,pomo)

        input_cat = torch.cat((encoded_graph_mean_pomo, capacity[:, :, None]), dim=2)
        # shape = (batch, pomo, embedding+1)
        q_last = self.Wq_last(input_cat)
        # shape: (batch, pomo, embedding_dim)

        q = q_last
        # shape: (batch, pomo, embedding_dim)

        cur_dist_out = cur_dist  # cur_dist is preprocessed in env.py
        # shape: (batch, pomo, problem)

        #  AFT
        #######################################################
        sigmoid_q = torch.sigmoid(q)
        # shape: (batch, pomo, embedding_dim)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha * cur_dist_out
        alpha_dist_bias_scale = alpha_dist_bias_scale + ninf_mask
        # shape: (batch, pomo, problem)
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(self.k), self.v)
        # shape: (batch, problem, embedding_dim)

        weighted = bias / (torch.exp(alpha_dist_bias_scale) @ torch.exp(self.k))

        # prevent nan in weighted, if nan, use torch.nan_to_num_ to solve it.
        # ref: https://pytorch.org/docs/1.10/generated/torch.nan_to_num.html
        ####################################################################
        if torch.isnan(weighted).any():
            weighted = torch.nan_to_num_(bias) / torch.nan_to_num_(torch.exp(alpha_dist_bias_scale) @ torch.exp(self.k))
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
        score_scaled = score_scaled + log_scale * self.probs_dist_alpha * cur_dist_out
        # shape: (batch, pomo, problem)

        logit_clipping = self.model_params['logit_clipping']
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        # shape: (batch, pomo, problem)

        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

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
