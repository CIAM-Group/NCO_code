import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from nets.mdvrp_self_pa_encoder2 import GraphMAttentionEncoder
from nets.self_pa_encoder import GraphAttentionEncoder
from nets.ha_hpa_encoder import GraphHAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from nets.positional_encoding import PostionalEncoding
from nets.positional_encoding import RotatePostionalEncoding

'''
from gpu_mem_track import MemTracker

device = torch.device('cuda:0')
gpu_tracker = MemTracker()  # define a GPU tracker
'''


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 agent_num=3,
                 depot_num=3,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 ft="N"):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.agent_num = agent_num
        self.depot_num = depot_num
        self.ft = ft
        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.positional_encoding = PostionalEncoding(d_model=embedding_dim, max_len=10000)
        self.RoPE = RotatePostionalEncoding(embedding_dim, 10000)
        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.agent_per = None
        # Problem specific context parameters (placeholder and step context dimension)
        step_context_dim = 2 * embedding_dim + 2  # Embedding of current_agent, current node, # of left cities and # of left agents
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.init_embed_agent = nn.Linear(2, embedding_dim)
        self.decay = 1.0
        self.pos_emb_proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim, bias=False))
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.beta = nn.Parameter(torch.Tensor(embedding_dim))

        if problem.NAME == "mtsp":
            self.dis_emb = nn.Sequential(nn.Linear(3, embedding_dim, bias=False))

            self.embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embedding_dim,
                n_layers=self.n_encode_layers,
                normalization=normalization
            )
        if problem.NAME == "mdvrp":
            self.dis_emb = nn.Sequential(nn.Linear(3, embedding_dim, bias=False))

            self.embedder = GraphMAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embedding_dim,
                n_layers=self.n_encode_layers,
                normalization=normalization
            )
        if problem.NAME == "fmdvrp":
            self.dis_emb = nn.Sequential(nn.Linear(3, embedding_dim, bias=False))

            self.embedder = GraphMAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embedding_dim,
                n_layers=self.n_encode_layers,
                normalization=normalization
            )
        elif problem.NAME == "mpdp":
            self.init_embed_pick = nn.Linear(node_dim * 2, embedding_dim)
            self.init_embed_delivery = nn.Linear(node_dim, embedding_dim)
            self.dis_emb = nn.Sequential(nn.Linear(5, embedding_dim, bias=False))
            self.embedder = GraphHAttentionEncoder(n_heads=n_heads,
                                                   embed_dim=embedding_dim,
                                                   n_layers=self.n_encode_layers,
                                                   normalization=normalization
                                                   )
            self.embedder.agent_num = agent_num
        # Using the finetuned context Encoder
        if self.ft == "Y":
            self.contextual_emb = nn.Sequential(nn.Linear(embedding_dim, 8 * embedding_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(8 * embedding_dim, embedding_dim, bias=False)
                                                )

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, subloss=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        if self.problem.NAME == 'mtsp':
            problem = torch.cat((input[:, 0:1, :].repeat(1, self.agent_num, 1), input[:, 1:, :]), dim=1)
            agent_embeddings, node_embeddings = self._init_embed(input)
            embeddings, _ = self.embedder(agent_embeddings, node_embeddings, agent_num=self.agent_num)
        if self.problem.NAME == 'mpdp':
            problem = torch.cat((input['depot'].repeat(1, self.agent_num, 1), input['loc']), dim=1)
            agent_embeddings, node_embeddings = self._init_embed(input)
            embeddings, _ = self.embedder(agent_embeddings, node_embeddings, agent_num=self.agent_num)
        if self.problem.NAME == 'mdvrp':
            problem = torch.cat((input['depot'][:, :self.depot_num, :], input['loc']), dim=1)
            agent_embeddings, depot_embeddings, node_embeddings = self._init_embed(input)
            agent_embeddings, embeddings = self.embedder(agent_embeddings, depot_embeddings, node_embeddings, agent_num=self.agent_num)
        if self.problem.NAME == 'fmdvrp':
            problem = torch.cat((input['depot'][:, :self.depot_num, :], input['loc']), dim=1)
            agent_embeddings, depot_embeddings, node_embeddings = self._init_embed(input)
            agent_embeddings, embeddings = self.embedder(agent_embeddings, depot_embeddings, node_embeddings, agent_num=self.agent_num)
        cur_dist = torch.cdist(problem, problem, p=2)
        max_dist = cur_dist.max(-1)[0].unsqueeze(-1)  # shape: (batch, pomo, 1)
        cur_dist_out = cur_dist / max_dist  # shape: (batch, pomo, problem)
        self.dist_out = -1 * cur_dist_out

        if self.problem.NAME == 'mdvrp' or self.problem.NAME == 'fmdvrp':
            _log_p, pi, makespan, diff, cost_cl = self._inner(input, embeddings, agent_embeddings, subloss=subloss)

        if self.problem.NAME == 'mtsp' or self.problem.NAME == 'mpdp':
            _log_p, pi, makespan, diff, cost_cl = self._inner(input, embeddings, subloss=subloss)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, None)

        if return_pi:
            return makespan, ll, pi

        return makespan, diff, cost_cl, ll

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(3, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p

    def _init_embed(self, input):

        # mTSP

        if self.problem.NAME == "mtsp":
            if len(input.size()) == 2:
                input = input.unsqueeze(0)
            num_cities = input.size(1) - 1
            self.num_cities = num_cities
            # Embedding of depot
            depot_embedding = self.init_embed_depot(input[:, 0:1, :])
            place_embedding = self.init_embed_agent(input[:, 0:1, :])
            # Make the depot embedding the same for all agents
            depot_embedding = depot_embedding.repeat(1, self.agent_num, 1)
            place_embedding = place_embedding.repeat(1, self.agent_num, 1)
            positional_embedding = self.RoPE(place_embedding)
            positional_embedding = self.alpha * self.pos_emb_proj(positional_embedding)
            depot_embedding = depot_embedding + positional_embedding
            # Add the positional embedding to the depot embedding to give order bias to the agents
            return depot_embedding, self.init_embed(input[:, 1:, :])

        elif self.problem.NAME == "mdvrp" or self.problem.NAME == "fmdvrp":
            num_cities = input['loc'].size(1)
            self.num_cities = num_cities
            # Embedding of depot
            depot_embedding = self.init_embed_depot(input['depot'][:, :self.depot_num, :])
            # Make the depot embedding the same for all agents
            agent_embedding = self.beta[None, None, :].repeat(depot_embedding.size(0), self.agent_num, 1)
            positional_embedding = self.RoPE(agent_embedding)
            positional_embedding = self.alpha * self.pos_emb_proj(positional_embedding)
            # Add the positional embedding to the depot embedding to give order bias to the agents
            return positional_embedding, depot_embedding, self.init_embed(input['loc'])

        elif self.problem.NAME == "mpdp":
            n_loc = input['loc'].size(1)
            if len(input['depot'].size()) == 2:
                input['depot'] = input['depot'][:, None, :]
            original_depot = input['depot']
            new_input = input['loc']
            new_depot = original_depot.repeat(1, self.agent_num, 1)
            place_embedding = self.init_embed_agent(new_depot)
            embed_depot = self.init_embed_depot(new_depot)
            self.num_request = n_loc // 2
            positional_embedding = self.RoPE(place_embedding)
            positional_embedding = self.alpha * self.pos_emb_proj(positional_embedding)
            embed_depot = embed_depot + positional_embedding
            feature_pick = torch.cat([new_input[:, :n_loc // 2, :], new_input[:, n_loc // 2:, :]], -1)
            feature_delivery = new_input[:, n_loc // 2:, :]  # [batch_size, graph_size//2, 2]
            embed_pick = self.init_embed_pick(feature_pick)
            embed_delivery = self.init_embed_delivery(feature_delivery)

            return embed_depot, torch.cat([embed_pick, embed_delivery], 1)

    def _inner(self, input, embeddings, agent_embeddings=None, subloss=False):

        outputs = []
        assign = []
        sequences = []

        if self.problem.NAME == "mdvrp" or self.problem.NAME == "fmdvrp":
            state = self.problem.make_state(input, self.agent_num, self.agent_per, self.depot_num)
        else:
            state = self.problem.make_state(input, self.agent_num, self.agent_per)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)
        batch_size = state.ids.size(0)
        # Perform decoding steps
        i = 0
        while not state.all_finished():
            self.curr_dist = self.dist_out.gather(1, state.get_current_node().expand(-1, -1, self.dist_out.size(-1)))
            log_p, mask = self._get_log_p(fixed, state, agent_embeddings)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp(), mask)  # Squeeze out steps dimension
            assign.append(state.count_depot[:, :, 0].clone())
            state = state.update(selected)
            # Collect output of step
            outputs.append(log_p[:, :, :])
            sequences.append(selected)
            i += 1
        makespan = state.lengths.max(-1)[0]
        tour = torch.stack(sequences, 2)
        prob = torch.stack(outputs, 2)
        if subloss:
            assignment = torch.stack(assign, 2)
            assignment_hot = assignment[:, :, None, :].repeat(1, 1, self.agent_num, 1)
            for i in range(state.lengths.size(-1)):
                assignment_hot[:, :, i, :] = (assignment_hot[:, :, i, :] == i).long()
            assignment_corr = assignment_hot.gather(-1, tour.argsort(-1).unsqueeze(2).expand_as(assignment_hot)).view(batch_size, -1, tour.size(-1))
            # cost = torch.max(state.lengths, dim=-1)[0]
            assignment_corr[:, :, :self.agent_num] = 0
            value = state.lengths.clone().view(batch_size, -1)
            value2 = value.clone()
            value_exp = value.clone()
            for i in range(value2.size(1)):
                corr = (assignment_corr[:, :, :] == assignment_corr[:, i, :].unsqueeze(1)).all(-1)
                baseline = (value * corr).sum(-1) / corr.sum(-1)
                value2[:, i] -= baseline
                cal_min = value * corr
                cal_min[~corr] = 1e5
                value_exp[:, i] = cal_min.min(-1)[0]
            value_out = value2.view(batch_size, -1, self.agent_num).gather(-1, assignment)
            value_exp = value_exp.view(batch_size, -1, self.agent_num)
            return prob, tour, makespan, value_exp.max(-1)[0], value_out
        else:
            return prob, tour, makespan, None, None

    def sample_many(self, input, batch_rep=1, iter_rep=1, agent_num=3, aug=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input), agent_num)[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep, aug
        )

    def _select_node(self, probs, mask):

        batch_size, pomo_size, node_size = probs.size()
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(-1)

        elif self.decode_type == "sampling":
            selected = probs.view(-1, node_size).multinomial(1).squeeze(1).view(batch_size, pomo_size)

            while mask.gather(2, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.view(-1, node_size).multinomial(1).squeeze(1).view(batch_size, pomo_size)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_node_embeddings(embeddings[:, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed),
            self._make_heads(glimpse_val_fixed),
            logit_key_fixed
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, agent_embeddings=None, normalize=True):

        if self.problem.NAME == "mdvrp":
            query = fixed.context_node_projected + \
                    self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, agent_embeddings)) + self.dis_emb(
                torch.cat((state.lengths.gather(-1, state.count_depot), state.max_distance, state.remain_max_distance), -1))
        if self.problem.NAME == "fmdvrp":
            query = fixed.context_node_projected + \
                    self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, agent_embeddings)) + self.dis_emb(
                torch.cat((state.lengths.gather(-1, state.count_depot), state.max_distance, state.remain_max_distance), -1))
        elif self.problem.NAME == "mtsp":
            query = fixed.context_node_projected + \
                    self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state)) + self.dis_emb(
                torch.cat((state.lengths.gather(-1, state.count_depot), state.max_distance, state.remain_max_distance), -1))
        elif self.problem.NAME == "mpdp":
            query = fixed.context_node_projected + \
                    self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state)) \
                    + self.dis_emb(torch.cat((state.lengths.gather(-1, state.count_depot), state.remain_pickup_max_distance, state.remain_delivery_max_distance,
                                              state.longest_lengths.gather(-1, state.count_depot), state.remain_sum_paired_distance / (self.agent_num - (state.count_depot))), -1))

            # Add Finetuned the context node to the query
        if self.ft == "Y":
            context = query.detach()
            query = self.contextual_emb(query)
            query += context

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, agent_embeddings=None, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, pomo_size, _ = current_node.size()
        batch_size, _, embedding_dim = embeddings.size()
        gathering_index = current_node.expand(batch_size, pomo_size, embedding_dim)
        gathering_agent = state.agent_idx.expand(batch_size, pomo_size, embedding_dim)
        # shape: (batch, pomo, embedding)
        picked_nodes = embeddings.gather(dim=1, index=gathering_index)
        if agent_embeddings is None:
            picked_agents = embeddings.gather(dim=1, index=gathering_agent)
        else:
            picked_agents = agent_embeddings.gather(dim=1, index=gathering_agent)
        if self.problem.NAME == "mtsp":
            return torch.cat((torch.cat((picked_nodes, picked_agents), dim=-1),
                              1.0 - torch.ones(size=state.count_depot.shape, device=embeddings.device) * (state.count_depot + 1) / self.agent_num,
                              state.left_city / self.num_cities), 2)
        elif self.problem.NAME == "mdvrp":
            return torch.cat((torch.cat((picked_nodes, picked_agents), dim=-1),
                              1.0 - torch.ones(size=state.count_depot.shape, device=embeddings.device) * (state.count_depot + 1) / self.agent_num,
                              state.left_city / self.num_cities), 2)
        elif self.problem.NAME == "fmdvrp":
            return torch.cat((torch.cat((picked_nodes, picked_agents), dim=-1),
                              1.0 - torch.ones(size=state.count_depot.shape, device=embeddings.device) * (state.count_depot + 1) / self.agent_num,
                              state.left_city / self.num_cities), 2)
        elif self.problem.NAME == "mpdp":
            return torch.cat((torch.cat((picked_nodes, picked_agents), dim=-1),
                              1.0 - torch.ones(size=state.count_depot.shape, device=embeddings.device) * (state.count_depot + 1) / self.agent_num,
                              state.left_request / self.num_request), 2)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, pomo_size, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        query = self._make_heads(query)
        out = multi_head_attention(query, glimpse_K, glimpse_V)
        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        logits = torch.matmul(self.project_out(out), logit_K.transpose(-2, -1)) / math.sqrt(out.size(-1))
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits + self.dist_alpha_1 * self.curr_dist) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        log_p = torch.log_softmax(logits / self.temp, dim=-1)
        return log_p

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v):
        batch_s = v.size(0)
        n = v.size(1)
        q_reshaped = v.reshape(batch_s, n, self.n_heads, -1)
        # shape: (batch, n, head_num, key_dim)
        q_transposed = q_reshaped.transpose(1, 2)
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
