import torch
import math
import numpy as np
from torch.nn import DataParallel
from problems.problem_tsp import TSP
from problems.problem_vrp import CVRP


def selposition_encoding(solutions, embedding_dim, device):
    batch_size, problem_size, _ = solutions.size()
    enc_pattern = position_encoding_init(problem_size, embedding_dim)
    position_enc = enc_pattern.expand(batch_size, problem_size, embedding_dim).to(device)
    return position_enc


def position_encoding_init(n_position, emb_dim):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        for pos in range(1, n_position + 1)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def position_encoding(graph, solutions, embedding_dim, device):
    batch_size, _, graph_size, _ = graph.shape
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(graph_size)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor).to(device)

    return position_enc[None, :, :].repeat(batch_size, 1, 1)


def load_problem(name):
    problem = {
        'tsp': TSP,
        'vrp': CVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped
