import random

import torch
from torch_geometric.data import Data
from PartitionModel import PartitionModel as PartitionModel
import os
import numpy as np
from logging import getLogger
import matplotlib.pyplot as plt

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *
import math


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 model_p_params,
                 tester_params):

        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        # save arguments
        self.env_params = env_params
        self.model_p_params = model_p_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.model_p = PartitionModel(self.model_p_params['embedding_dim'], 2, 100, 2, depth=self.model_p_params['depth']).cuda()
        self.model_t = Model(**self.model_params)
        self.env = Env(**self.env_params)

        # Restore
        model_load = tester_params['model_load']
        if model_load['t_enable']:
            checkpoint_fullname = '{t_path}/checkpoint-tsp-{t_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model_t.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['t_epoch']
            self.logger.info('Saved TSP Model Loaded !!')

        if model_load['p_enable']:
            checkpoint_fullname = '{p_path}/checkpoint-partition-{p_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model_p.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['p_epoch']
            self.logger.info('Saved Partition Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.model_t.eval()
        self.model_p.eval()
        self.node_coords, self.costs, self.instance_name = self.env.make_tsplib_data('tsp_lib.txt')
        # solution_list = self._init_insertion()
        test_num_episode = self.tester_params['test_episodes']
        for i in range(len(self.node_coords)):
            now_data = self.node_coords[i].clone()
            if now_data.size(0) > 3500:
                self.tester_params['aug_factor'] = 20
            now_data[:, 0] -= now_data[:, 0].min(-1)[0]
            now_data[:, 1] -= now_data[:, 1].min(-1)[0]
            factor = max(self.node_coords[i][:, 0].max(-1)[0], self.node_coords[i][:, 1].max(-1)[0])
            now_data /= factor
            solution_list = self._load_init_sol(now_data[None, :], i)
            solution = torch.stack(solution_list, dim=0)
            k = 0
            while k < 2:
                k += 1
                score_AM = AverageMeter()
                aug_score_AM = AverageMeter()
                solution, score, aug_score = self._test_one_batch(solution, now_data[None, :], i, k)
                score_AM.update(score, 1)
                aug_score_AM.update(aug_score, 1)
                ############################
                # Logs
                ############################
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(1, 1)
                self.logger.info("iter {:2d}, episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}, Gap:{:.3f} %".format(k,
                                                                                                                                                 1, test_num_episode,
                                                                                                                                                 elapsed_time_str,
                                                                                                                                                 remain_time_str,
                                                                                                                                                 score_AM.avg * factor,
                                                                                                                                                 aug_score_AM.avg * factor,
                                                                                                                                                 (aug_score_AM.avg * factor /
                                                                                                                                                  self.costs[
                                                                                                                                                      i] - 1) * 100))

            self.logger.info(" *** Test Done *** ")
            self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
            self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
            cost_file = open('lib_result.txt', mode='a+')
            cost_file.write(self.instance_name[i] + ' ' + str(score_AM.avg * factor.item()) + ' ' + str(aug_score_AM.avg * factor.item()) + ' ' + str(
                (aug_score_AM.avg * factor.item() / self.costs[i].item() - 1) * 100) + '\n')

    def coordinate_transformation(self, x):
        input = x.clone()
        max_x, indices_max_x = input[:, :, 0].max(dim=1)
        max_y, indices_max_y = input[:, :, 1].max(dim=1)
        min_x, indices_min_x = input[:, :, 0].min(dim=1)
        min_y, indices_min_y = input[:, :, 1].min(dim=1)
        # shapes: (batch_size, ); (batch_size, )

        diff_x = max_x - min_x
        diff_y = max_y - min_y
        xy_exchanged = diff_y > diff_x

        # shift to zero
        input[:, :, 0] -= (min_x).unsqueeze(-1)
        input[:, :, 1] -= (min_y).unsqueeze(-1)

        # exchange coordinates for those diff_y > diff_x
        input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] = input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]

        # scale to (0, 1)
        scale_degree = torch.max(diff_x, diff_y)
        scale_degree = scale_degree.view(input.shape[0], 1, 1)
        input /= scale_degree + 1e-10
        return input

    def _load_init_sol(self, data, i):
        solution_list = []
        self.env.load_raw_problems(1, 0, data)
        pyg_data = self.gen_pyg_data(self.env.raw_problems)
        index = torch.randint(0, self.env.raw_problems.size(1), [self.tester_params['aug_factor']])
        visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1))
        solution = index[:, None]
        visited = visited.scatter(-1, solution[:, 0:1], 1)
        selected = solution
        heatmap = None
        self.model_p.pre(pyg_data)
        if data.size(1) > 2000:
            subsize = self.env.raw_problems.size(1) // 10
        else:
            subsize = self.env.problem_size
        while solution.size(-1) < self.env.raw_problems.size(1):
            if (solution.size(-1) - 1) % subsize == 0:
                node_emb, heatmap = self.model_p(solution, visited)
                heatmap = heatmap / (heatmap.min() + 1e-5)
                heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5
            row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone()
            item = row.max(-1)[1]
            selected = item[:, None]
            visited = visited.scatter(-1, selected, 1)
            solution = torch.cat((solution, selected), dim=-1)
        solution_list.append(solution)
        return solution_list

    def _test_one_batch(self, solution_gnn, data, i, k):
        aug_factor = self.tester_params['aug_factor']
        solution = solution_gnn.clone()
        batch_size = 1
        now_problem = data.clone()
        if k == 1 or k == 2:
            roll = self.env.problem_size // 2
        else:
            roll = random.randint(1, self.env.problem_size)
        solution = solution.roll(dims=-1, shifts=roll)
        solving_length = ((solution.size(-1)) // self.env.problem_size) * self.env.problem_size
        solution_cut = solution[:, :, :solving_length]
        n_tsps_per_route = solution_cut.clone().view(solution.size(0), -1, self.env.problem_size)
        tsp_insts = now_problem[:, None, :, :].repeat(1, n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 2))
        tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
        solution_now = torch.arange(tsp_insts_now.size(-2))[None, :].expand((tsp_insts_now.size(0), -1))[:, None, :]
        reward_now = self.env.get_open_travel_distance(tsp_insts_now, solution_now)
        new_batch_size = tsp_insts_now.size(0)
        tsp_insts_now_norm = self.coordinate_transformation(tsp_insts_now)
        self.env.load_problems(new_batch_size, tsp_insts_now_norm)
        reset_state, _, _ = self.env.reset()
        self.model_t.pre_forward(reset_state)
        # shape: (batch, pomo, 0~problem)
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            cur_dist = self.env.get_local_feature()
            selected, prob = self.model_t(state, cur_dist)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        selected = torch.cat((self.env.problem_size - torch.ones(self.env.pomo_size // 2)[None, :].expand(new_batch_size, self.env.pomo_size // 2),
                              torch.zeros(self.env.pomo_size // 2)[None, :].expand(new_batch_size, self.env.pomo_size // 2)), dim=-1).long()
        state, reward, done = self.env.step(selected)
        reward = self.env.get_open_travel_distance(tsp_insts_now, self.env.selected_node_list)
        # Loss
        ###############################################
        tag = reward.view(batch_size, aug_factor, -1, self.env.pomo_size).min(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
        tag_solution = self.env.selected_node_list.view(batch_size, aug_factor, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze(3)
        reversed_tag_solution = torch.flip(tag_solution.clone(), dims=[-1])
        tag_solution[tag.squeeze(3) >= self.env.pomo_size / 2] = reversed_tag_solution[tag.squeeze(3) >= self.env.pomo_size / 2]
        r = (reward.min(1)[0] > reward_now.squeeze()).view(batch_size, aug_factor, -1, 1).expand((-1, -1, -1, tsp_insts_now.size(-2)))
        tag_solution[r] = solution_now.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2))[r]
        merge_solution = n_tsps_per_route.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2)).gather(-1, tag_solution).view(solution.size(0), solution_cut.size(1), -1)
        solution = torch.cat((merge_solution.clone(), solution[:, :, solving_length:]), dim=-1)
        merge_reward_0 = self.env._get_travel_distance2(now_problem, solution)
        solution_out = solution.clone()
        '''
        id = solution[0, 0].clone().cpu().numpy()
        for i in range(1, solution.size(-1)):
            place = now_problem.cpu().numpy()[0, id[i], :]
            place2 = now_problem.cpu().numpy()[0, id[i - 1], :]
            plt.scatter(place2[0], place2[1], color='b', s=5)
            plt.plot([place[0], place2[0]], [place[1], place2[1]], color='r', linewidth=0.5)
        place = now_problem.cpu().numpy()[0, id[0], :]
        place2 = now_problem.cpu().numpy()[0, id[- 1], :]
        plt.scatter(place2[0], place2[1], color='b', s=5)
        plt.plot([place[0], place2[0]], [place[1], place2[1]], color='r', linewidth=0.5)
        plt.show()
        '''

        return solution_out, merge_reward_0[:, 0].mean(0).item(), merge_reward_0.min(1)[0].mean(0).item()  # merge_reward_1.min(1)[0].mean(0).item()

    def gen_distance_matrix(self, coordinates):
        distances = torch.cdist(coordinates, coordinates, p=2)
        return distances

    def gen_pyg_data(self, coors, k_sparse=100):
        bs = coors.size(0)
        n_nodes = coors.size(1)
        cos_mat = -1 * self.gen_distance_matrix(coors)
        x = coors
        topk_values, topk_indices = torch.topk(cos_mat,
                                               k=100,
                                               dim=2, largest=True)
        edge_index = torch.cat(
            (torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse)[None, :].repeat(bs, 1)[:, None, :], topk_indices.view(bs, -1)[:, None, :]),
            dim=1)
        idx = torch.arange(bs)[:, None, None].repeat(1, n_nodes, k_sparse)
        edge_attr1 = topk_values.reshape(bs, -1, 1)
        edge_attr2 = cos_mat[idx.view(bs, -1), edge_index[:, 0], edge_index[:, 1]].reshape(bs, k_sparse * n_nodes, 1)
        edge_attr = torch.cat((edge_attr1, edge_attr2), dim=2)
        pyg_data = Data(x=x[0], edge_index=edge_index[0], edge_attr=edge_attr[0])
        return pyg_data

