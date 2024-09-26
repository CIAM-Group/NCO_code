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
        self.node_coords, self.tours = self.env.make_dataset(self.tester_params['data_load'], self.tester_params['test_episodes'])
        # solution_list = self._init_insertion()
        solution_list = self._load_init_sol(self.node_coords)
        solution = torch.stack(solution_list, dim=0)
        test_num_episode = self.tester_params['test_episodes']
        k = 0
        while k < 251:
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(self.tester_params['test_batch_size'], remaining)
                solution, score, aug_score = self._test_one_batch(solution, batch_size, episode, k)
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size
                ############################
                # Logs
                ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("iter {:2d}, episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.4f}, aug_score:{:.4f}".format(k,
                                                                                                                               episode, test_num_episode, elapsed_time_str,
                                                                                                                               remain_time_str, score_AM.avg, aug_score_AM.avg))

        self.logger.info(" *** Test Done *** ")
        self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
        self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

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

    def _load_init_sol(self, data):
        solution_list = []
        for i in range(self.tester_params['test_episodes']):
            self.env.load_raw_problems(1, i, data)
            pyg_data = self.gen_pyg_data(self.env.raw_problems)
            index = torch.randint(0, self.env.raw_problems.size(1), [self.tester_params['aug_factor']])
            visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1))
            solution = index[:, None]
            visited = visited.scatter(-1, solution[:, 0:1], 1)
            selected = solution
            heatmap = None
            self.model_p.pre(pyg_data)
            while solution.size(-1) < self.env.raw_problems.size(1):
                if (solution.size(-1) - 1) % self.env.problem_size == 0:
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

    def _test_one_batch(self, solution_gnn, batch_size, episode, k):
        aug_factor = self.tester_params['aug_factor']
        solution = solution_gnn.clone()[episode:episode + batch_size]
        now_problem = self.node_coords.clone()[episode:episode + batch_size]
        # now_optimal_tour = self.tours.clone()[episode:episode + batch_size]
        if k == 1 or k == 2:
            roll = self.env.problem_size // 2
        else:
            roll = random.randint(1, self.env.problem_size)
        solution = solution.roll(dims=-1, shifts=roll)
        n_tsps_per_route = solution.clone().view(solution.size(0), -1, self.env.problem_size)
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
        merge_solution = n_tsps_per_route.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2)).gather(-1, tag_solution).view(solution.size(0), solution.size(1), -1)
        solution = merge_solution.clone()
        merge_reward_0 = self.env._get_travel_distance2(now_problem, solution)
        solution_out = solution_gnn.clone()
        solution_out[episode:episode + batch_size] = solution
        '''Codes for printing solutions
        if k == 2 or k == 50 or k == 250: # 24.1011, 23.7346 23.7338
            best = merge_reward_0.min(-1)[1].item()
            plt.rcParams['pdf.use14corefonts'] = True
            plt.axis('off')
            tsp_show = tsp_insts_now.clone().cpu().numpy()
            #id = merge_solution[0, best].clone().cpu().numpy()
            id = self.tours[0].cpu().numpy()
            for i in range(1, merge_solution.size(-1)):
                place = now_problem.cpu().numpy()[0, id[i], :]
                place2 = now_problem.cpu().numpy()[0, id[i - 1], :]
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color='r', linewidth=1.6)
            place = now_problem.cpu().numpy()[0, id[0], :]
            place2 = now_problem.cpu().numpy()[0, id[- 1], :]
            plt.plot([place[0], place2[0]], [place[1], place2[1]], color='r', linewidth=1.6)
            for i in range(tag_solution.size(-2)):
                plt.scatter(tsp_show[i, :, 0], tsp_show[i, :, 1], color='grey', s=20)
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

