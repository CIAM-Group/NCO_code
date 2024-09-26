import random

import torch
from logging import getLogger
from torch_geometric.data import Data
from MTSPEnv import MTSPEnv as Env
from PartitionModel import PartitionModel as PartitionModel
from MTSPModel import MTSPModel as Model
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
import numpy as np


class MTSPTesterrrc:
    def __init__(self,
                 env_params,
                 model_params,
                 model_p_params,
                 optimizer_params,
                 trainer_params):

        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.model_p_params = model_p_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model_p = PartitionModel(self.model_p_params['embedding_dim'], 2, 100, 2, depth=self.model_p_params['depth']).cuda()
        self.model_t = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer_t = Optimizer(self.model_t.parameters(), **self.optimizer_params['optimizer'])
        self.optimizer_p = Optimizer(self.model_p.parameters(), **self.optimizer_params['optimizer_p'])
        self.scheduler_t = Scheduler(self.optimizer_t, **self.optimizer_params['scheduler'])
        self.scheduler_p = Scheduler(self.optimizer_p, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['t_enable']:
            checkpoint_fullname = '{t_path}/checkpoint-tsp-{t_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model_t.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['t_epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer_t.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_t.last_epoch = model_load['t_epoch'] - 1
            self.logger.info('Saved TSP Model Loaded !!')

        if model_load['p_enable']:
            checkpoint_fullname = '{p_path}/checkpoint-partition-{p_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model_p.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['p_epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer_p.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_p.last_epoch = model_load['p_epoch'] - 1
            self.logger.info('Saved Partition Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        self.node_coords_500 = self.env.make_dataset_pickle(self.trainer_params['data_load_500'], self.trainer_params['validation_test_episodes'])
        # self.node_coords_1000 = self.env.make_dataset_pickle(self.trainer_params['data_load_1000'], self.trainer_params['validation_test_episodes'])
        self.logger.info('=================================================================')

        self.validation(self.node_coords_500, 500)
        # self.validation(self.node_coords_1000, 1000)
        # LR Decay
        self.scheduler_p.step()
        self.scheduler_t.step()

    def route_ranking2(self, problem, solution, solution_flag):
        roll = ((solution_flag * torch.arange(solution.size(-1))[None, None, :]).max(-1)[1] + 1) % solution.size(-1)
        roll_init = solution.size(-1) - roll[:, :, None]
        roll_diff = (torch.arange(solution.size(-1))[None, None, :].expand_as(solution) + roll[:, :, None]) % solution.size(-1)
        now_solution = solution.gather(-1, roll_diff)
        now_solution_flag = solution_flag.gather(-1, roll_diff)
        solution = now_solution.clone()
        solution_flag = now_solution_flag.clone()
        vector = problem - problem[:, 0, :][:, None, :]
        vector_rank = vector[:, None].repeat(1, solution.size(1), 1, 1).gather(2, solution.unsqueeze(-1).expand(-1, -1, -1, 2))
        solution_start = torch.cummax(solution_flag.roll(dims=-1, shifts=1) * torch.arange(solution.size(-1))[None, :], dim=-1)[0]
        solution_end = solution.size(-1) - 1 - torch.flip(torch.cummax(torch.flip(solution_flag, dims=[-1]) * torch.arange(solution.size(-1))[None, :], dim=-1)[0], dims=[-1])
        num_vector2 = solution_end - solution_start + 1
        cum_vr = torch.cumsum(vector_rank.clone(), dim=-2)
        sum_vector2 = cum_vr.clone().gather(2, solution_end.unsqueeze(-1).expand_as(vector_rank)) - \
                      cum_vr.clone().gather(2, solution_start.unsqueeze(-1).expand_as(vector_rank)) + \
                      vector_rank.clone().gather(2, solution_start.unsqueeze(-1).expand_as(vector_rank))
        vector_angle = torch.atan2(sum_vector2[:, :, :, 1] / num_vector2, sum_vector2[:, :, :, 0] / num_vector2)
        total_indi = vector_angle
        total_rank = np.argsort(total_indi.cpu().numpy(), kind='stable')
        total_rank = torch.from_numpy(total_rank).cuda()
        roll = total_rank.min(-1)[1]
        roll_diff = (torch.arange(solution.size(-1))[None, None, :].expand_as(solution) + roll[:, :, None]) % solution.size(-1)
        now_rank = total_rank.gather(-1, roll_diff)
        # self.env.cal_leagal(problem[:, :, -1], solution, solution_flag)
        solution_rank = solution.gather(-1, now_rank)
        solution_flag_rank = solution_flag.gather(-1, now_rank)
        # self.env.cal_leagal(problem[:, :, -1], solution_rank, solution_flag_rank)
        roll_diff = (torch.arange(solution.size(-1))[None, None, :].expand_as(solution) + roll_init) % solution.size(-1)
        solution_rank = solution_rank.gather(-1, roll_diff)
        solution_flag_rank = solution_flag_rank.gather(-1, roll_diff)
        return solution_rank, solution_flag_rank

    def route_ranking(self, problem, solution, solution_flag):
        roll = ((solution_flag * torch.arange(solution.size(-1))[None, :]).max(-1)[1] + 1) % solution.size(-1)
        roll_init = solution.size(-1) - roll[:, None]
        roll_diff = (torch.arange(solution.size(-1))[None, :].expand_as(solution) + roll[:, None]) % solution.size(-1)
        now_solution = solution.gather(1, roll_diff)
        now_solution_flag = solution_flag.gather(1, roll_diff)
        solution = now_solution.clone()
        solution_flag = now_solution_flag.clone()
        vector = problem - problem[:, 0, :][:, None, :]
        vector_rank = vector.repeat(solution.size(0), 1, 1).gather(1, solution.unsqueeze(-1).expand(-1, -1, 2))
        solution_start = torch.cummax(solution_flag.roll(dims=1, shifts=1) * torch.arange(solution.size(-1))[None, :], dim=-1)[0]
        solution_end = solution.size(-1) - 1 - torch.flip(torch.cummax(torch.flip(solution_flag, dims=[1]) * torch.arange(solution.size(-1))[None, :], dim=-1)[0], dims=[1])
        num_vector2 = solution_end - solution_start + 1
        cum_vr = torch.cumsum(vector_rank.clone(), dim=1)
        sum_vector2 = cum_vr.clone().gather(1, solution_end.unsqueeze(-1).expand_as(vector_rank)) - \
                      cum_vr.clone().gather(1, solution_start.unsqueeze(-1).expand_as(vector_rank)) + \
                      vector_rank.clone().gather(1, solution_start.unsqueeze(-1).expand_as(vector_rank))
        vector_angle = torch.atan2(sum_vector2[:, :, 1] / num_vector2, sum_vector2[:, :, 0] / num_vector2)
        total_indi = vector_angle
        total_rank = np.argsort(total_indi.cpu().numpy(), kind='stable')
        total_rank = torch.from_numpy(total_rank).cuda()
        roll = total_rank.min(-1)[1]
        roll_diff = (torch.arange(solution.size(-1))[None, :].expand_as(solution) + roll[:, None]) % solution.size(-1)
        now_rank = total_rank.gather(1, roll_diff)
        # self.env.cal_leagal(problem[:, :, -1], solution, solution_flag)
        solution_rank = solution.gather(1, now_rank)
        solution_flag_rank = solution_flag.gather(1, now_rank)
        # self.env.cal_leagal(problem[:, :, -1], solution_rank, solution_flag_rank)
        roll_diff = (torch.arange(solution.size(-1))[None, :].expand_as(solution) + roll_init) % solution.size(-1)
        solution_rank = solution_rank.gather(1, roll_diff)
        solution_flag_rank = solution_flag_rank.gather(1, roll_diff)
        return solution_rank, solution_flag_rank

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

    def gen_distance_matrix(self, coordinates):
        distances = torch.cdist(coordinates, coordinates, p=2)
        return distances

    def gen_pyg_data(self, coors, k_sparse=100):
        coors_with_M = torch.cat((coors[:, 0:1, :].repeat(1, self.env.M_number + 1, 1), coors[:, 1:, :]), dim=1)
        bs = coors_with_M.size(0)
        n_nodes = coors_with_M.size(1)
        cos_mat = -1 * self.gen_distance_matrix(coors_with_M)
        x = coors_with_M
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

    def validation(self, data, scale):
        a = self.env.pomo_size
        self.env.pomo_size = 1
        self.time_estimator.reset()
        self.model_t.eval()
        self.model_p.eval()
        if scale == 500:
            self.env.M_number = 50
        if scale == 1000:
            self.env.M_number = 50
        self.model_t.model_params['eval_type'] = 'argmax'
        solution_list, solution_flag_list = self._load_init_sol(data)
        solution = torch.stack(solution_list, dim=0)
        solution_flag = torch.stack(solution_flag_list, dim=0)
        test_num_episode = self.trainer_params['validation_test_episodes']
        k = 0
        while k < 50:
            solution, solution_flag = self.route_ranking2(data, solution, solution_flag)
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(self.trainer_params['validation_test_batch_size'], remaining)
                solution, solution_flag, score, aug_score = self._test_one_batch(solution, solution_flag, data, batch_size, episode, k)
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("iter {:2d}, episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(k,
                                                                                                                               episode, test_num_episode,
                                                                                                                               elapsed_time_str,
                                                                                                                               remain_time_str, score_AM.avg,
                                                                                                                               aug_score_AM.avg))

        self.logger.info(" *** Validation Scale " + str(scale) + " Done *** ")
        self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
        self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

        cost_file = open(self.result_folder + '/curve' + str(scale) + '.txt', mode='a+')
        cost_file.write(str(score_AM.avg) + ' ' + str(aug_score_AM.avg) + '\n')
        self.model_t.model_params['eval_type'] = 'softmax'
        self.env.pomo_size = a

    def _load_init_sol(self, data):
        solution_list = []
        solution_flag_list = []
        for i in range(self.trainer_params['validation_test_episodes']):
            self.env.load_raw_problems(1, i, data)
            pyg_data = self.gen_pyg_data(self.env.raw_problems)
            agent_per = torch.arange(self.env.M_number).cuda()[None, :].repeat(self.trainer_params['validation_aug_factor'], 1)
            if (self.trainer_params['validation_aug_factor'] > 1):
                for i in range(100):
                    a = torch.randint(0, self.env.M_number, (self.trainer_params['validation_aug_factor'],)).cuda()
                    b = torch.randint(0, self.env.M_number, (self.trainer_params['validation_aug_factor'],)).cuda()
                    p = agent_per[torch.arange(self.trainer_params['validation_aug_factor']), a].clone()
                    q = agent_per[torch.arange(self.trainer_params['validation_aug_factor']), b].clone()
                    agent_per = agent_per.scatter(dim=1, index=b[:, None], src=p[:, None])
                    agent_per = agent_per.scatter(dim=1, index=a[:, None], src=q[:, None])
            index = agent_per[:, 0]
            situ = torch.ones(self.trainer_params['validation_aug_factor'], dtype=torch.long)
            left_depot = torch.ones(self.trainer_params['validation_aug_factor'], dtype=torch.long) * (self.env.M_number - 1)
            left_city = torch.ones(self.trainer_params['validation_aug_factor'], dtype=torch.long) * self.env.raw_problems.size(1)
            visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1) + self.env.M_number)
            visited = visited.scatter(-1, index[:, None], 1)
            selected = index[:, None]
            solution = torch.zeros((self.trainer_params['validation_aug_factor'], self.env.raw_problems.size(1)), dtype=torch.long)
            solution_flag = torch.zeros((self.trainer_params['validation_aug_factor'], self.env.raw_problems.size(1)), dtype=torch.long)
            node_count = -1 * torch.ones((self.trainer_params['validation_aug_factor'], 1), dtype=torch.long)
            heatmap = None
            step = 0
            self.model_p.pre(pyg_data, self.env.M_number)
            while (left_depot > 0).any() or (left_city > 0).any():
                if step % self.env.problem_size == 0:
                    node_emb, heatmap = self.model_p(solution, visited)
                    heatmap = heatmap / (heatmap.min() + 1e-5)
                    heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5
                depot_mask = torch.zeros_like(visited)
                crt_depot = agent_per.gather(1, situ.clamp_max_(self.env.M_number - 1)[:, None])
                depot_mask[:, :self.env.M_number] = 1
                situ_unmask = (selected.squeeze(-1) >= self.env.M_number)
                depot_mask[situ_unmask] = depot_mask[situ_unmask].scatter(1, crt_depot[situ_unmask], 0)
                row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone() * (1 - depot_mask).clone()
                row[left_depot >= left_city, self.env.M_number:] = 0.
                dist = Categorical(row)
                item = dist.sample()  # row.argmax(dim=-1)  #
                selected = item[:, None]  # row.reshape(batch_size * row.size(1), -1).multinomial(1).squeeze(dim=1).reshape(batch_size, row.size(1))[:, :, None]
                visited = visited.scatter(-1, selected, 1)
                left_depot -= (item < self.env.M_number).long()
                situ += (item < self.env.M_number).long()
                left_city -= (item >= self.env.M_number).long()
                step += 1
                if step > 1:
                    solution_flag = solution_flag.scatter_add(dim=-1, index=node_count, src=(selected < self.env.M_number).long())
                node_count[selected >= self.env.M_number] += 1
                solution = solution.scatter_add(dim=-1, index=node_count, src=(selected - self.env.M_number).clamp_min_(0))
            solution_flag[:, -1] = 1
            solution_flag_list.append(solution_flag)
            solution_list.append(solution)
        return solution_list, solution_flag_list

    def _test_one_batch(self, solution_gnn, solution_flag_gnn, data, batch_size, episode, k):
        aug_factor = self.trainer_params['validation_aug_factor']
        solution = solution_gnn.clone()[episode:episode + batch_size]
        solution_flag = solution_flag_gnn.clone()[episode:episode + batch_size]
        now_problem = data.clone()[episode:episode + batch_size]
        if k <= 4:
            roll = self.env.problem_size // 2
        else:
            roll = random.randint(1, self.env.problem_size)
        if k % 2:
            solution = solution.clone().roll(dims=-1, shifts=roll)
            solution_flag = solution_flag.clone().roll(dims=-1, shifts=roll)
        front_depot, next_depot = self.env.cal_info2(now_problem, solution, solution_flag)
        front_depot_length = front_depot.view(solution.size(0) * solution.size(1), -1, self.env.problem_size)[:, :, 0].reshape(-1)
        next_depot_length = next_depot.view(solution.size(0) * solution.size(1), -1, self.env.problem_size)[:, :, -1].reshape(-1)
        n_tsps_per_route = solution.view(solution.size(0), -1, self.env.problem_size)
        tsp_insts = now_problem[:, None, :].repeat(1, n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 2))
        n_tsps_per_route_flag = solution_flag.view(solution.size(0), -1, self.env.problem_size).view(-1, self.env.problem_size)
        tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
        add_depot_insts_now = torch.cat((now_problem[:, 0, :][:, None, None, :].repeat(1, tsp_insts.size(1), 1, 1), tsp_insts), dim=2)
        tsp_insts_now_temp = add_depot_insts_now.view(-1, add_depot_insts_now.size(-2), add_depot_insts_now.size(-1))
        route_num = n_tsps_per_route_flag[:, :-1].sum(-1) + 1
        solution_now = torch.arange(1, tsp_insts_now_temp.size(-2))[None, :].expand((tsp_insts_now.size(0), -1))[:, None, :]
        reward_now = self.env.get_open_travel_distance(tsp_insts_now_temp, solution_now, n_tsps_per_route_flag[:, None, :], front_depot_length, next_depot_length)
        new_batch_size = tsp_insts_now.size(0)
        tsp_insts_now_input = torch.cat(
            (now_problem[:, 0, :][:, None, None, :].repeat(1, tsp_insts.size(1), route_num.max(-1)[0], 1).view(-1, route_num.max(-1)[0], 2), tsp_insts_now), dim=1)
        tsp_insts_now_norm = self.coordinate_transformation(tsp_insts_now_input)

        self.env.load_problems(new_batch_size, tsp_insts_now_norm, route_num, n_tsps_per_route_flag[:, -1], front_depot_length, next_depot_length)
        reset_state, _, _ = self.env.reset()
        self.model_t.pre_forward(reset_state)
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            cur_dist = self.env.get_local_feature()
            selected, prob = self.model_t(state, cur_dist, n_tsps_per_route_flag[:, -1])
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
        # Loss
        ###############################################
        # shape: (batch, pomo)
        selected_solution = self.env.solution_list[:, :, :-1]
        selected_flag = self.env.solution_flag[:, :, :-1]
        selected_flag[:, :, -1] = n_tsps_per_route_flag[:, -1][:, None].repeat(1, self.env.pomo_size)

        reward = self.env.get_open_travel_distance(tsp_insts_now_temp, selected_solution + 1, selected_flag, front_depot_length, next_depot_length)
        # Loss
        ###############################################
        tag = reward.view(batch_size, aug_factor, -1, self.env.pomo_size).min(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
        tag_solution = selected_solution.view(batch_size, aug_factor, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze(3)
        tag_solution_flag = selected_flag.view(batch_size, aug_factor, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze(3)
        r = (reward.min(1)[0] > reward_now.squeeze()).view(solution.size(0), aug_factor, -1, 1).expand((-1, -1, -1, tsp_insts_now.size(-2)))
        route_tag = torch.arange(k % 2, r.size(2) - 1 + k % 2, 2)[None, :, None].expand((solution.size(0), aug_factor, -1, tsp_insts_now.size(-2)))
        r_tag = torch.ones_like(r).bool().clone().scatter(2, route_tag, False)
        tag_solution[r | r_tag] = (solution_now - 1).view(batch_size, aug_factor, -1, tsp_insts_now.size(-2))[r | r_tag]
        tag_solution_flag[r | r_tag] = n_tsps_per_route_flag.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2))[r | r_tag]
        solution = n_tsps_per_route.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2)).gather(-1, tag_solution).view(solution.size(0), solution.size(1), -1)
        solution_flag = tag_solution_flag.view(solution.size(0), solution.size(1), -1)
        solution_out = solution_gnn.clone()
        solution_out_flag = solution_flag_gnn.clone()
        solution_out[episode:episode + batch_size] = solution
        solution_out_flag[episode:episode + batch_size] = solution_flag
        merge_reward_0 = self.env._get_travel_distance2(now_problem, solution, solution_flag)

        # order_node_ = solution.clone()
        # order_flag_ = solution_flag.clone()
        # index_small = torch.le(order_flag_, 0.5)
        # index_bigger = torch.gt(order_flag_, 0.5)
        # order_flag_[index_small] = order_node_[index_small]
        # order_flag_[index_bigger] = 0
        # solution2 = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        # flag2 = torch.stack((torch.zeros_like(solution_flag.clone()), solution_flag.clone()), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        # id = solution2[0, 0].clone()
        # flag2 = flag2[0, 0].clone()
        # max2 = 0
        # length = []
        # now = 0
        # dist = torch.cdist(now_problem, now_problem, p=2)[0]
        # for i in range(1, id.shape[0]):
        #     now += dist[id[i - 1], id[i]].item()
        #     if (flag2[i] == 1):
        #         length.append(now)
        #         now = 0
        # length.append(now)
        # length[0] += length[-1]
        # length[0] += dist[id[0], id[-1]].item()
        # max2 = max(length)
        # merge_reward_0 = self.env._get_travel_distance2(now_problem, solution, solution_flag)
        '''
        best = merge_reward_0.min(-1)[1].squeeze()[0].item()
        if k == 2 or k == 50 or k == 250:
            id = solution[0, best].clone()
            flag = solution_flag[0, best].clone()
            index_small = torch.le(flag, 0.5)
            index_bigger = torch.gt(flag, 0.5)
            flag[index_small] = id[index_small]
            flag[index_bigger] = 0
            id = torch.stack((id, flag), dim=-1).view(-1).cpu().numpy()
            colorboard = np.array(sns.color_palette("deep", 500))
            color = 0
            my_colors = colorboard[color]
            for i in range(1, id.shape[0]):
                place = now_problem.cpu().numpy()[0, id[i], :]
                place2 = now_problem.cpu().numpy()[0, id[i - 1], :]
                plt.scatter(place2[0], place2[1], color='b', s=3)
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
                if id[i] == 0:
                    color += 1
                    my_colors = colorboard[color]
            place = now_problem.cpu().numpy()[0, id[0], :]
            place2 = now_problem.cpu().numpy()[0, id[- 1], :]
            plt.scatter(place2[0], place2[1], color='b', s=3)
            plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
            for i in range(1, id.shape[0]):
                place = now_problem.cpu().numpy()[0, id[i], :]
                place2 = now_problem.cpu().numpy()[0, id[i - 1], :]
                plt.scatter(place2[0], place2[1], color='b', s=3)
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
                if id[i] == 0:
                    break
            plt.show()
        '''
        return solution_out, solution_out_flag, merge_reward_0[:, 0].mean(0).item(), merge_reward_0.min(1)[0].mean(0).item()  # merge_reward_1.min(1)[0].mean(0).item()
