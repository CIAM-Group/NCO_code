import torch
from logging import getLogger
from torch_geometric.data import Data
import math
import numpy as np
from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from PartitionModel import PartitionModel as PartitionModel
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold
import random
import seaborn as sns


class CVRPLiber:
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
        self.model_p = PartitionModel(self.model_p_params['embedding_dim'], 3, 100, 2, depth=self.model_p_params['depth']).cuda()
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
        self.node_coords, self.node_demands, self.costs = self.env.make_dataset_lib('lib data.txt')

        for epoch in range(31, 32):
            self.logger.info('=================================================================')

            self.validation(self.node_coords[epoch][None, :], self.node_demands[epoch][None, :], epoch)
            # self.validation(self.node_coords_2000.clone(), self.node_demands_2000.clone(), 2000)
            # LR Decay

            # Train
            ############################
            # Logs & Checkpoint
            ############################

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        loss_P = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, _ = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            loss_P.update(_, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  Loss_P: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg, loss_P.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Loss_P: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg, loss_P.avg))

        return score_AM.avg, loss_AM.avg

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
        input[:, :, :2] /= scale_degree + 1e-10
        return input

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

    def gen_distance_matrix(self, coordinates):
        distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
        return distances

    def gen_cos_sim_matrix(self, shift_coors):
        dot_products = torch.bmm(shift_coors, shift_coors.transpose(1, 2))
        magnitudes = torch.sqrt(torch.sum(shift_coors ** 2, dim=-1)).unsqueeze(-1)
        magnitude_matrix = torch.bmm(magnitudes, magnitudes.transpose(1, 2)) + 1e-10
        cosine_similarity_matrix = dot_products / magnitude_matrix
        return cosine_similarity_matrix

    def gen_pyg_data(self, coors, demand, k_sparse=100, cvrplib=False):
        bs = demand.size(0)
        n_nodes = demand.size(1)
        norm_demand = demand
        shift_coors = coors - coors[:, 0, :]
        _x, _y = shift_coors[:, :, 0], shift_coors[:, :, 1]
        r = torch.sqrt(_x ** 2 + _y ** 2)
        theta = torch.atan2(_y, _x)
        x = torch.stack((norm_demand, r, theta)).permute(1, 2, 0)
        cos_mat = self.gen_cos_sim_matrix(shift_coors)
        if cvrplib:
            cos_mat = (cos_mat + cos_mat.min()) / cos_mat.max()
            euc_mat = self.gen_distance_matrix(coors)
            euc_aff = 1 - euc_mat
            topk_values, topk_indices = torch.topk(cos_mat + euc_aff,
                                                   k=k_sparse,
                                                   dim=1, largest=True)
            edge_index = torch.stack([
                torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                        repeats=k_sparse),
                torch.flatten(topk_indices)
            ])
            edge_attr1 = euc_aff[edge_index[0], edge_index[1]].reshape(k_sparse * n_nodes, 1)
            edge_attr2 = cos_mat[edge_index[0], edge_index[1]].reshape(k_sparse * n_nodes, 1)
        else:
            topk_values, topk_indices = torch.topk(cos_mat,
                                                   k=100,
                                                   dim=2, largest=True)
            edge_index = torch.cat(
                (
                    torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse)[None, :].repeat(bs, 1)[:, None, :],
                    topk_indices.view(bs, -1)[:, None, :]),
                dim=1)
            idx = torch.arange(bs)[:, None, None].repeat(1, n_nodes, k_sparse)
            edge_attr1 = topk_values.reshape(bs, -1, 1)
            edge_attr2 = cos_mat[idx.view(bs, -1), edge_index[:, 0], edge_index[:, 1]].reshape(bs, k_sparse * n_nodes, 1)
        edge_attr = torch.cat((edge_attr1, edge_attr2), dim=2)
        pyg_data = Data(x=x[0], edge_index=edge_index[0], edge_attr=edge_attr[0])
        return pyg_data

    def validation(self, data, demand, i):
        # data[:, 0] -= data[:, 0].min(-1)[0]
        # data[:, 1] -= data[:, 1].min(-1)[0]
        factor = 1000
        data /= factor
        a = self.env.pomo_size
        self.env.pomo_size = 1
        self.time_estimator.reset()
        self.model_t.eval()
        self.model_p.eval()
        self.model_t.model_params['eval_type'] = 'argmax'
        solution_list, solution_flag = self._load_init_sol(data, demand)
        solution = torch.stack(solution_list, dim=0)
        solution_flag = torch.stack(solution_flag, dim=0)
        test_num_episode = self.trainer_params['validation_test_episodes']
        k = 0
        while k < 251:

            solution, solution_flag = self.route_ranking2(data, solution, solution_flag)
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(self.trainer_params['validation_test_batch_size'], remaining)
                solution, solution_flag, score, aug_score = self._test_one_batch(solution, solution_flag, data, demand, batch_size, episode, k)
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("iter {:2d}, episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}, gap:{:.3f}".format(k,
                                                                                                                                           episode, test_num_episode,
                                                                                                                                           elapsed_time_str,
                                                                                                                                           remain_time_str, score_AM.avg,
                                                                                                                                           aug_score_AM.avg,
                                                                                                                                           (aug_score_AM.avg * factor / self.costs[
                                                                                                                                               i]) - 1))

        self.logger.info(" *** Validation Scale Done *** ")
        self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
        self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
        self.logger.info(" Gap: {:.4f} %".format(((aug_score_AM.avg * factor / self.costs[i]) - 1) * 100))
        self.model_t.model_params['eval_type'] = 'softmax'
        self.env.pomo_size = a

        cost_file = open('lib_result.txt', mode='a+')
        cost_file.write(str(aug_score_AM.avg) + ' ' + str(((aug_score_AM.avg * factor / self.costs[i]) - 1)) + '\n')
        self.model_t.model_params['eval_type'] = 'softmax'
        self.env.pomo_size = a

    def _load_init_sol(self, data, demand):
        solution_list = []
        solution_flag_list = []
        for j in range(self.trainer_params['validation_test_episodes']):
            self.env.load_raw_problems(1, j, data, demand)
            pyg_data = self.gen_pyg_data(self.env.raw_depot_node_xy, self.env.raw_depot_node_demand)
            index = torch.zeros(self.trainer_params['validation_aug_factor'], dtype=torch.long, device=data.device)
            vehicle_count = torch.zeros((self.trainer_params['validation_aug_factor'],), device=index.device)
            demand_count = torch.zeros((self.trainer_params['validation_aug_factor'],), device=index.device)
            visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1))
            solution_raw = index[:, None]
            visited = visited.scatter(-1, solution_raw[:, 0:1], 1)
            selected = solution_raw
            self.model_p.pre(pyg_data)
            max_vehicle = (self.env.raw_depot_node_demand.sum().ceil() + 1).item()
            total_demand = self.env.raw_depot_node_demand.sum()
            remaining_demand = total_demand.clone()
            solution = torch.zeros((self.trainer_params['validation_aug_factor'], self.env.raw_problems.size(1) - 1), dtype=torch.long)
            solution[:, 0] = solution_raw[:, 0]
            step = 0
            solution_flag = torch.zeros((self.trainer_params['validation_aug_factor'], self.env.raw_problems.size(1) - 1), dtype=torch.long)
            node_count = -1 * torch.ones((self.trainer_params['validation_aug_factor'], 1), dtype=torch.long)
            capacity = torch.ones_like(index)[:, None].float()
            capacity -= self.env.raw_depot_node_demand.expand((self.trainer_params['validation_aug_factor'], -1)).gather(-1, selected)
            if data.size(1) > 2000:
                subsize = data.size(1) // 10
            else:
                subsize = self.env.problem_size
            for i in range(((data.size(1) - 1) // subsize) + 1):
                node_emb, heatmap = self.model_p(solution, selected, visited)
                heatmap = heatmap / (heatmap.min() + 1e-5)
                heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5
                while ((visited.sum(-1) - i * subsize) < subsize).any():
                    step += 1
                    capacity_mask = (self.env.raw_depot_node_demand > capacity + 1e-5).long().squeeze()
                    row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone() * (1 - capacity_mask).clone()
                    row[:, 0][selected[:, 0] == 0] = 0
                    row[:, 0][remaining_demand > (max_vehicle - vehicle_count)] = 0
                    row[:, 0][row[:, 1:].sum(-1) < 1e-8] = 1
                    if step < 50:
                        dist = Categorical(row)
                        item = dist.sample()  # row.argmax(dim=-1)  #
                        selected = item[:, None]
                    else:
                        selected = row.max(-1)[1][:, None]  # row.argmax(dim=-1)
                    demand_count += self.env.raw_depot_node_demand.expand(self.trainer_params['validation_aug_factor'], -1).gather(1, selected).squeeze()
                    remaining_demand = total_demand - demand_count
                    visited = visited.scatter(-1, selected, 1)
                    visited[:, 0] = 0
                    capacity -= self.env.raw_depot_node_demand.expand((self.trainer_params['validation_aug_factor'], -1)).gather(-1, selected)
                    capacity[selected == 0] = 1
                    vehicle_count[selected.squeeze() == 0] += 1
                    if step > 1:
                        solution_flag = solution_flag.scatter_add(dim=-1, index=node_count, src=(selected == 0).long())
                    node_count[selected != 0] += 1
                    solution = solution.scatter_add(dim=-1, index=node_count, src=selected)
                    if (visited.sum(-1) == self.env.raw_problems.size(1) - 1).all():
                        break
            solution_flag[:, -1] = 1
            solution_list.append(solution)
            solution_flag_list.append(solution_flag)
        return solution_list, solution_flag_list

    def _test_one_batch(self, solution_gnn, solution_gnn_flag, data, demand, batch_size, episode, k):
        aug_factor = self.trainer_params['validation_aug_factor']
        solution = solution_gnn.clone()[episode:episode + batch_size]
        solution_flag = solution_gnn_flag.clone()[episode:episode + batch_size]
        now_problem = torch.cat((data.clone(), demand[:, :, None].clone()), dim=-1)[episode:episode + batch_size]
        now_data = data.clone()[episode:episode + batch_size]
        now_demand = demand.clone()[episode:episode + batch_size]
        if k == 1 or k == 2:
            roll = self.env.problem_size // 2
        else:
            roll = random.randint(1, self.env.problem_size)
        solving_length = ((solution.size(-1)) // self.env.problem_size) * self.env.problem_size
        padding_aim = ((solution.size(-1) - 1) // self.env.problem_size + 1) * self.env.problem_size
        solution = solution.clone().roll(dims=-1, shifts=roll)
        solution_flag = solution_flag.clone().roll(dims=-1, shifts=roll)
        solution_cut = solution[:, :, :solving_length].clone()
        solution_cut_flag = solution_flag[:, :, :solving_length].clone()
        n_tsps_per_route = solution_cut.view(solution.size(0), solution.size(1), -1, self.env.problem_size)
        n_tsps_per_route_flag = solution_cut_flag.view(solution.size(0), solution.size(1), -1, self.env.problem_size)
        demand_per_route = now_demand[:, None, None, :].repeat(1, n_tsps_per_route.size(1), n_tsps_per_route.size(2), 1).gather(-1, n_tsps_per_route)
        n_tsps_per_route = n_tsps_per_route.view(-1, n_tsps_per_route.size(-2), self.env.problem_size)
        n_tsps_per_route_flag = n_tsps_per_route_flag.view(-1, n_tsps_per_route.size(-2), self.env.problem_size)
        demand_per_route = demand_per_route.view(-1, n_tsps_per_route.size(-2), self.env.problem_size)

        real_demand = now_demand.clone()[:, None, :].repeat(1, solution.size(1), 1).gather(-1, solution)
        double_segment_sum = torch.cumsum(torch.cat((real_demand, real_demand), dim=-1).clone(), dim=-1)
        double_solution_flag = torch.cat((solution_flag, solution_flag), dim=-1).clone()
        solution_start = torch.cummax(double_solution_flag.roll(dims=-1, shifts=1) * torch.arange(solution.size(-1) * 2)[None, :], dim=-1)[0]
        before = double_segment_sum.roll(dims=-1, shifts=1) - double_segment_sum.gather(-1, (solution_start - 1).clamp_min_(0))
        before = before[:, :, solution.size(-1):]
        solution_end = 2 * solution.size(-1) - 1 - torch.flip(torch.cummax(torch.flip(double_solution_flag, dims=[-1]) * torch.arange(solution.size(-1) * 2)[None, :], dim=-1)[0],
                                                              dims=[-1])
        double_segment_sum[:, :, -1] = 0
        end = double_segment_sum.gather(-1, solution_end) - double_segment_sum.roll(dims=-1, shifts=1)
        end = end[:, :, :solution.size(-1)]
        if padding_aim > solution.size(-1):
            before_padding = before.roll(dims=-1, shifts=-1).clone()
            before_padding = torch.cat((before_padding, before_padding[:, :, -1].unsqueeze(-1).repeat(1, 1, padding_aim - solution.size(-1))), dim=-1)
            end_padding = torch.cat((end.clone(), end[:, :, -1].unsqueeze(-1).repeat(1, 1, padding_aim - solution.size(-1))), dim=-1)
        else:
            before_padding = before.roll(dims=-1, shifts=-1).clone()
            end_padding = end.clone()
        capacity_end = 1 - before_padding.reshape(solution.size(0) * solution.size(1), -1, self.env.problem_size)[:, :, -1]
        capacity_now = 1 - end_padding.reshape(solution.size(0) * solution.size(1), -1, self.env.problem_size)[:, :, 0]

        n_tsps_per_route = n_tsps_per_route.view(solution.size(0), solution.size(1), -1, self.env.problem_size).view(solution.size(0), -1, self.env.problem_size)
        n_tsps_per_route_flag = n_tsps_per_route_flag.view(solution.size(0), solution.size(1), -1, self.env.problem_size).view(solution.size(0), -1, self.env.problem_size)
        tsp_insts = now_problem[:, None, :].repeat(1, n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 3))
        add_depot_insts_now = torch.cat((now_problem[:, 0, :][:, None, None, :].repeat(1, tsp_insts.size(1), 1, 1), tsp_insts), dim=2)
        tsp_insts_now = add_depot_insts_now.view(-1, add_depot_insts_now.size(-2), add_depot_insts_now.size(-1))
        solution_now = torch.arange(1, tsp_insts_now.size(-2))[None, :].expand((tsp_insts_now.size(0), -1))[:, None, :]
        reward_now = self.env.cal_open_length(tsp_insts_now[:, :, [0, 1]], solution_now, n_tsps_per_route_flag.view(-1, tsp_insts_now.size(-2) - 1)[:, None, :])
        capacity_pair2 = capacity_end.clone().view(-1, 1)
        capacity_pair1 = capacity_now.clone().roll(dims=1, shifts=-1).view(-1, 1)
        capacity_pair = torch.cat((capacity_pair1, capacity_pair2), dim=-1)
        capacity_pair[:, 0][(capacity_pair[:, 1] == 1.)] = 0.
        tag = ((capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] > 0.5)).clone()
        capacity_pair[:, 1][tag] = 0.5
        capacity_pair[:, 0][tag] = 0.5
        capacity_pair[:, 0][(capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)] = 1 - capacity_pair[:, 1][(capacity_pair[:, 0] > 0.5) & (capacity_pair[:, 1] <= 0.5)]
        capacity_pair[:, 1][(capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)] = 1 - capacity_pair[:, 0][(capacity_pair[:, 1] > 0.5) & (capacity_pair[:, 0] <= 0.5)]
        if padding_aim > solution.size(-1):
            capacity_head = capacity_pair[:, 1].clone().view(batch_size * aug_factor, -1).roll(dims=1, shifts=1)[:, :-1].reshape(-1, 1)
            capacity_tail = capacity_pair[:, 0].clone().view(batch_size * aug_factor, -1)[:, :-1].reshape(-1, 1)
        else:
            capacity_head = capacity_pair[:, 1].clone().view(batch_size * aug_factor, -1).roll(dims=1, shifts=1).view(-1, 1)
            capacity_tail = capacity_pair[:, 0].clone().view(-1, 1)
        new_batch_size = tsp_insts_now.size(0)
        tsp_insts_now_norm = self.coordinate_transformation(tsp_insts_now)

        '''
        if k == 1 or k == 2:
            plt.rcParams['pdf.use14corefonts'] = True
            plt.axis('off')
            id = solution[0, 0].clone()
            flag = solution_flag[0, 0].clone()
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
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors,linewidth = 1)
                if id[i] == 0:
                    color += 1
                    my_colors = colorboard[color]
            place = now_problem.cpu().numpy()[0, id[0], :]
            place2 = now_problem.cpu().numpy()[0, id[- 1], :]
            plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors,linewidth = 1.6)
            for i in range(1, id.shape[0]):
                place = now_problem.cpu().numpy()[0, id[i], :]
                place2 = now_problem.cpu().numpy()[0, id[i - 1], :]
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors,linewidth = 1.6)
                if id[i] == 0:
                    break
            tsp_show = tsp_insts_now.clone().cpu().numpy()
            for i in range(10):
               plt.scatter(tsp_show[i, :, 0], tsp_show[i, :, 1],s = 10)
            plt.show()

        '''
        with torch.no_grad():
            self.env.load_problems(new_batch_size, tsp_insts_now_norm[:, 0:1, :2], tsp_insts_now_norm[:, 1:, :2], tsp_insts_now_norm[:, 1:, -1],
                                   n_tsps_per_route_flag[:, :, -1].clone().view(-1))
            reset_state, _, _ = self.env.reset(capacity_head, capacity_tail)
            self.model_t.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            while not done:
                cur_dist = self.env.get_local_feature()
                selected, prob = self.model_t(state, cur_dist, n_tsps_per_route_flag[:, :, -1].clone().view(-1))
                state, reward, done = self.env.step(selected)
        new_solution = torch.cat((state.solution_list.unsqueeze(-1), state.solution_flag.unsqueeze(-1)), dim=-1)
        reward = self.env.cal_length(tsp_insts_now[:, :, [0, 1]], new_solution[:, :, :, 0], new_solution[:, :, :, 1])
        tag = reward.view(batch_size, aug_factor, -1, self.env.pomo_size).min(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
        tag_solution = state.solution_list.view(batch_size, aug_factor, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze(3)
        tag_solution_flag = state.solution_flag.view(batch_size, aug_factor, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze(3)
        r = (reward.min(1)[0] > reward_now.squeeze()).view(solution.size(0), aug_factor, -1, 1).expand((-1, -1, -1, tsp_insts_now.size(-2) - 1))
        tag_solution[r] = solution_now.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2) - 1)[r]
        tag_solution_flag[r] = n_tsps_per_route_flag.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2) - 1)[r]
        merge_solution = n_tsps_per_route.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2) - 1).gather(-1, tag_solution - 1).view(solution.size(0), solution.size(1), -1)
        merge_solution_flag = tag_solution_flag.view(solution.size(0), solution.size(1), -1)
        solution = torch.cat((merge_solution.clone(), solution[:, :, solving_length:]), dim=-1)
        solution_flag = torch.cat((merge_solution_flag.clone(), solution_flag[:, :, solving_length:]), dim=-1)
        # self.env.cal_leagal(now_problem[:, :, -1], solution[0], solution_flag[0])
        solution_out = solution_gnn.clone()
        solution_out_flag = solution_gnn_flag.clone()
        solution_out[episode:episode + batch_size] = solution
        solution_out_flag[episode:episode + batch_size] = solution_flag
        merge_reward_0 = self.env.cal_length_total2(now_problem[:, :, [0, 1]], solution, solution_flag)
        # tsp_show = tsp_insts_now.clone().cpu().numpy()
        # for i in range(tag_solution.size(-2)):
        #    plt.scatter(tsp_show[i, :, 0], tsp_show[i, :, 1])
        '''
        best = merge_reward_0.min(-1)[1].squeeze().item()  # 15.2% 6.0% 5.1%
        if k == 250:
            plt.rcParams['pdf.use14corefonts'] = True
            plt.axis('off')
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
                plt.scatter(place2[0], place2[1], color='grey', s=10)
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
                if id[i] == 0:
                    color += 1
                    my_colors = colorboard[color]
            place = now_problem.cpu().numpy()[0, id[0], :]
            place2 = now_problem.cpu().numpy()[0, id[- 1], :]
            plt.scatter(place2[0], place2[1], color='grey', s=10)
            plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
            for i in range(1, id.shape[0]):
                place = now_problem.cpu().numpy()[0, id[i], :]
                place2 = now_problem.cpu().numpy()[0, id[i - 1], :]
                plt.scatter(place2[0], place2[1], color='grey', s=10)
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
                if id[i] == 0:
                    break
            plt.show()
        if k == 1:
            plt.rcParams['pdf.use14corefonts'] = True
            plt.axis('off')
            f = open('solution.txt', encoding='gbk')
            solution = []
            solution_flag = []
            for line in f:
                lines = line.split("\t")
                for i in lines:
                    if i != '' and i != '\n' :
                        solution.append(float(i))
                        solution_flag.append(0)
                solution_flag[-1] = 1
            id = torch.tensor(solution).clone().long()
            flag = torch.tensor(solution_flag).clone().long()
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
                plt.scatter(place2[0], place2[1], color='grey', s=10)
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
                if id[i] == 0:
                    color += 1
                    my_colors = colorboard[color]
            place = now_problem.cpu().numpy()[0, id[0], :]
            place2 = now_problem.cpu().numpy()[0, id[- 1], :]
            plt.scatter(place2[0], place2[1], color='grey', s=10)
            plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
            for i in range(1, id.shape[0]):
                place = now_problem.cpu().numpy()[0, id[i], :]
                place2 = now_problem.cpu().numpy()[0, id[i - 1], :]
                plt.scatter(place2[0], place2[1], color='grey', s=10)
                plt.plot([place[0], place2[0]], [place[1], place2[1]], color=my_colors, linewidth=1)
                if id[i] == 0:
                    break
            plt.show()
        '''
        return solution_out, solution_out_flag, merge_reward_0[:, 0].mean(0).item(), merge_reward_0.min(1)[0].mean(0).item()  # merge_reward_1.min(1)[0].mean(0).item()
