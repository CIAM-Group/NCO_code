import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import os
import networkx as nx
import numpy as np
from logging import getLogger
from torch_geometric.data import Data
from MISEnv import MISEnv as Env
from PartitionModel import PartitionModel as PartitionModel
from MISModel import MISModel as Model
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from mis_dataset import MISDataset
import random
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

from MISProblemDef import get_random_problems, augment_xy_data_by_8_fold


class MISTesterrrc:
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
        self.model_p = PartitionModel(self.model_p_params['embedding_dim'], 1, 100, 1, depth=self.model_p_params['depth']).cuda()
        self.model_t = Model(self.model_params['embedding_dim'], 1, 100, 1, depth=self.model_params['depth']).cuda()
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
        # self.ER_test_dataset = MISDataset(self.trainer_params['data_load_500'])
        self.ER_large_test_dataset = MISDataset(self.trainer_params['data_load_1000'])
        length = torch.arange(self.trainer_params['validation_test_episodes'])
        self.data_1000 = []
        for i in range(self.trainer_params['validation_test_episodes']):
            self.data_1000.append(self.ER_large_test_dataset.get_example(idx=i))
        self.logger.info('=================================================================')

        self.validation(1000, self.data_1000)
        # LR Decay
        self.scheduler_p.step()
        self.scheduler_t.step()
        # Train

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

    def gen_sub_pyg_data(self, coors, feasi, device):
        feasi_t = torch.ones((feasi.size(0), coors.size(1)), dtype=torch.float32, device=device) * feasi.clone()
        x = feasi_t.view(-1, 1)
        edge_index = coors.nonzero()
        edge_index[:, 1:] += edge_index[:, 0:1] * coors.size(1)
        edge_attr = torch.ones((edge_index.size(0), 1), dtype=torch.float32, device=device)
        pyg_data = Data(x=x, edge_index=edge_index[:, 1:].T, edge_attr=edge_attr)
        return pyg_data

    def gen_pyg_data(self, coors, device):
        x = torch.ones((coors.number_of_nodes(), 1), dtype=torch.float32, device=device)
        adj = nx.to_scipy_sparse_matrix(coors).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long).cuda()
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long).cuda()
        edge_index = torch.stack([row, col], dim=0).cuda()
        edge_attr = torch.ones((coors.number_of_edges() * 2, 1), dtype=torch.float32, device=device)
        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return pyg_data

    def validation(self, scale, data):
        a = self.env.pomo_size
        self.env.pomo_size = 1
        self.time_estimator.reset()
        self.model_t.eval()
        # self.model_p.eval()
        solution_list = self._load_init_sol(data)
        solution_flag = torch.stack(solution_list, dim=0)

        test_num_episode = self.trainer_params['validation_test_episodes']
        k = 0
        while k < 250:
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(self.trainer_params['validation_test_batch_size'], remaining)
                solution_flag, score, aug_score = self._test_one_batch(solution_flag, data, episode)
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("iter {:2d}, episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(k,
                                                                                                                               episode, test_num_episode, elapsed_time_str,
                                                                                                                               remain_time_str, score_AM.avg, aug_score_AM.avg))

        self.logger.info(" *** Validation " + str(scale) + " Done *** ")
        self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
        self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

        cost_file = open(self.result_folder + '/curve' + str(scale) + '.txt', mode='a+')
        cost_file.write(str(score_AM.avg) + ' ' + str(aug_score_AM.avg) + '\n')
        self.env.pomo_size = a

    def _load_init_sol(self, data):
        solution_list = []
        for i in range(self.trainer_params['validation_test_episodes']):
            self.env.load_raw_problems(1, i, data)
            pyg_data = self.gen_pyg_data(self.env.raw_problems, self.env.device)
            heatmap = self.model_p(pyg_data)
            heatmap = heatmap / (heatmap.min() + 1e-5)
            index = torch.topk(heatmap, k=self.trainer_params['validation_aug_factor'])[1]
            visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.number_of_nodes() + 1)
            fesibility_mask = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.number_of_nodes() + 1)
            selected = index[:, None]
            visited = visited.scatter(-1, selected[:, 0:1], 1)
            fesibility_mask = fesibility_mask.scatter(-1, selected[:, 0:1], 1)
            adj_mat = torch.from_numpy(nx.to_numpy_array(self.env.raw_problems)).cuda()[None, :].expand(self.trainer_params['validation_aug_factor'], -1, -1)
            adj_mat_withdummy = torch.cat((adj_mat.clone(), torch.zeros_like(adj_mat[:, 0:1, :])), dim=1)
            fesibility_step = adj_mat_withdummy.gather(1, selected[:, :, None].expand(-1, -1, self.env.raw_problems.number_of_nodes())).squeeze().clone()
            fesibility_mask[:, :-1][fesibility_step == 1] = 1
            heatmap_prob = torch.cat((heatmap.clone(), torch.zeros_like(heatmap[0:1])), dim=-1)[None, :].repeat(self.trainer_params['validation_aug_factor'], 1)
            while not (selected == self.env.raw_problems.number_of_nodes()).all():
                row = heatmap_prob * (1 - fesibility_mask).clone()
                row[:, -1][row[:, :-1].sum(-1) < 1e-8] = 1.
                selected = row.max(-1)[1][:, None]
                visited = visited.scatter(-1, selected, 1)
                fesibility_mask = fesibility_mask.scatter(-1, selected, 1)
                fesibility_step = adj_mat_withdummy.gather(1, selected[:, :, None].expand(-1, -1, self.env.raw_problems.number_of_nodes())).squeeze().clone()
                fesibility_mask[:, :-1][fesibility_step == 1] = 1
                visited[:, -1] = 0
            solution_flag = visited[:, :-1].clone()
            solution_list.append(solution_flag)
        return solution_list

    def _test_one_batch(self, solution_flag_gnn, data, episode):
        aug_factor = self.trainer_params['validation_aug_factor']
        solution_flag = solution_flag_gnn.clone()[episode]
        now_problem = data[episode]
        adj_mat = torch.from_numpy(nx.to_numpy_array(now_problem)).cuda()[None, :].expand(aug_factor, -1, -1)
        solution = torch.randperm(now_problem.number_of_nodes())[None, :].repeat(solution_flag.size(0), 1)[:, :self.env.problem_size]
        solution_flag_per = solution_flag.gather(1, solution)
        n_tsps_per_route = solution.clone()
        sub_graphs = adj_mat.clone().gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, now_problem.number_of_nodes())). \
            gather(-1, n_tsps_per_route.unsqueeze(-2).expand(-1, self.env.problem_size, -1))
        solution_flag_else = solution_flag.clone().scatter(1, solution, 0)
        fesi_now = (adj_mat.clone().gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, now_problem.number_of_nodes())) * solution_flag_else.unsqueeze(1)).sum(-1) == 0
        sub_graphs_now = sub_graphs.view(-1, sub_graphs.size(-2), sub_graphs.size(-1))
        solution_flag_now = solution_flag_per.clone()
        reward_now = solution_flag_now.sum(-1).clone()
        new_batch_size = sub_graphs_now.size(0)
        pyg_data = self.gen_sub_pyg_data(sub_graphs, fesi_now, self.env.device)
        heatmap_bg = self.model_t(pyg_data)
        heatmap_bg = heatmap_bg / (heatmap_bg.min() + 1e-5)
        heatmap_t = heatmap_bg.view(new_batch_size, -1)
        visited_solver = torch.zeros((aug_factor, self.env.pomo_size, self.env.problem_size + 1), dtype=torch.long)
        fesibility_solver = torch.cat(((1 - fesi_now.clone().long()), torch.zeros_like(fesi_now[:, 0:1])), dim=-1)[:, None, :].repeat(1, self.env.pomo_size, 1)
        sub_graphs_ = sub_graphs.clone()[:, None, :].expand(-1, self.env.pomo_size, -1, -1)
        sub_graphs_withdummy = torch.cat((sub_graphs_.clone(), torch.zeros_like(sub_graphs_[:, :, 0:1, :])), dim=-2)
        selected_solver = torch.zeros((aug_factor, self.env.pomo_size), dtype=torch.long)
        while not (selected_solver == self.env.problem_size).all():
            row = torch.cat((heatmap_t.clone(), torch.zeros_like(heatmap_t[:, 0:1])), dim=-1)[:, None].repeat(1, self.env.pomo_size, 1) * (1 - fesibility_solver).clone()
            row[:, :, -1][row[:, :, :-1].sum(-1) < 1e-8] = 1.
            selected_solver = row.max(-1)[1][:, :, None]  # row.gather(2, selected).log().squeeze()
            visited_solver = visited_solver.scatter(-1, selected_solver, 1)
            fesibility_solver = fesibility_solver.scatter(-1, selected_solver, 1)
            fesibility_step_solver = sub_graphs_withdummy.gather(-2, selected_solver[:, :, :, None].expand(-1, -1, -1, self.env.problem_size)).squeeze(-2).clone()
            fesibility_solver[:, :, :-1][fesibility_step_solver == 1] = 1
            visited_solver[:, :, -1] = 0
        solution_flag_out = visited_solver[:, :, :-1].clone()
        reward = solution_flag_out.sum(-1).clone()
        # Loss
        ###############################################
        tag = reward.view(1, aug_factor, self.env.pomo_size).max(-1)[1][..., None, None].expand(-1, -1, -1, self.env.problem_size)
        tag_solution = solution_flag_out.view(1, aug_factor, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()
        r = (reward.max(1)[0] < reward_now).view(aug_factor, 1).expand((-1, sub_graphs_now.size(-2)))
        tag_solution[r] = solution_flag_now.clone().view(aug_factor, sub_graphs_now.size(-2))[r]
        solution_flag = solution_flag.clone().scatter(1, solution, tag_solution)
        # for
        assert ((solution_flag[:, :, None] & solution_flag[:, None, :]).long() * adj_mat).sum(-1).sum(-1).sum(-1) == 0
        merge_reward = solution_flag.sum(-1)
        solution_out = solution_flag_gnn.clone()
        solution_out[episode] = solution_flag.clone()
        return solution_out, merge_reward[0].item(), merge_reward.max(0)[0].item()  # merge_reward_1.min(1)[0].mean(0).item()
