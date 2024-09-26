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

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

from MISProblemDef import get_random_problems, augment_xy_data_by_8_fold


class MISTrainerPartition:
    def __init__(self,
                 env_params,
                 model_params,
                 model_p_params,
                 optimizer_params,
                 trainer_params):

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
        # for i in range(self.trainer_params['validation_test_episodes']):
        #    self.data_1000.append(self.ER_large_test_dataset.get_example(idx=i))
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler_p.step()
            self.scheduler_t.step()
            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            # self.validation(500, self.data_500, self.capacity_500)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict_t = {
                    'epoch': epoch,
                    'model_state_dict': self.model_t.state_dict(),
                    'optimizer_state_dict': self.optimizer_t.state_dict(),
                    'scheduler_state_dict': self.scheduler_t.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict_t, '{}/checkpoint-tsp-{}.pt'.format(self.result_folder, epoch))
                checkpoint_dict_p = {
                    'epoch': epoch,
                    'model_state_dict': self.model_p.state_dict(),
                    'optimizer_state_dict': self.optimizer_p.state_dict(),
                    'scheduler_state_dict': self.scheduler_p.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict_p, '{}/checkpoint-partition-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

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
        input /= scale_degree + 1e-10
        return input

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model_t.train()
        self.model_p.train()
        self.env.load_raw_problems(batch_size)
        pyg_data_1 = self.gen_pyg_data(self.env.raw_problems, self.env.device)
        heatmap = self.model_p(pyg_data_1)
        heatmap = heatmap / (heatmap.min() + 1e-5)
        index = torch.topk(heatmap, k=self.env.sample_size)[1]
        logp = torch.zeros(self.env.sample_size, dtype=torch.float32)
        visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problem_size + 1)
        fesibility_mask = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problem_size + 1)
        selected = index[:, None]
        visited = visited.scatter(-1, selected[:, 0:1], 1)
        fesibility_mask = fesibility_mask.scatter(-1, selected[:, 0:1], 1)
        adj_mat = torch.from_numpy(nx.to_numpy_array(self.env.raw_problems)).cuda()[None, :].expand(self.env.sample_size, -1, -1)
        adj_mat_withdummy = torch.cat((adj_mat.clone(), torch.zeros_like(adj_mat[:, 0:1, :])), dim=1)
        fesibility_step = adj_mat_withdummy.gather(1, selected[:, :, None].expand(-1, -1, self.env.raw_problem_size)).squeeze()
        fesibility_mask[:, :-1][fesibility_step == 1] = 1
        heatmap_prob = torch.cat((heatmap.clone(), torch.zeros_like(heatmap[0:1])), dim=-1)[None, :].repeat(self.env.sample_size, 1)
        while not (selected == self.env.raw_problem_size).all():
            row = heatmap_prob * (1 - fesibility_mask)
            row[:, -1][row[:, :-1].sum(-1) < 1e-8] = 1.
            dist = Categorical(row)
            item = dist.sample()  # row.argmax(dim=-1)  #
            log_prob = dist.log_prob(item)
            selected = item[:, None]  # row.reshape(batch_size * row.size(1), -1).multinomial(1).squeeze(dim=1).reshape(batch_size, row.size(1))[:, :, None]
            logp += log_prob  # row.gather(2, selected).log().squeeze()
            visited = visited.scatter(-1, selected, 1)
            fesibility_mask = fesibility_mask.scatter(-1, selected, 1)
            fesibility_step = adj_mat_withdummy.gather(1, selected[:, :, None].expand(-1, -1, self.env.raw_problem_size)).squeeze()
            fesibility_mask[:, :-1][fesibility_step == 1] = 1
            visited[:, -1] = 0
        solution_flag = visited[:, :-1].clone()
        for i in range(2):
            # sub_graph_num = (self.env.raw_problem_size // self.env.problem_size) * self.env.problem_size
            solution = torch.randperm(self.env.raw_problem_size)[None, :].repeat(solution_flag.size(0), 1)[:, :self.env.problem_size]
            solution_flag_per = solution_flag.gather(1, solution)
            n_tsps_per_route = solution.clone()
            sub_graphs = adj_mat.clone().gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, self.env.raw_problem_size)). \
                gather(-1, n_tsps_per_route.unsqueeze(-2).expand(-1, self.env.problem_size, -1))
            solution_flag_else = solution_flag.clone().scatter(1, solution, 0)
            fesi_now = (adj_mat.clone().gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, self.env.raw_problem_size)) * solution_flag_else.unsqueeze(1)).sum(-1) == 0
            sub_graphs_now = sub_graphs.view(-1, sub_graphs.size(-2), sub_graphs.size(-1))
            solution_flag_now = solution_flag_per.clone()
            reward_now = solution_flag_now.sum(-1).clone()
            new_batch_size = sub_graphs_now.size(0)
            pyg_data = self.gen_sub_pyg_data(sub_graphs, fesi_now, self.env.device)
            heatmap_bg = self.model_t(pyg_data)
            heatmap_bg = heatmap_bg / (heatmap_bg.min() + 1e-5)
            heatmap_t = heatmap_bg.view(new_batch_size, -1)
            prob_list = torch.zeros((self.env.sample_size, self.env.pomo_size), dtype=torch.float32)
            visited_solver = torch.zeros((self.env.sample_size, self.env.pomo_size, self.env.problem_size + 1), dtype=torch.long)
            fesibility_solver = torch.cat(((1 - fesi_now.clone().long()), torch.zeros_like(fesi_now[:, 0:1])), dim=-1)[:, None, :].repeat(1, self.env.pomo_size, 1)
            sub_graphs_ = sub_graphs.clone()[:, None, :].expand(-1, self.env.pomo_size, -1, -1)
            sub_graphs_withdummy = torch.cat((sub_graphs_.clone(), torch.zeros_like(sub_graphs_[:, :, 0:1, :])), dim=-2)
            selected_solver = torch.zeros((self.env.sample_size, self.env.pomo_size), dtype=torch.long)
            heatmap_prob_t = torch.cat((heatmap_t.clone(), torch.zeros_like(heatmap_t[:, 0:1])), dim=-1)[:, None].repeat(1, self.env.pomo_size, 1)
            while not (selected_solver == self.env.problem_size).all():
                # if (solution.size(-1) - 1) % self.env.problem_size == 0:
                #     heatmap = self.model_p(solution, visited_solver[:, :-1], (fesibility_solver[:, :-1] - visited_solver[:, :-1]).clone())
                #     heatmap = heatmap / (heatmap.min() + 1e-5)
                row = heatmap_prob_t * (1 - fesibility_solver)
                row[:, :, -1][row[:, :, :-1].sum(-1) < 1e-8] = 1.
                dist2 = Categorical(row)
                item2 = dist2.sample()  # row.argmax(dim=-1)  #
                prob = dist2.log_prob(item2)
                selected_solver = item2[:, :, None]  # row.reshape(batch_size * row.size(1), -1).multinomial(1).squeeze(dim=1).reshape(batch_size, row.size(1))[:, :, None]
                prob_list += prob  # row.gather(2, selected).log().squeeze()
                visited_solver = visited_solver.scatter(-1, selected_solver, 1)
                fesibility_solver = fesibility_solver.scatter(-1, selected_solver, 1)
                fesibility_step_solver = sub_graphs_withdummy.gather(-2, selected_solver[:, :, :, None].expand(-1, -1, -1, self.env.problem_size)).squeeze().clone()
                fesibility_solver[:, :, :-1][fesibility_step_solver == 1] = 1
                visited_solver[:, :, -1] = 0
            solution_flag_out = visited_solver[:, :, :-1].clone()
            reward = solution_flag_out.sum(-1).clone()
            # Loss
            ###############################################
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            loss = (-advantage * prob_list).mean()
            self.model_t.zero_grad()
            loss.backward()
            self.optimizer_t.step()
            # Loss
            ###############################################
            tag = reward.view(batch_size, self.env.sample_size, self.env.pomo_size).max(-1)[1][..., None, None].expand(-1, -1, -1, self.env.problem_size)
            tag_solution = solution_flag_out.view(batch_size, self.env.sample_size, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()
            r = (reward.max(1)[0] < reward_now.squeeze()).view(self.env.sample_size, 1).expand((-1, sub_graphs_now.size(-2)))
            tag_solution[r] = solution_flag_now.clone().view(self.env.sample_size, sub_graphs_now.size(-2))[r]
            solution_flag = solution_flag.clone().scatter(1, solution, tag_solution)

        merge_reward = solution_flag.sum(-1)
        advantage2 = merge_reward - merge_reward.float().mean(dim=0, keepdims=True)
        # shape: (batch, pomo)
        loss_partition = (-advantage2 * logp).mean()  # + (-advantage2 * logp_a).mean()
        # Score
        ###############################################
        max_pomo_reward, _ = merge_reward.max(dim=0)  # get best results from pomo
        score_mean = max_pomo_reward.float().mean()  # negative sign to make positive value
        # Step & Return
        ###############################################
        self.model_p.zero_grad()
        loss_partition.backward()
        self.optimizer_p.step()
        return score_mean.item(), 1, loss_partition.item()

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
