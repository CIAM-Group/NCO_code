import torch
from logging import getLogger
from torch_geometric.data import Data
from KPEnv import KPEnv as Env
from PartitionModel import PartitionModel as PartitionModel
from KPModel import KPModel as Model
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

from KProblemDef import get_random_problems, augment_xy_data_by_8_fold


class KPTrainerPartition:
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
        self.data_500, self.capacity_500 = self.env.make_dataset(self.trainer_params['data_load_500'], self.trainer_params['validation_test_episodes'])
        self.data_1000, self.capacity_1000 = self.env.make_dataset(self.trainer_params['data_load_1000'], self.trainer_params['validation_test_episodes'])
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler_p.step()
            self.scheduler_t.step()
            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.validation(500, self.data_500, self.capacity_500)
            self.validation(1000, self.data_1000, self.capacity_1000)

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
        data_with_dummy = torch.cat((self.env.raw_problems, torch.zeros_like(self.env.raw_problems[:, 0:1, :])), dim=1)
        pyg_data = self.gen_pyg_data(data_with_dummy)
        index = self.env.raw_problem_size * torch.ones(self.env.sample_size, dtype=torch.long)
        logp = torch.zeros(self.env.sample_size, dtype=torch.float32)
        collected_weight = torch.zeros_like(logp)
        visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1) + 1)
        depot_mask = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1) + 1)
        depot_mask[:, -1] = 1  # unmask the depot when 1) enough prize collected; 2) all nodes visited
        solution = index[:, None]
        visited = visited.scatter(-1, solution[:, 0:1], 1)
        selected = solution
        heatmap = None
        self.model_p.pre(pyg_data)
        tag = torch.ones_like(index)
        weight_with_dummy = torch.cat((self.env.raw_problems[:, :, 0].repeat(self.env.sample_size, 1), torch.zeros_like(selected)), dim=-1)
        capacity = torch.cat((self.env.capacitys.repeat(self.env.sample_size, 1), torch.zeros_like(selected)), dim=-1)
        while solution.size(-1) < self.env.raw_problems.size(1) and ~(tag == 0).all():
            if (solution.size(-1) - 1) % self.env.problem_size == 0:
                node_emb, heatmap = self.model_p(solution, visited)
                heatmap = heatmap / (heatmap.min() + 1e-5)
                heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5
            row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone()
            row[(weight_with_dummy + collected_weight[:, None]) > capacity + 1e-8] = 0
            row[:, -1][(row[:, :-1] == 0).all(1)] = 1
            tag[(row[:, :-1] == 0).all(1)] = 0
            dist = Categorical(row)
            item = dist.sample()  # row.argmax(dim=-1)  #
            log_prob = dist.log_prob(item)
            selected = item[:, None]  # row.reshape(batch_size * row.size(1), -1).multinomial(1).squeeze(dim=1).reshape(batch_size, row.size(1))[:, :, None]
            logp += log_prob  # row.gather(2, selected).log().squeeze()
            visited = visited.scatter(-1, selected, 1)
            collected_weight += weight_with_dummy.expand(self.env.sample_size, -1).gather(1, selected).squeeze()
            visited[:, -1] = 0
            solution = torch.cat((solution, selected), dim=-1)
        solution_flag = torch.zeros_like(visited)
        solution_flag = solution_flag.scatter(1, solution, 1)[:, :-1]
        for i in range(1): #TODO: There to set Enable DCR
            solution = torch.randperm(self.env.raw_problem_size)[None, :].repeat(solution.size(0), 1)
            solution_flag_per = solution_flag.gather(1, solution)
            # tsp_solution_all = solution.gather(1, (torch.cumsum(solution_flag.clone(), dim=-1) - 1).clamp_min_(0))
            # solution_all[solution_flag == 1] = tsp_solution_all[solution_flag == 1]
            capacity_per_route = self.env.raw_problems[:, :, 0].repeat(solution.size(0), 1).gather(1, solution) * solution_flag_per
            capacity_per_route = capacity_per_route.view(solution.size(0), -1, self.env.problem_size).sum(-1)
            n_tsps_per_route = solution.view(solution.size(0), -1, self.env.problem_size)
            tsp_insts = self.env.raw_problems[:, None, :].repeat(solution.size(0), n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 2))
            tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
            capacity_now = capacity_per_route.view(-1)
            solution_flag_per = solution_flag_per.view(self.env.sample_size, -1, self.env.problem_size)
            solution_flag_now = solution_flag_per.reshape(-1, self.env.problem_size)
            reward_now = self.env.get_open_travel_distance(tsp_insts_now, solution_flag_now)
            new_batch_size = tsp_insts_now.size(0)
            self.env.load_problems(new_batch_size, tsp_insts_now, capacity_now)
            reset_state, _, _ = self.env.reset()
            self.model_t.pre_forward(reset_state)
            prob_list = torch.zeros(size=(new_batch_size, self.env.pomo_size, 0))
            # shape: (batch, pomo, 0~problem)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                cur_dist = self.env.get_local_feature()
                selected, prob = self.model_t(state, cur_dist)

                action_w_finisehd = selected.clone()
                action_w_finisehd[state.finished] = self.env.problem_size  # dummy item: 0 weight, 0 value
                state, reward, done = self.env.step(action_w_finisehd)

                chosen_prob = prob.clone()
                chosen_prob[state.finished] = 1  # done episode will gain no more probability
                prob_list = torch.cat((prob_list, chosen_prob[:, :, None]), dim=2)
            # Loss
            ###############################################
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            loss = (-advantage * prob_list.log().sum(dim=2)).mean()
            self.model_t.zero_grad()
            loss.backward()
            self.optimizer_t.step()
            selected_solution = torch.zeros((reward.size(0), self.env.pomo_size, self.env.problem_size + 1))
            selected_solution = selected_solution.scatter(-1, self.env.selected_node_list, 1)[:, :, :-1]
            # Loss
            ###############################################
            tag = reward.view(batch_size, self.env.sample_size, -1, self.env.pomo_size).max(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
            tag_solution = selected_solution.view(batch_size, self.env.sample_size, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()
            r = (reward.max(1)[0] < reward_now.squeeze()).view(self.env.sample_size, -1, 1).expand((-1, -1, tsp_insts_now.size(-2)))
            tag_solution[r] = solution_flag_now.float().view(self.env.sample_size, -1, tsp_insts_now.size(-2))[r]
            merge_solution_flag = tag_solution.view(solution.size(0), -1).long()
            solution_flag = torch.zeros_like(solution)
            solution_flag = solution_flag.scatter(-1, solution, merge_solution_flag)

        merge_reward = (self.env.raw_problems[:, :, 1].repeat(solution.size(0), 1) * solution_flag)[None, :].sum(2)
        advantage2 = merge_reward - merge_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        loss_partition = (-advantage2 * logp).mean()  # + (-advantage2 * logp_a).mean()

        # Score
        ###############################################
        max_pomo_reward, _ = merge_reward.max(dim=1)  # get best results from pomo
        score_mean = max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model_p.zero_grad()
        loss_partition.backward()
        self.optimizer_p.step()
        return score_mean.item(), loss.item(), loss_partition.item()

    def gen_distance_matrix(self, coordinates):
        problems_weights = coordinates[:, :, 0]
        problems_values = coordinates[:, :, 1]
        value_addition = problems_values[:, :, None] + problems_values[:, None, :]
        weight_addition = (problems_weights[:, :, None] + problems_weights[:, None, :]) + 1e-10
        dist_original = value_addition / weight_addition  # shape: (batch, problem, problem)
        max_dist = dist_original.max(-1)[0].unsqueeze(-1)  # shape: (batch, problem, 1)
        dist = (dist_original / max_dist) - 1  # change the range from [0, 1] to [-1, 0]
        return -1 * dist

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

    def validation(self, scale, data, capacity):
        a = self.env.pomo_size
        self.env.pomo_size = 2
        self.time_estimator.reset()
        self.model_t.eval()
        self.model_p.eval()
        self.model_t.model_params['eval_type'] = 'argmax'
        solution_list = self._load_init_sol(data, capacity)
        solution_flag = torch.stack(solution_list, dim=0)

        test_num_episode = self.trainer_params['validation_test_episodes']
        k = 0
        while k < 1:
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(self.trainer_params['validation_test_batch_size'], remaining)
                solution_flag, score, aug_score = self._test_one_batch(solution_flag, data, capacity, batch_size, episode, k)
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size

        self.logger.info(" *** Validation " + str(scale) + " Done *** ")
        self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
        self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

        cost_file = open(self.result_folder + '/curve' + str(scale) + '.txt', mode='a+')
        cost_file.write(str(score_AM.avg) + ' ' + str(aug_score_AM.avg) + '\n')
        self.model_t.model_params['eval_type'] = 'softmax'
        self.env.pomo_size = a

    def _load_init_sol(self, data, capacity):
        solution_list = []
        for i in range(self.trainer_params['validation_test_episodes']):
            self.env.load_raw_problems(1, i, data, capacity)
            data_with_dummy = torch.cat((self.env.raw_problems, torch.zeros_like(self.env.raw_problems[:, 0:1, :])), dim=1)
            pyg_data = self.gen_pyg_data(data_with_dummy)
            logp = torch.zeros(self.trainer_params['validation_aug_factor'], dtype=torch.float32)
            index = data.size(1) * torch.ones(self.trainer_params['validation_aug_factor'], dtype=torch.long)
            collected_weight = torch.zeros_like(logp)
            visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1) + 1)
            depot_mask = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1) + 1)
            depot_mask[:, -1] = 1  # unmask the depot when 1) enough prize collected; 2) all nodes visited
            solution = index[:, None]
            visited = visited.scatter(-1, solution[:, 0:1], 1)
            selected = solution
            heatmap = None
            step = 0
            self.model_p.pre(pyg_data)
            tag = torch.ones_like(index)
            weight_with_dummy = torch.cat((self.env.raw_problems[:, :, 0].repeat(self.trainer_params['validation_aug_factor'], 1), torch.zeros_like(selected)), dim=-1)
            capacity_dummy = torch.cat((self.env.capacitys.repeat(self.trainer_params['validation_aug_factor'], 1), torch.zeros_like(selected)), dim=-1)
            while solution.size(-1) < self.env.raw_problems.size(1) and ~(tag == 0).all():
                step += 1
                if (solution.size(-1) - 1) % self.env.problem_size == 0:
                    node_emb, heatmap = self.model_p(solution, visited)
                    heatmap = heatmap / (heatmap.min() + 1e-5)
                    heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5
                row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone()
                row[(weight_with_dummy + collected_weight[:, None]) > capacity_dummy + 1e-8] = 0
                row[:, -1][(row[:, :-1] == 0).all(1)] = 1
                tag[(row[:, :-1] == 0).all(1)] = 0
                if step < 5:
                    dist = Categorical(row)
                    item = dist.sample()  # row.argmax(dim=-1)  #
                    selected = item[:, None]
                else:
                    selected = row.max(-1)[1][:, None]  # row.gather(2, selected).log().squeeze()
                visited = visited.scatter(-1, selected, 1)
                collected_weight += weight_with_dummy.expand(self.trainer_params['validation_aug_factor'], -1).gather(1, selected).squeeze()
                visited[:, -1] = 0
                solution = torch.cat((solution, selected), dim=-1)
            solution_flag = torch.zeros((self.trainer_params['validation_aug_factor'], data.size(1) + 1))
            solution_flag = solution_flag.scatter(1, solution, 1)[:, :-1]
            solution_list.append(solution_flag)
        return solution_list

    def _test_one_batch(self, solution_flag_gnn, data, capacity, batch_size, episode, k):
        aug_factor = self.trainer_params['validation_aug_factor']
        solution_flag = solution_flag_gnn.clone()[episode:episode + batch_size]
        now_problem = data.clone()[episode:episode + batch_size]
        solution = torch.randperm(data.size(1))[None, None, :].repeat(batch_size, solution_flag.size(1), 1)
        solution_flag_per = solution_flag.gather(2, solution)
        capacity_per_route = now_problem[:, :, 0][:, None, :].repeat(1, solution.size(1), 1).gather(2, solution) * solution_flag_per
        capacity_per_route = capacity_per_route.view(solution.size(0), -1, self.env.problem_size).sum(-1)
        n_tsps_per_route = solution.view(solution.size(0), -1, self.env.problem_size)
        tsp_insts = now_problem[:, None, :].repeat(1, n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(-1, -1, -1, 2))
        tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
        capacity_now = capacity_per_route.view(-1)
        solution_flag_per = solution_flag_per.view(solution.size(0), -1, self.env.problem_size)
        solution_flag_now = solution_flag_per.reshape(-1, self.env.problem_size)
        reward_now = (tsp_insts_now[:, :, 1] * solution_flag_now).sum(1)
        new_batch_size = tsp_insts_now.size(0)
        self.env.load_problems(new_batch_size, tsp_insts_now, capacity_now)
        reset_state, _, _ = self.env.reset()
        self.model_t.pre_forward(reset_state)
        prob_list = torch.zeros(size=(new_batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            cur_dist = self.env.get_local_feature()
            selected, _ = self.model_t(state, cur_dist)

            action_w_finisehd = selected.clone()
            action_w_finisehd[state.finished] = self.env.problem_size  # dummy item: 0 weight, 0 value
            state, reward, done = self.env.step(action_w_finisehd)
        # Loss
        selected_solution = torch.zeros((reward.size(0), self.env.pomo_size, self.env.problem_size + 1))
        selected_solution = selected_solution.scatter(-1, self.env.selected_node_list, 1)[:, :, :-1]
        # Loss
        ###############################################
        tag = reward.view(batch_size, aug_factor, -1, self.env.pomo_size).max(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
        tag_solution = selected_solution.view(batch_size, aug_factor, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze(3)
        r = (reward.max(-1)[0] < reward_now.squeeze()).view(batch_size, aug_factor, -1, 1).expand((-1, -1, -1, tsp_insts_now.size(-2)))
        tag_solution[r] = solution_flag_now.float().view(batch_size, aug_factor, -1, tsp_insts_now.size(-2))[r]
        merge_solution_flag = tag_solution.view(solution.size(0), solution.size(1), -1).long()
        solution_flag = torch.zeros_like(solution)
        solution_flag = solution_flag.scatter(-1, solution, merge_solution_flag)

        merge_reward_0 = (now_problem[:, :, 1][:, None, :].repeat(1, aug_factor, 1) * solution_flag).sum(2)
        solution_out = solution_flag_gnn.clone()
        solution_out[episode:episode + batch_size] = solution_flag.clone()
        return solution_out, merge_reward_0[:, 0].mean(0).item(), merge_reward_0.max(1)[0].mean(0).item()  # merge_reward_1.min(1)[0].mean(0).item()
