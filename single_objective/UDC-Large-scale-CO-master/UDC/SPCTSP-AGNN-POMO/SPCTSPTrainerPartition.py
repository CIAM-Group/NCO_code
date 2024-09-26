import torch
from logging import getLogger
from torch_geometric.data import Data
from SPCTSPEnv import SPCTSPEnv as Env
from PartitionModel import PartitionModel as PartitionModel
from SPCTSPModel import SPCTSPModel as Model
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

from SPCTSProblemDef import get_random_problems, augment_xy_data_by_8_fold


class SPCTSPTrainerPartition:
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
        self.node_coords_500 = self.env.make_dataset_pickle(self.trainer_params['data_load_500'], self.trainer_params['validation_test_episodes'])
        self.node_coords_1000 = self.env.make_dataset_pickle(self.trainer_params['data_load_1000'], self.trainer_params['validation_test_episodes'])
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler_p.step()
            self.scheduler_t.step()
            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.validation(500, self.node_coords_500)
            self.validation(1000, self.node_coords_1000)

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
        pyg_data = self.gen_pyg_data(self.env.raw_problems)
        index = torch.zeros(self.env.sample_size, dtype=torch.long)
        logp = torch.zeros(self.env.sample_size, dtype=torch.float32)
        collected_prize = torch.zeros_like(logp)
        visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1))
        depot_mask = torch.zeros_like(visited)
        return_mask = torch.zeros_like(visited)
        depot_mask[:, 0] = 1  # unmask the depot when 1) enough prize collected; 2) all nodes visited
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
            row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone() * (1 - depot_mask).clone() * (
                    1 - return_mask).clone()
            dist = Categorical(row)
            item = dist.sample()  # row.argmax(dim=-1)  #
            log_prob = dist.log_prob(item)
            selected = item[:, None]  # row.reshape(batch_size * row.size(1), -1).multinomial(1).squeeze(dim=1).reshape(batch_size, row.size(1))[:, :, None]
            logp += log_prob  # row.gather(2, selected).log().squeeze()
            return_mask[(selected == 0).expand_as(visited)] = 1
            return_mask[:, 0] = 0
            visited = visited.scatter(-1, selected, 1)
            collected_prize += self.env.raw_problems[:, :, 2].expand(self.env.sample_size, -1).gather(1, selected).squeeze()
            depot_mask[:, 0][collected_prize + 1e-8 > 1] = 0
            visited[:, 0] = 0
            solution = torch.cat((solution, selected), dim=-1)
        for i in range(2):
            # roll = ((solution == 0).long() * torch.ones_like(solution)).max(-1)[1] % solution.size(-1)
            # roll_diff = (torch.arange(solution.size(-1))[None, :].expand_as(solution) + roll[:, None]) % solution.size(-1)
            # solution = solution.gather(1, roll_diff)
            solution_flag = torch.zeros_like(solution).scatter(1, solution, 1)
            solution_all = torch.randperm(solution.size(-1))[None, :].repeat(solution.size(0), 1)
            solution_flag = solution_flag.gather(1, solution_all)
            tsp_solution_all = solution.gather(1, (torch.cumsum(solution_flag.clone(), dim=-1) - 1).clamp_min_(0))
            solution_all[solution_flag == 1] = tsp_solution_all[solution_flag == 1]
            roll = self.env.problem_size // 2
            solution_all = solution_all.roll(dims=1, shifts=roll)
            solution_flag = solution_flag.roll(dims=1, shifts=roll)
            depot_flag = (solution_all.clone() == 0).long()
            n_tsps_per_route = solution_all.view(solution.size(0), -1, self.env.problem_size)
            n_tsps_flag_per_route = solution_flag.view(solution.size(0), -1, self.env.problem_size)
            n_depot_flag_per_route = depot_flag.view(solution.size(0), -1, self.env.problem_size)
            tsp_insts = self.env.raw_problems[:, None, :].repeat(solution.size(0), n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(
                -1, -1, -1, 5))
            tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
            solution_flag_now = n_tsps_flag_per_route.view(-1, n_tsps_flag_per_route.size(-1))[:, None, :].squeeze(1)
            depot_flag_now = n_depot_flag_per_route.view(-1, n_depot_flag_per_route.size(-1))[:, None, :].squeeze(1)

            last = (solution_flag_now * torch.arange(solution_flag_now.size(-1))).max(-1)[0]
            solution_flag_long = torch.cat((solution_flag_now, torch.ones_like(solution_flag_now[:, -1][:, None])), dim=-1)
            solution_now = n_tsps_per_route.size(-1) - torch.flip(
                torch.cummax(torch.flip(solution_flag_long, dims=[-1]) * torch.arange(n_tsps_per_route.size(-1) + 1)[None, :], dim=-1)[0], dims=[-1])[:, :n_tsps_per_route.size(-1)]
            solution_now[solution_now == n_tsps_per_route.size(-1)] = last.clone()[:, None].repeat(1, n_tsps_per_route.size(-1))[solution_now == n_tsps_per_route.size(-1)]
            reward_now = self.env.get_open_travel_distance(tsp_insts_now, solution_now)

            new_batch_size = tsp_insts_now.size(0)
            collected_prize = (solution_flag_now * tsp_insts_now[:, :, 2]).sum(-1)
            first = (solution_flag_now.clone() * torch.arange(solution_flag_now.size(-1)))
            first[solution_flag_now == 0] = solution_flag_now.size(-1)
            first = first.min(-1)[0]
            self.env.load_problems(new_batch_size, tsp_insts_now[:, :, [0, 1, 3, 4]], tsp_insts_now[:, :, 2], first, last, collected_prize, depot_flag_now)
            reset_state, _, _ = self.env.reset()
            self.model_t.pre_forward(reset_state)
            prob_list = torch.zeros(size=(new_batch_size, self.env.pomo_size, 0))
            # shape: (batch, pomo, 0~problem)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not (done and self.env.selected_count == self.env.problem_size):
                cur_dist = self.env.get_local_feature()
                selected, prob = self.model_t(state, cur_dist)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            # Loss
            ###############################################
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            loss = (-advantage * prob_list.log().sum(dim=2)).mean()
            self.model_t.zero_grad()
            loss.backward()
            self.optimizer_t.step()
            reward = -1 * reward.clone()
            # Loss
            ###############################################
            tag = reward.view(batch_size, self.env.sample_size, -1, self.env.pomo_size).min(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
            tag_solution = self.env.selected_node_list.view(batch_size, self.env.sample_size, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze()
            reversed_tag_solution = torch.flip(tag_solution, dims=[2])
            tag_solution[tag.squeeze() >= self.env.pomo_size / 2] = reversed_tag_solution[tag.squeeze() >= self.env.pomo_size / 2]
            r = (reward.min(1)[0] > reward_now.squeeze()).view(self.env.sample_size, -1, 1).expand((-1, -1, tsp_insts_now.size(-2)))
            tag_solution[r] = solution_now.view(self.env.sample_size, -1, tsp_insts_now.size(-2))[r]
            merge_solution = n_tsps_per_route.gather(-1, tag_solution).view(solution.size(0), -1)
            solution = merge_solution.clone()
            solution_pos = (solution != solution.roll(dims=-1, shifts=-1)).long()
            gather_idx = solution_pos * torch.arange(solution.size(-1))
            gather_idx[solution_pos == 0] = solution.size(-1)
            gather_idx = gather_idx.sort(-1)[0]
            solution = solution.clone().gather(-1, gather_idx % solution.size(-1))

        merge_reward = -1 * self.env._get_travel_distance(self.env.raw_problems, solution).detach()
        advantage2 = merge_reward - merge_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        loss_partition = (-advantage2 * logp).mean()  # + (-advantage2 * logp_a).mean()

        '''
        tsp_show = tsp_insts_now.clone().cpu().numpy()
        for i in range(10):
            plt.scatter(tsp_show[i, :, 0], tsp_show[i, :, 1])
        id = merge_solution[0].clone().cpu().numpy()
        for i in range(1, merge_solution.size(-1)):
            place = self.env.raw_problems.cpu().numpy()[0, id[i], :]
            place2 = self.env.raw_problems.cpu().numpy()[0, id[i - 1], :]
            plt.plot([place[0], place2[0]], [place[1], place2[1]], color='r', linewidth=0.5)
        place = self.env.raw_problems.cpu().numpy()[0, id[0], :]
        place2 = self.env.raw_problems.cpu().numpy()[0, id[- 1], :]
        plt.plot([place[0], place2[0]], [place[1], place2[1]], color='r', linewidth=0.5)
        plt.show()
        '''

        # Score
        ###############################################
        max_pomo_reward, _ = merge_reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model_p.zero_grad()
        loss_partition.backward()
        self.optimizer_p.step()
        return score_mean.item(), loss.item(), loss_partition.item()

    def gen_distance_matrix(self, coordinates):
        distances = torch.cdist(coordinates, coordinates, p=2)
        return distances

    def gen_pyg_data(self, coors, k_sparse=100):
        bs = coors.size(0)
        n_nodes = coors.size(1)
        cos_mat = -1 * self.gen_distance_matrix(coors[:, :, :2])
        x = coors[:, :, 3:]
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

    def validation(self, scale, node_coords):
        a = self.env.pomo_size
        self.env.pomo_size = 2
        self.time_estimator.reset()
        self.model_t.eval()
        # self.model_p.eval()
        self.model_t.model_params['eval_type'] = 'argmax'
        solution_list = self._load_init_sol(node_coords)
        solution = torch.stack(solution_list, dim=0)
        test_num_episode = self.trainer_params['validation_test_episodes']
        k = 0
        while k < 2:
            # roll = ((solution == 0).long() * torch.ones_like(solution)).max(-1)[1] % solution.size(-1)
            # roll_diff = (torch.arange(solution.size(-1))[None, None, :].expand_as(solution) + roll[:, :, None]) % solution.size(-1)
            # solution = solution.gather(-1, roll_diff)
            k += 1
            episode = 0
            score_AM = AverageMeter()
            aug_score_AM = AverageMeter()
            while episode < test_num_episode:
                remaining = test_num_episode - episode
                batch_size = min(self.trainer_params['validation_test_batch_size'], remaining)
                solution, score, aug_score = self._test_one_batch(solution, node_coords, batch_size, episode, k)
                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)
                episode += batch_size

        self.logger.info(" *** Validation Scale " + str(scale) + " Done *** ")
        self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
        self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

        cost_file = open(self.result_folder + '/curve' + str(scale) + '.txt', mode='a+')
        cost_file.write(str(score_AM.avg) + ' ' + str(aug_score_AM.avg) + '\n')
        self.model_t.model_params['eval_type'] = 'softmax'
        self.env.pomo_size = a

    def _load_init_sol(self, data):
        solution_list = []
        for i in range(self.trainer_params['validation_test_episodes']):
            self.env.load_raw_problems(1, i, data)
            pyg_data = self.gen_pyg_data(self.env.raw_problems)
            index = torch.zeros(self.trainer_params['validation_aug_factor'], dtype=torch.long)
            collected_prize = torch.zeros(self.trainer_params['validation_aug_factor'], dtype=torch.float32)
            visited = torch.zeros_like(index)[:, None].repeat(1, self.env.raw_problems.size(1))
            depot_mask = torch.zeros_like(visited)
            depot_mask[:, 0] = 1
            return_mask = torch.zeros_like(visited)
            solution = index[:, None]
            visited = visited.scatter(-1, solution[:, 0:1], 1)
            selected = solution
            heatmap = None
            step = 0
            self.model_p.pre(pyg_data)
            while solution.size(-1) < self.env.raw_problems.size(1):
                step += 1
                if (solution.size(-1) - 1) % self.env.problem_size == 0:
                    node_emb, heatmap = self.model_p(solution, visited)
                    heatmap = heatmap / (heatmap.min() + 1e-5)
                    heatmap = self.model_p.reshape(pyg_data, heatmap) + 1e-5
                row = heatmap.gather(1, selected[:, None, :].expand(-1, -1, heatmap.size(-1))).clone().squeeze(1) * (1 - visited).clone() * (1 - depot_mask).clone() * (
                        1 - return_mask).clone()
                if step < 3:
                    dist = Categorical(row)
                    item = dist.sample()  # row.argmax(dim=-1)  #
                    selected = item[:, None]
                else:
                    selected = row.max(-1)[1][:, None]  # row.argmax(dim=-1)
                return_mask[(selected == 0).expand_as(visited)] = 1
                return_mask[:, 0] = 0
                visited = visited.scatter(-1, selected, 1)
                collected_prize += self.env.raw_problems[:, :, 2].expand(self.trainer_params['validation_aug_factor'], -1).gather(1, selected).squeeze()
                depot_mask[:, 0][collected_prize + 1e-8 > 1] = 0
                visited[:, 0] = 0
                solution = torch.cat((solution, selected), dim=-1)
            solution_list.append(solution)
        return solution_list

    def _test_one_batch(self, solution_gnn, data, batch_size, episode, k):
        aug_factor = self.trainer_params['validation_aug_factor']
        solution = solution_gnn.clone()[episode:episode + batch_size]
        now_problem = data.clone()[episode:episode + batch_size]
        solution_flag = torch.zeros_like(solution).scatter(-1, solution, 1)
        solution_all = torch.randperm(solution.size(-1))[None, None, :].repeat(solution.size(0), solution.size(1), 1)
        solution_flag = solution_flag.gather(-1, solution_all)
        tsp_solution_all = solution.gather(-1, (torch.cumsum(solution_flag.clone(), dim=-1) - 1).clamp_min_(0))
        solution_all[solution_flag == 1] = tsp_solution_all[solution_flag == 1]
        if k == 1 or k == 2:
            roll = self.env.problem_size // 2
        else:
            roll = random.randint(1, self.env.problem_size)
        solution_all = solution_all.roll(dims=-1, shifts=roll)
        solution_flag = solution_flag.roll(dims=-1, shifts=roll)
        solving_length = ((solution.size(-1)) // self.env.problem_size) * self.env.problem_size
        solution_cut = solution_all[:, :, :solving_length].clone()
        solution_flag_cut = solution_flag[:, :, :solving_length].clone()
        depot_flag = (solution_cut.clone() == 0).long()
        n_tsps_per_route = solution_cut.view(solution.size(0), -1, self.env.problem_size)
        n_tsps_flag_per_route = solution_flag_cut.view(solution.size(0), -1, self.env.problem_size)
        n_depot_flag_per_route = depot_flag.view(solution.size(0), -1, self.env.problem_size)
        tsp_insts = now_problem[:, None, :].repeat(solution.size(0), n_tsps_per_route.size(1), 1, 1).gather(-2, n_tsps_per_route.unsqueeze(-1).expand(
            -1, -1, -1, 5))
        tsp_insts_now = tsp_insts.view(-1, tsp_insts.size(-2), tsp_insts.size(-1))
        solution_flag_now = n_tsps_flag_per_route.view(-1, n_tsps_flag_per_route.size(-1))[:, None, :].squeeze(1)
        depot_flag_now = n_depot_flag_per_route.view(-1, n_depot_flag_per_route.size(-1))[:, None, :].squeeze(1)

        last = (solution_flag_now * torch.arange(solution_flag_now.size(-1))).max(-1)[0]
        solution_flag_long = torch.cat((solution_flag_now, torch.ones_like(solution_flag_now[:, -1][:, None])), dim=-1)
        solution_now = n_tsps_per_route.size(-1) - torch.flip(
            torch.cummax(torch.flip(solution_flag_long, dims=[-1]) * torch.arange(n_tsps_per_route.size(-1) + 1)[None, :], dim=-1)[0], dims=[-1])[:, :n_tsps_per_route.size(-1)]
        solution_now[solution_now == n_tsps_per_route.size(-1)] = last.clone()[:, None].repeat(1, n_tsps_per_route.size(-1))[solution_now == n_tsps_per_route.size(-1)]
        reward_now = self.env.get_open_travel_distance(tsp_insts_now, solution_now)

        new_batch_size = tsp_insts_now.size(0)
        collected_prize = (solution_flag_now * tsp_insts_now[:, :, 2]).sum(-1)
        collected_prize_norm = collected_prize.view(solution.size(0), solution.size(1), -1).sum(-1) - 1
        collected_prize = collected_prize.view(solution.size(0), solution.size(1), -1) - collected_prize_norm.clone()[:, :, None] / (
            collected_prize.view(solution.size(0), solution.size(1), -1).size(-1))
        collected_prize = collected_prize.view(-1)
        first = (solution_flag_now.clone() * torch.arange(solution_flag_now.size(-1)))
        first[solution_flag_now == 0] = solution_flag_now.size(-1)
        first = first.min(-1)[0]
        self.env.load_problems(new_batch_size, tsp_insts_now[:, :, [0, 1, 3, 4]], tsp_insts_now[:, :, 2], first, last, collected_prize, depot_flag_now)
        reset_state, _, _ = self.env.reset()
        self.model_t.pre_forward(reset_state)
        ###############################################
        state, reward, done = self.env.pre_step()
        while not (done and self.env.selected_count == self.env.problem_size):
            cur_dist = self.env.get_local_feature()
            selected, prob = self.model_t(state, cur_dist)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
        reward = -1 * reward
        # Loss
        ###############################################
        tag = reward.view(batch_size, aug_factor, -1, self.env.pomo_size).min(-1)[1][..., None, None].expand(-1, -1, -1, -1, self.env.problem_size)
        tag_solution = self.env.selected_node_list.view(batch_size, aug_factor, -1, self.env.pomo_size, self.env.problem_size).gather(-2, tag).squeeze(3)
        reversed_tag_solution = torch.flip(tag_solution.clone(), dims=[-1])
        tag_solution[tag.squeeze(3) >= self.env.pomo_size / 2] = reversed_tag_solution[tag.squeeze(3) >= self.env.pomo_size / 2]
        r = (reward.min(1)[0] > reward_now.squeeze()).view(batch_size, aug_factor, -1, 1).expand((-1, -1, -1, tsp_insts_now.size(-2)))
        tag_solution[r] = solution_now.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2))[r]
        merge_solution = n_tsps_per_route.view(batch_size, aug_factor, -1, tsp_insts_now.size(-2)).gather(-1, tag_solution).view(solution.size(0), solution.size(1), -1)
        solution = solution_all.clone()
        solution[:, :, :solving_length] = merge_solution.clone()
        solution[:, :, -1][solution_flag[:, :, -1] == 0] = solution[:, :, -2][solution_flag[:, :, -1] == 0]
        solution_pos = (solution != solution.roll(dims=-1, shifts=-1)).long()
        gather_idx = solution_pos * torch.arange(solution.size(-1))
        gather_idx[solution_pos == 0] = solution.size(-1)
        gather_idx = gather_idx.sort(-1)[0]
        solution = solution.clone().gather(-1, gather_idx % solution.size(-1))
        merge_reward_0 = self.env._get_travel_distance2(now_problem, solution)
        solution_out = solution_gnn.clone()
        solution_out[episode:episode + batch_size] = solution
        return solution_out, merge_reward_0[:, 0].mean(0).item(), merge_reward_0.min(1)[0].mean(0).item()  # merge_reward_1.min(1)[0].mean(0).item()
