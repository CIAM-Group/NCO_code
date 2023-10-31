
import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from difflib import SequenceMatcher

class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
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
        self.model = Model(**self.model_params)
        self.env_params['data'] = None
        self.env_params['batch_size'] = self.trainer_params['train_batch_size']
        self.env = Env(**self.env_params)
        self.vali_env_list = []
        self.vali_size = None
        self.nll_loss = torch.nn.functional.nll_loss
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch_supervised(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch_supervised(self, batch_size):
        problem_size = self.env_params['problem_size']

        resorted_label, recounter, model_dist, pomo_idx = self._label_resort(batch_size)

        # Prep
        ###############################################
        self.model.train()
        reset_state, _, _ = self.env.reset()
        cos_sim = self.model.pre_forward(reset_state)

        prob_list = []
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        t = 0
        # while not done:
        selected_list = []
        while t < resorted_label.size(-1):
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done, _ = self.env.step(resorted_label[:, :, t])
            prob_list.append(prob)
            selected_list.append(selected)
            # print(t, torch.nonzero(torch.isnan(prob_list[-1])))

            t += 1
        prob_list = torch.stack(prob_list, -1)
        selected_list = torch.stack(selected_list, -1)

        reshape_prob_list = prob_list.permute(0, 2, 1, 3).contiguous() + 1e-15
        relax_mask = selected_list != resorted_label
        loss = self.nll_loss(reshape_prob_list.log(), resorted_label, reduce=False)
        loss = relax_mask.double() * loss
        loss = loss.sum(-1).min(-1)[0].mean()

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return model_dist.mean().item(), loss.item()

    def _label_resort(self, batch_size):
        problem_size = self.env_params['problem_size']
        # resort solutions
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state)
                # shape: (batch, pomo)
                state, reward, done, solutions = self.env.step(selected)

            model_dist, pomo_idx = reward.max(dim=1)
            model_dist = -model_dist
            model_better = model_dist < self.env.label_cost.cuda()

            recounter = 0
            resorted_label = []

            label_depot_idx = torch.nonzero(self.env.solutions == 0)[:, 1].view(batch_size, -1)
            solutions = solutions.gather(1, pomo_idx[:, None, None].expand(-1, 1, solutions.size(-1))).squeeze(1)
            solution_depot_idx = torch.nonzero(solutions == 0)[:, 1].view(batch_size, -1)

            num_depot = max(label_depot_idx.size(1), solution_depot_idx.size(1))
            more_cir = (label_depot_idx[:, -1] != self.env.solutions.size(-1) - 1)
            max_solution_length = max(self.env.solutions.size(1), solutions.size(1))
            for b_idx in range(batch_size):
                res = []
                crt_solutions = solutions[b_idx]
                crt_label = self.env.solutions[b_idx]
                crt_label_depot_idx = label_depot_idx[b_idx]
                crt_solution_depot_idx = solution_depot_idx[b_idx]
                label_solution = []
                model_solution = []
                crt_more_cir = more_cir[b_idx]
                if model_better[b_idx]:
                    recounter += 1
                    for d_idx in range(solution_depot_idx.size(-1) - 1):
                        head = crt_solution_depot_idx[d_idx]
                        tail = crt_solution_depot_idx[d_idx + 1]
                        if tail > head + 1:
                            model_solution.append(crt_solutions[head + 1: tail].tolist())
                        else:
                            break
                    for tour in model_solution:
                        res += [0]
                        res += tour
                else:
                    for d_idx in range(num_depot - 1):
                        if d_idx < label_depot_idx.size(1) - 1:
                            head = crt_label_depot_idx[d_idx]
                            tail = crt_label_depot_idx[d_idx + 1]
                            if tail > head + 1:
                                label_solution.append(crt_label[head + 1: tail].tolist())
                        elif crt_more_cir and d_idx == label_depot_idx.size(1) - 1:
                            tail = crt_label_depot_idx[d_idx]
                            label_solution.append(crt_label[tail+1:].tolist())

                        if d_idx < solution_depot_idx.size(1) - 1:
                            head = crt_solution_depot_idx[d_idx]
                            tail = crt_solution_depot_idx[d_idx + 1]
                            if tail > head + 1:
                                model_solution.append(crt_solutions[head + 1: tail].tolist())

                    if len(model_solution) == len(label_solution):
                        resorted, resorted_ = sort_label_by_solution2(model_solution, label_solution)
                        if resorted_:
                            recounter += 1
                        for tour in resorted:
                            res += [0]
                            res += tour
                    else:
                        for tour in label_solution:
                            res += [0]
                            res += tour

                ori_solution = torch.tensor(res)
                mask_size = ori_solution.size(-1)
                triu = torch.triu(torch.ones((mask_size, mask_size), dtype=torch.bool))
                tril = torch.tril(torch.ones((mask_size, mask_size), dtype=torch.bool), diagonal=-1)
                mask = torch.cat([triu, tril], dim=1)

                ori_solution_rep = torch.cat([ori_solution, ori_solution], dim=0).unsqueeze(0).expand(mask_size, -1)
                ori_solution_rep = ori_solution_rep[mask].view(mask_size, mask_size)

                ori_idx = ori_solution_rep[:, 1].sort(0)[1]
                depot_num = mask_size - problem_size
                equal_solution = torch.index_select(ori_solution_rep, 0, ori_idx)[depot_num:]

                flip_solution_rep = torch.fliplr(ori_solution_rep)
                flip_idx = flip_solution_rep[:, 1].sort(0)[1]
                flip_solution = torch.index_select(flip_solution_rep, 0, flip_idx)[depot_num:]

                merge_mask = (flip_solution[:, 0] == 0).unsqueeze(-1).expand_as(flip_solution)
                equal_solution[merge_mask] = flip_solution[merge_mask]

                padding = torch.zeros((problem_size, max_solution_length - equal_solution.size(-1)), dtype=torch.long)
                equal_solution = torch.cat((equal_solution, padding), dim=1)

                resorted_label.append(equal_solution)

        resorted_label = torch.stack(resorted_label, dim=0)
        cut_idx = torch.count_nonzero(resorted_label.sum(1).sum(0))
        resorted_label = resorted_label[:, :, :cut_idx]
        return resorted_label, recounter, model_dist, pomo_idx


def sort_label_by_solution2(solution, label):
    sorted_label = []
    label_copy = label.copy()

    sorted_ = False
    for s_idx in range(len(solution)):
        best_idx = 0
        best_score = 0
        reverse_flag = False
        crt_s_set = set(solution[s_idx])
        for l_idx in range(len(label_copy)):
            crt_l_set = set(label_copy[l_idx])
            score = len(crt_s_set.intersection(crt_l_set))
            if score > best_score:
                best_score = score
                best_idx = l_idx
        score = similar(solution[s_idx], label_copy[best_idx])
        score_lr = similar(solution[s_idx], label_copy[best_idx][::-1])
        if score_lr > score:
            sorted_label.append(label_copy[best_idx][::-1])
            reverse_flag = True
        else:
            sorted_label.append(label_copy[best_idx])

        del label_copy[best_idx]


        if s_idx != best_idx or reverse_flag:
            sorted_ = True
    assert len(label_copy) == 0

    return sorted_label, sorted_

def similar(a, b):
    return SequenceMatcher(None, a, b).quick_ratio()