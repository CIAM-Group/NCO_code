
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

##########################################################################################
# Machine Environment Config

DEBUG_MODE = True

USE_CUDA = False
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, "../..")
sys.path.insert(0, "..")

##########################################################################################
# cuda

import torch

if USE_CUDA:
    torch.cuda.set_device(CUDA_DEVICE_NUM)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src


# =========  LB  ====================
from FFSP.FFSP_Greedy.LowerBound import calc_lower_bound

# ==========  SJF ===================
from FFSP.FFSP_Greedy.GreedyAlgorithm import GreedyAlgorithm
# ==========  GA ===================
from FFSP.FFSP_GA.FFSPEnv_MH import FFSPEnv_MH as GA_Env
from FFSP.FFSP_GA.GA import GA as GA_Trainer

# ==========  PSO ===================
from FFSP.FFSP_PSO.FFSPEnv_MH import  FFSPEnv_MH as PSO_Env
from FFSP.FFSP_PSO.PSO import PSO as PSO_Trainer



##########################################################################################
# parameters

# Common Params
logger_params = {
    'log_file': {
        'desc': 'ffsp_MH_test',
        'filename': 'ffsp_MH_test_log.txt'
    }
}

env_params = {
    'stage_cnt': 3,
    'machine_cnt': 4,
    'job_cnt': 20,
    'min_process_time': 2,
    'max_process_time': 10,
    'distribution': 'uniform',
    'same_process_time_within_stage': False,
    'seed': 123456789,
    'file': {
        'load_filename': './data/unrelated_10000_problems_444_job20_2_10.pt',
        'problem_index': [0]
    }
}

env_params['max_step_length'] = int(env_params['stage_cnt'] * env_params['job_cnt'] * env_params['max_process_time'] / env_params['machine_cnt'])

# GA
ga_trainer_params = {
    'epochs': 1000,
    'target_fitness': None,
    'num_of_chromosome': 25,
    'num_of_gene': env_params['job_cnt'],
    'selection_method': 'softmax',
    'selection_ratio': 0.1,
    'crossover_method': 'FFSP',
    'crossover_ratio': 0.3,
    'mutation_mode': None,
    'mutation_rate': 0.3,
    'max_mutation_rate': 0.3,
    'logging': {
        'img_save_interval': 1,
        'filename': 'figure'
    }
}

# PSO
pso_trainer_params = {
    'n_particles': 40,
    'epochs': 1000,
    'c1': 1.49445,
    'c2': 1.49445
}


def set_debug_mode():
    global ga_trainer_params
    global pso_trainer_params
    ga_trainer_params['epochs'] = 2
    ga_trainer_params['num_of_chromosome'] = 4
    pso_trainer_params['epochs'] = 2
    pso_trainer_params['n_particles'] = 4




def print_config(logger):
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################
# main

def main():
    create_logger(**logger_params)

    if DEBUG_MODE:
        set_debug_mode()

    logger = logging.getLogger('root')
    print_config(logger)

    sjf_best_result = []
    ga_best_result = []
    pso_best_result = []

    test_prob_index = env_params['file']['problem_index']
    test_iter = 1

    for idx in range(len(test_prob_index)):
        env_params['file']['problem_index'] = idx

        # ==========  SJF ===================
        greedy = GreedyAlgorithm(episodes=1, batch_size=1, env_params=env_params)
        sjf_rst = greedy.solve(method='SJFv2', plot='render')
        end_time = greedy.env.job_end_time_on_stage.max().to(dtype=torch.long)
        schedule = greedy.env.total_schedule_info[0, :, :end_time]
        sjf_best_result.append(sjf_rst)

        ga_result = []
        pso_result = []

        for i in range(test_iter):
            # ==========  GA ===================
            ga_env = GA_Env(**env_params)
            ga_trainer = GA_Trainer(env=ga_env, schedule=schedule, trainer_params=ga_trainer_params)
            cmax = ga_trainer.solve()
            ga_result.append(cmax)

            # ==========  PSO ===================
            pso_env = PSO_Env(**env_params)
            pso_trainer = PSO_Trainer(env=pso_env, schedule=None, **pso_trainer_params)
            pso_rst = pso_trainer.optimize()
            pso_result.append(pso_rst)

            logger.info("test_iter : {}, ga_cmax : {}, pso_cmax : {}".format(i, cmax, pso_rst))


        # =========  LB  ====================
        machine_cnt_list = env_params['machine_cnt'] if type(env_params['machine_cnt']) == list \
            else [env_params['machine_cnt']] * env_params['stage_cnt']
        lb = calc_lower_bound(job_cnt=env_params['job_cnt'],
                              stage_cnt=env_params['stage_cnt'],
                              machine_cnt_list=machine_cnt_list,
                              process_time=pso_trainer.problem.process_time[0, :, 1:].t(),
                              problem_type='unrelated')

        logger.info("------------------------------------------------------------------------------------------------------")
        logger.info("[LB] problem_{} : {}".format(idx, lb))

        logger.info("[GA] problem_{}, cmax : {}".format(idx, ga_result))

        logger.info("[GA] problem_{} min : {}, max : {}, average : {}".format(idx, min(ga_result), max(ga_result),
                                                                         sum(ga_result) / len(ga_result)))

        logger.info("[PSO_SJF] problem_{}, cmax : {}".format(idx, pso_result))
        logger.info("[PSO_SJF] problem_{} min : {}, max : {}, average : {}".format(idx, min(pso_result), max(pso_result),
                                                                   sum(pso_result) / len(pso_result)))
        logger.info("------------------------------------------------------------------------------------------------------")


        ga_best_result.append((min(ga_result)))
        pso_best_result.append(min(pso_result))


    logger.info("[Total best_result] \n LB : {} \n SJF : {} \n GA : {} \n  PSO(with SJF) : {}".format(lb, sjf_best_result,
                                                                                                          ga_best_result,  pso_best_result))
    logger.info("[Min] \n SJF : {} \n GA : {} \n PSO(with SJF) : {}".format(min(sjf_best_result),
                                                                                                  min(ga_best_result), min(pso_best_result)))
    logger.info("[Max] \n SJF : {} \n GA : {} \n  PSO(with SJF) : {}".format(max(sjf_best_result),
                                                                                                  max(ga_best_result),  max(pso_best_result)))
    logger.info("[Average] \n SJF : {} \n GA : {} \n PSO(with SJF) : {}".format(sum(sjf_best_result) / len(sjf_best_result),
                                                                                                      sum(ga_best_result) / len(ga_best_result),
                                                                                                      sum(pso_best_result) / len(pso_best_result)))


if __name__ == "__main__":
    main()
