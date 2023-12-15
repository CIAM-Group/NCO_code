##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import
import torch
import logging
from utils.utils import create_logger, copy_all_src

from CVRPTester import CVRPTester as Tester_regret

##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params_regret = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../../pretrained/vrp100',
        'epoch': 8100,
    },
    'test_episodes': 10000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': True,
        'filename': '../../../data/vrp100.pt'
    },
}

if tester_params_regret['augmentation_enable']:
    tester_params_regret['test_batch_size'] = tester_params_regret['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__CVRP100',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()


    tester_regret = Tester_regret(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params_regret)

    copy_all_src(tester_regret.result_folder)

    tester_regret.run()


def _set_debug_mode():
    global tester_params_regret
    tester_params_regret['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
