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

import logging
from utils.utils import create_logger, copy_all_src

from TSPTesterrrclib import TSPTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size_low': 500,
    'problem_size_high': 1000,
    'sub_size': 100,
    'pomo_size': 2,
    'sample_size': 8,
    'optimal': {
        100: 7.7632,
        200: 10.7036,
        500: 16.5215,
        1000: 23.1199,
    },
}

model_p_params = {
    'embedding_dim': 64,
    'depth': 12,
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

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        't_enable': True,  # enable loading pre-trained conquering model
        't_path': './',  # directory path of pre-trained conquering model.
        't_epoch': 460,  # epoch version of pre-trained conquering model to laod.
        'p_enable': True,  # enable loading pre-trained dividing model
        'p_path': './',  # directory path of pre-trained dividing model.
        'p_epoch': 460,  # epoch version of pre-trained dividing model to laod.
    },
    'data_load': 're_generate_test_TSP1000_0423_n128.txt',
    'test_episodes': 1, # size of the test set, should be 1
    'test_batch_size': 1, # batch size of testing, should be 1
    'augmentation_enable': False,
    'aug_factor': 50, # \alpha in paper
    'aug_batch_size': 1,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp100_longTrain',
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

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      model_p_params=model_p_params,
                      tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
