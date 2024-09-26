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

from ATSPTesterrrc import ATSPTesterrrc as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size_low': 250,
    'problem_size_high': 500,
    'sub_size': 50,
    'pomo_size': 50,
    'sample_size': 30,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
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
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'argmax',
    'one_hot_seed_cnt': 50,  # must be >= node_cnt
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 0
    },
    'optimizer_p': {
        'lr': 1e-4,
        'weight_decay': 0
    },
    'scheduler': {
        'milestones': [3001,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'data_load_1000': 're_generate_test_TSP1000_0423_n128.txt',
    'data_load_500': 're_generate_test_TSP500_0423_n128.txt',
    'epochs': 3100,
    'train_episodes': 1000,
    'train_batch_size': 1,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        't_enable': True,  # enable loading pre-trained conquering model
        't_path': './',  # directory path of pre-trained conquering model.
        't_epoch': 300,  # epoch version of pre-trained conquering model to laod.
        'p_enable': True,  # enable loading pre-trained dividing model
        'p_path': './',  # directory path of pre-trained dividing model.
        'p_epoch': 300,  # epoch version of pre-trained dividing model to laod.
    },
    'validation_test_episodes': 128, # size of the test set
    'validation_test_batch_size': 2, # batch size of testing
    'validation_aug_factor': 20, # \alpha in paper
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n500__3000epoch',
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

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      model_p_params=model_p_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
