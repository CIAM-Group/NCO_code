DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import logging
from LEHD.utils.utils import create_logger, copy_all_src
from LEHD.CVRP.VRPTrainer import VRPTrainer as Trainer

##########################################################################################
# parameters
b = os.path.abspath(".").replace('\\', '/')

training_data_path = b + '/data/vrp100_hgs_train_100w.txt'

env_params = {
    'data_path' : training_data_path,
    'mode': 'train',
    'sub_path': True
}

model_params = {
    'mode': 'train',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
                 },
    'scheduler': {
        'milestones': [1 * i for i in range(1, 40)],
        'gamma': 0.9
                 }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 40,
    'train_episodes': 1000000,
    'train_batch_size': 1024,
    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 10000,
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
        'enable': False ,  # enable loading pre-trained model
        'path': './result/20230817_235537_train',  # directory path of pre-trained model and log files saved.
        'epoch': 40,  # epoch version of pre-trained model to laod.
                  }
    }

logger_params = {
    'log_file': {
        'desc': 'train',
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
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 4
    trainer_params['train_episodes'] = 100
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
