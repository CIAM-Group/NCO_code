##########################################################################################
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
import logging
import numpy as np
from LEHD.utils.utils import create_logger, copy_all_src
from LEHD.CVRP.Tester_inCVRPlib import VRPTester as Tester

##########################################################################################
# parameters

problem_size = 0      # testing problem size
test_in_vrplib = True  # test in vrplib or not
Use_RRC = False          # decode method: use RRC or not (greedy)
RRC_budget = 1000         # RRC budget

########### model ###############

model_load_path = 'result/20230817_235537_train'
model_load_epoch = 40


if test_in_vrplib == True:
    problem_size = 0

if not Use_RRC:
    RRC_budget = 0

mode = 'test'
test_paras = {
   # problem_size: [filename, episode, batch]
    100: [ 'vrp100_test_lkh.txt',10000,10000, 0],
    200: ['vrp200_test_lkh.txt', 128, 128, 0],
    500: ['vrp500_test_lkh.txt', 128, 128, 0],
    1000: ['vrp1000_test_lkh.txt', 128, 128, 0],
    0: ['vrplib_192instances.txt', 192, 1, 0 ]
}


##########################################################################################
# parameters
b = os.path.abspath(".").replace('\\', '/')

env_params = {
    'mode': mode,
    'test_in_vrplib':test_in_vrplib,
    'vrplib_path': b +f'/data/{test_paras[problem_size][0]}',
    'data_path': b + f"/data/{test_paras[problem_size][0]}",
    'sub_path': False,
    'RRC_budget': RRC_budget
}


model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'begin_index': test_paras[problem_size][3],
    'test_episodes': test_paras[problem_size][1],   # 65
    'test_batch_size': test_paras[problem_size][2],
}

logger_params = {
    'log_file': {
        'desc': f'test__vrp{problem_size}',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main_test(epoch,path,use_RRC=None):
    if DEBUG_MODE:
        _set_debug_mode()
    create_logger(**logger_params)
    _print_config()
    tester_params['model_load']={
        'path': path,
        'epoch': epoch,
    }
    if use_RRC is not None:
        env_params['RRC_budget']=0

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()


    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap

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
    path = f'./{model_load_path}'
    allin = []
    for i in [model_load_epoch]:
        score_optimal, score_student,gap = main_test(i,path)
        allin.append([ score_optimal, score_student,gap])
    np.savetxt('result.txt',np.array(allin),delimiter=',')
