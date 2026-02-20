import yaml


from train import process_train, process_pre_filter, process_test, process_test_icl

def load_config(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

import os
import torch
import numpy as np
import random

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) # Current CPU
    torch.cuda.manual_seed(seed) # Current GPU
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
    np.random.seed(seed) # Numpy module
    random.seed(seed) # Python random module
    torch.backends.cudnn.benchmark = False # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization



if __name__ == '__main__':
    
    config = load_config('config.yaml')
    seed_everything(config['seed'])
    if config["stage"]["pre_filter"]:
        process_pre_filter(config, step_1=True, step_2=True, step_3=True)
    if config["stage"]["train"]:
        process_train(config, step_1=False, step_2=False, step_3=False, step_4=True)
    if config["stage"]["test"]:
        process_test(config, step1=False, step2=False, step3=True)
    
    # process_test_icl(config)
