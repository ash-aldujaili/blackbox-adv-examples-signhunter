"""
Implements the experiment of keeping K coords according to the crit
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from collections import OrderedDict

import numpy as np

from attacks.whitebox.eval_attack import main as eval_attack
from attacks.whitebox.run_attack import main as exec_attack
from utils.helper_fcts import data_path_join, config_path_join, create_dir, get_files
from utils.plt_fcts import plot_keep_k_sign_exp

# EXPERIMENT GLOBAL PARAMETERS
np.random.seed(1)
config_files = ['cifar10_topk_linf_config.json',
                'cifar10_topk_l2_config.json',
                'mnist_topk_linf_config.json',
                'mnist_topk_l2_config.json',
                'imagenet_topk_linf_config.json',
                'imagenet_topk_l2_config.json'
                ]

# for efficiency batch size are customized for each dataset
batch_sz = [100, 100, 200, 200, 50, 50]

_dir = data_path_join('keep_k_res')
create_dir(_dir)

num_eval_examples = 1000

for idx, _cf in enumerate(config_files):
    eval_batch_size = batch_sz[idx]
    res = {}
    print(_cf)
    config_file = config_path_join(_cf)
    dset = _cf.split('_')[0]
    p = _cf.split('_')[2]
    res[dset] = {}
    res['retain_p'] = list(np.linspace(0, 1, 11))
    with open(config_file, 'r') as f:
        config_json = json.load(f, object_pairs_hook=OrderedDict)
    config_json["eval_batch_size"] = eval_batch_size
    config_json["num_eval_examples"] = num_eval_examples
    for crit in ['random', 'top']:
        config_json["attack_config"]["crit"] = crit
        res[dset][crit] = {}
        res[dset][crit]['adv_acc'] = []
        res[dset][crit]['nat_acc'] = []
        for retain_p in res['retain_p']:
            config_json["attack_config"]["retain_p"] = retain_p
            # write the new config
            with open(config_file, 'w') as f:
                json.dump(config_json, f, indent=4)
            exec_attack(config_file)
            _res = eval_attack(config_file)
            res[dset][crit]['adv_acc'].append(_res[0])
            res[dset][crit]['nat_acc'].append(_res[1])

    with open(os.path.join(_dir, '{}_{}_res.json'.format(dset, p)), 'w') as f:
        print(res)
        json.dump(res, f, indent=4)

# plot keep_k signal
files = get_files(data_path_join('keep_k_res'), 'json')
plot_keep_k_sign_exp(files)
