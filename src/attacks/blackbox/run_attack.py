"""
Script for running black-box attacks and report their metrics

Note this script makes use of both pytorch and tensor flow to make use of the GPUs
pytorch: for performing the perturbations
tensorflow: for querying the loss/gradient oracle

A good practice is to let pytorch be allocated on GPU:0
and let tensorflow be allocated on GPU:1
CPU mode works fine too.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




import tensorflow as tf
import torch as ch
import time
import numpy as np
import json
import math
import os
import pandas as pd

from utils.helper_fcts import config_path_join, data_path_join, \
    construct_model, get_model_file, create_dir, get_dataset_shape
from utils.latex_fcts import res_json_2_tbl_latex
from utils.compute_fcts import tf_nsign, sign
from datasets.dataset import Dataset
from utils.plt_fcts import plt_from_h5tbl


from attacks.blackbox.nes_attack import NESAttack
from attacks.blackbox.cheat_attack import CheatAttack
from attacks.blackbox.bandit_attack import BanditAttack
from attacks.blackbox.zo_sign_sgd_attack import ZOSignSGDAttack
from attacks.blackbox.sign_attack import SignAttack


# to run the attacks on a quadratic function with no constraint
# i.e. a concave fct with a single global solution
IS_DEBUG_MODE = False

if __name__ == '__main__':
    exp_id = 'ens_imagenet'
    print("Running Experiment {} with DEBUG MODE {}".format(exp_id, IS_DEBUG_MODE))
    cfs = [
        #'mnist_zosignsgd_linf_config.json',
        #'mnist_nes_linf_config.json',
        #'mnist_sign_linf_config.json',
        #'mnist_bandit_linf_config.json',
        #'mnist_zosignsgd_l2_config.json',
        #'mnist_nes_l2_config.json',
        #'mnist_sign_l2_config.json',
        #'mnist_bandit_l2_config.json',
        #'cifar10_zosignsgd_linf_config.json',
        #'cifar10_nes_linf_config.json',
        #'cifar10_sign_linf_config.json',
        #'cifar10_bandit_linf_config.json',
        #'cifar10_zosignsgd_l2_config.json',
        #'cifar10_nes_l2_config.json',
        #'cifar10_sign_l2_config.json',
        #'cifar10_bandit_l2_config.json',
        #'imagenet_zosignsgd_linf_config.json',
        #'imagenet_nes_linf_config.json',
        #'imagenet_sign_linf_config.json',
        #'imagenet_bandit_linf_config.json',
        #'imagenet_zosignsgd_l2_config.json',
        #'imagenet_nes_l2_config.json',
        #'imagenet_sign_l2_config.json',
        #'imagenet_bandit_l2_config.json'
        'imagenet_sign_linf_ens_config.json'
    ]

    # create/ allocate the result json for tabulation
    data_dir = data_path_join('blackbox_attack_exp')
    create_dir(data_dir)
    res = {}

    # create a store for logging / if the store is there remove it
    store_name = os.path.join(data_dir, '{}_tbl.h5'.format(exp_id))
    offset = 0
    # rewrite all the results alternatively one could make use of `offset` to append to the h5 file above.
    if os.path.exists(store_name):
        os.remove(store_name)

    for _cf in cfs:
        x_adv = []
        # for reproducibility
        np.random.seed(1)
        config_file = config_path_join(_cf)
        tf.reset_default_graph()

        with open(config_file) as config_file:
            config = json.load(config_file)

        dset = Dataset(config['dset_name'], config['dset_config'])
        dset_dim = np.prod(get_dataset_shape(config['dset_name']))

        model_file = get_model_file(config)
        with tf.device(config['device']):
            model = construct_model(config['dset_name'])
            flat_est_grad = tf.placeholder(tf.float32, shape=[None, dset_dim])
            flat_grad = tf.reshape(tf.gradients(model.xent, model.x_input)[0], [-1, dset_dim])
            norm_flat_grad = tf.maximum(
                tf.norm(flat_grad, axis=1, keepdims=True), np.finfo(np.float64).eps)
            norm_flat_est_grad = tf.maximum(
                tf.norm(flat_est_grad, axis=1, keepdims=True), np.finfo(np.float64).eps)
            cos_sim = tf.reduce_sum(
                tf.multiply(tf.div(flat_grad, norm_flat_grad),
                            tf.div(flat_est_grad, norm_flat_est_grad)), axis=1, keepdims=False)
            ham_sim = tf.reduce_mean(
                tf.cast(
                    tf.math.equal(tf_nsign(flat_grad), tf_nsign(flat_est_grad)),
                    dtype=tf.float32),
                axis=1, keepdims=False)

        # set torch default device:
        if 'gpu' in config['device'] and ch.cuda.is_available():
            ch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            ch.set_default_tensor_type('torch.FloatTensor')

        saver = tf.train.Saver()

        attacker = eval(config['attack_name'])(
            **config['attack_config'],
            #max_loss_queries=10000,
            lb=dset.min_value,
            ub=dset.max_value
        )

        # to over-ride attacker's configuration
        # attacker.max_loss_queries = 5000

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75))
        ) as sess:
            # Restore the checkpoint
            saver.restore(sess, model_file)

            # Iterate over the samples batch-by-batch
            num_eval_examples = config['num_eval_examples']
            eval_batch_size =  config['eval_batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()

            # batch offset in case of running the dataset over multiple nodes
            eval_offset = config['eval_offset'] if 'eval_offset' in config else 0

            for _ in range(num_batches):
                ibatch = int(_ + eval_offset / eval_batch_size)
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples + eval_offset)
                print('batch size: {}, batch id:{}, ({},{})'.format(bend - bstart, ibatch,
                    bstart, bend))

                x_batch, y_batch = dset.get_eval_data(bstart, bend)

                if IS_DEBUG_MODE:
                    # write the loss, grad, and metric for a concave quadratic fct
                    # to verify the functionality of the attacks
                    x_opt = 0.5 * dset.max_value

                    def loss_fct(xs):
                        _l = - np.sum((xs.reshape(xs.shape[0], -1) - x_opt) ** 2, axis=1)
                        #_l = - np.sum(xs.reshape(xs.shape[0], -1), axis=1)
                        return _l


                    def early_stop_crit_fct(xs):
                        return np.logical_not(np.ones(xs.shape[0]))


                    def metric_fct(xs, flat_est_grad_vals):
                        _grad_val = (- 2 * (xs - x_opt)).reshape(xs.shape[0], -1)
                        #_grad_val = -np.ones(xs.shape).reshape(xs.shape[0], -1)
                        norm_grad = np.linalg.norm(_grad_val, axis=1)
                        norm_est = np.linalg.norm(flat_est_grad_vals, axis=1)
                        _cos_sim_val = np.sum(_grad_val * flat_est_grad_vals, axis=1) / (norm_grad * norm_est)
                        _ham_sim_val = np.sum((sign(_grad_val) * sign(flat_est_grad_vals)) == 1., axis=1)
                        return _cos_sim_val, _ham_sim_val


                    # handy function for performance tracking (or for cheat attack)
                    def grad_fct(xs):
                        _grad_val = - 2 * (xs - x_opt)
                        #_grad_val = -np.ones(xs.shape).reshape(xs.shape[0], -1)
                        return _grad_val

                else:
                    def loss_fct(xs):
                        _l = sess.run(model.y_xent, feed_dict={
                            model.x_input: xs,
                            model.y_input: y_batch
                        })
                        return _l

                    def early_stop_crit_fct(xs):
                        _is_correct = sess.run(model.correct_prediction, feed_dict={
                            model.x_input: xs,
                            model.y_input: y_batch
                        })
                        return np.logical_not(_is_correct)

                    def metric_fct(xs, flat_est_grad_vals):
                        _cos_sim_val, _ham_sim_val = sess.run([cos_sim, ham_sim],
                                                feed_dict={
                                                    model.x_input: xs,
                                                    model.y_input: y_batch,
                                                    flat_est_grad: flat_est_grad_vals
                                                }
                        )
                        return _cos_sim_val, _ham_sim_val

                    # handy function for performance tracking (or for cheat attack)
                    def grad_fct(xs):
                        _grad_val = sess.run(flat_grad, feed_dict={
                            model.x_input: xs,
                            model.y_input: y_batch
                        })
                        return _grad_val

                if config['attack_name'] == 'CheatAttack':
                    # applicable only for cheatattack
                    attacker.set_grad_fct(grad_fct)

                x_batch_adv, logs_dict = attacker.run(x_batch, loss_fct, early_stop_crit_fct, metric_fct)
                if config['dset_name'] != 'imagenet':
                    x_adv.append(x_batch_adv)
                _len = len(logs_dict['iteration'])
                if _len != 0: # to save i/o ops
                    logs_dict['p'] = [config['attack_config']['p']] * _len
                    logs_dict['attack'] = [config['attack_name']] * _len
                    logs_dict['dataset'] = [config['dset_name']] * _len
                    logs_dict['batch_id'] = ibatch
                    logs_dict['idx'] = [_ + offset for _ in range(_len)]
                    offset += _len
                    pd.DataFrame(logs_dict).set_index('idx').to_hdf(store_name, 'tbl', append=True, min_itemsize={'p':3, 'attack':30, 'dataset':10})
                print(attacker.summary())

        print("Batches done after {} s".format(time.time()-start_time))

        if config['dset_name'] not in res:
            res[config['dset_name']] = [attacker.summary()]
        else:
            res[config['dset_name']].append(attacker.summary())

        if config['dset_name'] != 'imagenet':
            print('Storing examples')
            path = data_path_join(config["store_adv_path"])
            x_adv = np.concatenate(x_adv, axis=0)
            np.save(path, x_adv)
    # create latex table
    res_fname = os.path.join(data_dir, '{}_res.json'.format(exp_id))
    print("Storing tabular data in {}".format(res_fname))
    with open(res_fname, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)

    res_json_2_tbl_latex(res_fname)
    plt_from_h5tbl([store_name])






