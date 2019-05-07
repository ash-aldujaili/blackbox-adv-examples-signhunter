"""
Script for adversarial cone computation based on
the SignHunter estimation in comparison to the
linearization of the model (i.e., maximal alignment of the gradient).
For cones of size {1, 4, 16, 36, 64, 100}
For more details, see Tramer et al., 2018
"""
import math
import json
import os
import pickle
import numpy as np
import tensorflow as tf
import torch as ch
from datasets.dataset import Dataset
from utils.compute_fcts import tf_nsign, sign
from utils.helper_fcts import config_path_join, data_path_join, \
    construct_model, get_model_file, create_dir, get_dataset_shape, src_path_join
from utils.plt_fcts import plot_adv_cone_res

from attacks.blackbox.sign_attack import SignAttack

# EXPERIMENT GLOBAL PARAMETERS
np.random.seed(1)


# Adv. Cone Order
K = [1, 4, 16, 36, 64, 100]
EPS = [4 / 255., 10/255., 16/255.]

NUM_DATA_PTS = 500 # the  number of data points for which we compute the adv cone

def update_adv_cone_metrics(x_batch, g_batch, early_stop_crit_fct, res):
    _shape = x_batch.shape
    batch_size = _shape[0]
    n = g_batch.shape[1]
    # go over the k-dim adversarial cone and check evasion
    for ik, k in enumerate(K):
        if k == 1:
            H = np.ones((1, 1))
        else:
            H = np.load(src_path_join('reg_hadamard_mats/reg_hadamard_mat_order-{}.npy'.format(k)))
        for ie, eps in enumerate(EPS):
            mis_count = np.zeros(batch_size)
            for _ in range(k):
                R = np.zeros((batch_size, n))
                R[:, :n // k * k] = np.repeat(np.repeat(H[_,None], n // k, axis=1), batch_size, axis=0)
                R *= g_batch
                X = x_batch.reshape(R.shape) + eps * R
                mis_count += early_stop_crit_fct(X.reshape(_shape))
            res[ie, ik] += sum(mis_count == k)


def main():
    """
    main routine of the experiment, results are stored in data
    :return:
    """
    # results dir setup
    _dir = data_path_join('adv_cone_exp')
    create_dir(_dir)

    # for reproducibility
    np.random.seed(1)

    # init res data structure
    res = {'epsilon': EPS, 'adv-cone-orders': K, 'sign-hunter-step': 10 / 255., 'num_queries': 1000}

    # config files
    config_files = ['imagenet_sign_linf_config.json', 'imagenet_sign_linf_ens_config.json']

    # config load
    for _n, _cf in zip(['nat', 'adv'], config_files):
        tf.reset_default_graph()
        config_file = config_path_join(_cf)
        with open(config_file) as config_file:
            config = json.load(config_file)

        # dset load
        dset = Dataset(config['dset_name'], config['dset_config'])
        dset_dim = np.prod(get_dataset_shape(config['dset_name']))

        # model tf load/def
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

        # init res file: ijth entry of the matrix should
        # denote the probability that at least K[j] orthogonal vectors r_p such that
        # x + EPS[i] * r_p is misclassified
        res[_n] = {
            'grad-sign': np.zeros((len(EPS), len(K))),
            'sign-hunter': np.zeros((len(EPS), len(K)))
        }

        # main block of code
        attacker = SignAttack(
            **config['attack_config'],
            lb=dset.min_value,
            ub=dset.max_value
        )

        # to over-ride attacker's configuration
        attacker.max_loss_queries = res['num_queries']
        attacker.epsilon = res['sign-hunter-step']

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9))
        ) as sess:
            # Restore the checkpoint
            saver.restore(sess, model_file)

            # Iterate over the samples batch-by-batch
            num_eval_examples = int(NUM_DATA_PTS / 0.7) # only correctly classified are considered (boost the total number sampled by the model accuracy)~
            eval_batch_size = 30 # config['eval_batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
            # consider only correctly classified pts
            eff_num_eval_examples = 0
            print('Iterating over {} batches'.format(num_batches))

            for ibatch in range(num_batches):
                if eff_num_eval_examples >= NUM_DATA_PTS:
                    break
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}:({},{})'.format(bend - bstart, bstart, bend))

                x_batch, y_batch = dset.get_eval_data(bstart, bend)

                # filter misclassified pts
                is_correct = sess.run(model.correct_prediction, feed_dict={
                    model.x_input: x_batch,
                    model.y_input: y_batch
                })

                # pass only correctly classified data till the NUM_DATA_PTS
                x_batch = x_batch[is_correct, :]
                y_batch = y_batch[is_correct]

                batch_size = min(NUM_DATA_PTS - eff_num_eval_examples, sum(is_correct))
                x_batch = x_batch[:batch_size, :]
                y_batch = y_batch[:batch_size]

                eff_num_eval_examples += batch_size

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

                attacker.run(x_batch, loss_fct, early_stop_crit_fct, metric_fct)

                # get attacker adv perturb estimate:
                g_batch = attacker.get_gs().cpu().numpy()
                # compute adv cone
                update_adv_cone_metrics(x_batch, g_batch, early_stop_crit_fct, res[_n]['sign-hunter'])

                # get gradient as adv perturb estimate:
                g_batch = sign(grad_fct(x_batch))
                # compute adversarial cones
                update_adv_cone_metrics(x_batch, g_batch, early_stop_crit_fct, res[_n]['grad-sign'])
                print(attacker.summary())
                print("Adv. Cone Stats for SH:")
                print(res[_n]['sign-hunter'])
                print("Adv. Cone Stats for GS:")
                print(res[_n]['grad-sign'])

        res[_n]['sign-hunter'] /= eff_num_eval_examples
        res[_n]['grad-sign'] /= eff_num_eval_examples

    p_fname = os.path.join(_dir, 'adv-cone_step-{}.p'.format(res['sign-hunter-step']))
    with open(p_fname, 'wb') as f:
        pickle.dump(res, f)

    plot_adv_cone_res(p_fname)


if __name__ == '__main__':
    main()
