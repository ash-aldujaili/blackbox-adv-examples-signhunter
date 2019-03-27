"""
Implements the experiment of distribution of hamming distances

where the hamming distance is measured among sign-vectors of gradients (wrt loss) around data points
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from datasets.dataset import Dataset
from utils.compute_fcts import tf_nsign
from utils.helper_fcts import get_model_file, construct_model
from utils.helper_fcts import load_config, data_path_join, get_dataset_shape

# EXPERIMENT GLOBAL PARAMETERS
NUM_SAMPLES = 100
EVAL_BATCH_SIZE = 1  # do not change this
NUM_EVAL_EXAMPLES = 1000


def main(config_file):
    np.random.seed(1)
    tf.reset_default_graph()
    config = load_config(config_file)

    # dataset
    dset_name = config['dset_name']
    dset = Dataset(dset_name, config['dset_config'])
    dset_shape = get_dataset_shape(config['dset_name'])
    dim = np.prod(dset_shape)

    # model and computational graph
    model_file = get_model_file(config)
    with tf.device(config['device']):
        model = construct_model(dset_name)
        grad = tf.gradients(model.xent, model.x_input)[0]
        flat_grad = tf.reshape(grad, [NUM_SAMPLES, -1])
        flat_sgn = tf_nsign(flat_grad)
        norm_flat_grad = tf.div(flat_grad, tf.norm(flat_grad, axis=1, keepdims=True))

        sim_mat = tf.matmul(norm_flat_grad, norm_flat_grad, transpose_b=True)
        sims = tf.gather_nd(sim_mat, list(zip(*np.triu_indices(NUM_SAMPLES, k=1))))

        dist_mat = (dim - tf.matmul(flat_sgn, flat_sgn, transpose_b=True)) / 2.0
        dists = tf.gather_nd(dist_mat, list(zip(*np.triu_indices(NUM_SAMPLES, k=1))))

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(
        data_path_join("hamming_dist_exp")
    )

    epsilon = config['attack_config']['epsilon']
    num_batches = int(math.ceil(NUM_EVAL_EXAMPLES / EVAL_BATCH_SIZE))

    for _epsilon in np.linspace(epsilon / 10, epsilon, 3):
        # histogram recorder
        tf.summary.histogram(
            "{}_hamming_dist_xr_sgn_grad_eps_{}_{}_samples_{}_pts".format(dset_name, _epsilon, NUM_SAMPLES,
                                                                          NUM_EVAL_EXAMPLES),
            dists
        )

        tf.summary.histogram(
            "{}_cosine_sim_xr_grad_eps_{}_{}_samples_{}_pts".format(dset_name, _epsilon, NUM_SAMPLES,
                                                                    NUM_EVAL_EXAMPLES),
            sims
        )

        summs = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore the checkpoint
            saver.restore(sess, model_file)
            # Iterate over the data points one-by-one

            print('Iterating over {} batches'.format(num_batches))

            for ibatch in range(num_batches):
                bstart = ibatch * EVAL_BATCH_SIZE
                bend = min(bstart + EVAL_BATCH_SIZE, NUM_EVAL_EXAMPLES)
                print('batch size: {}'.format(bend - bstart))

                x_batch, y_batch = dset.get_eval_data(bstart, bend)

                xr_batch = np.clip(
                    x_batch + np.random.uniform(-_epsilon, _epsilon, [NUM_SAMPLES, *x_batch.shape[1:]]),
                    dset.min_value,
                    dset.max_value
                )
                yr_batch = y_batch.repeat(NUM_SAMPLES)

                summ_val = sess.run(summs, feed_dict={
                    model.x_input: xr_batch,
                    model.y_input: yr_batch
                })

                writer.add_summary(summ_val, global_step=ibatch)


if __name__ == '__main__':
    # you can view the results with tensorboard
    # tensorboard --logdir ../data/hamming_dist_exp
    main('mnist_topk_config.json')
    main('imagenet_topk_config.json')
    main('cifar10_topk_config.json')
