"""
Implements the experiment of computing the partial derivatives of the loss function
wrt to the data points (or random perturbations within their l-inf epsilon-ball
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from datasets.dataset import Dataset
from utils.helper_fcts import get_model_file, \
    construct_model, load_config, get_dataset_shape, \
    data_path_join

# from utils.plt_fcts import plot_as_3d_ts


# EXPERIMENT GLOBAL PARAMETERS
NUM_EVAL_EXAMPLES = 1000


def main(config_file):
    np.random.seed(1)
    tf.reset_default_graph()

    config = load_config(config_file)

    dset_name = config['dset_name']
    dset = Dataset(dset_name, config['dset_config'])
    model_file = get_model_file(config)
    epsilon = config['attack_config']['epsilon']

    with tf.device(config['device']):
        model = construct_model(dset_name)
        abs_grad = tf.abs(tf.gradients(model.xent, model.x_input)[0])

    # histogram recorder
    # place holder for dx at x0 and x_rand
    dxo = tf.placeholder(tf.float32, shape=get_dataset_shape(dset_name))
    tf.summary.histogram("{}_part_deriv_mag_xo".format(dset_name), dxo)

    dxr = tf.placeholder(tf.float32, shape=get_dataset_shape(dset_name))
    tf.summary.histogram("{}_part_deriv_mag_xr".format(dset_name), dxr)

    writer = tf.summary.FileWriter(
        data_path_join("partial_derivative_exp")
    )
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)
        # Iterate over the samples batch-by-batch
        eval_batch_size = config['eval_batch_size']
        num_batches = int(math.ceil(NUM_EVAL_EXAMPLES / eval_batch_size))

        # dxs = None  # grads accumulator

        print('Iterating over {} batches'.format(num_batches))

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, NUM_EVAL_EXAMPLES)
            print('batch size: {}'.format(bend - bstart))

            x_batch, y_batch = dset.get_eval_data(bstart, bend)
            xr_batch = np.clip(x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape),
                               dset.min_value,
                               dset.max_value)
            # print(y_batch)
            dxo_batch = sess.run(abs_grad, feed_dict={
                model.x_input: x_batch,
                model.y_input: y_batch
            })

            dxr_batch = sess.run(abs_grad, feed_dict={
                model.x_input: xr_batch,
                model.y_input: y_batch
            })

            for i, step in enumerate(range(bstart, bend)):
                summ = sess.run(summaries, feed_dict={dxo: dxo_batch[i],
                                                      dxr: dxr_batch[i]})
                writer.add_summary(summ, global_step=step)

        # dxs = dxo_batch if dxs is None else np.vstack((dxs, dxo_batch))
        # bin_edges = np.histogram_bin_edges(np.array(dxs).reshape(-1), bins='auto')
        # dx_hists = np.array([np.histogram(_, bin_edges)[0] for _ in dxs])
        # plot_as_3d_ts(dx_hists,
        #               xticks=bin_edges[:-1],
        #               ylabel='test image index',
        #               xlabel='partial derivative magnitude',
        #               zlabel='counts')


if __name__ == '__main__':
    # you can view the results with tensorboard
    # tensorboard --logdir ../data/partial_derivative_exp
    main('mnist_topk_config.json')
    main('imagenet_topk_config.json')
    main('cifar10_topk_config.json')
