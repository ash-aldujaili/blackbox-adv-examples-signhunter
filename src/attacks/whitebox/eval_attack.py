"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math

import numpy as np
import tensorflow as tf

from datasets.dataset import Dataset
from utils.helper_fcts import config_path_join, data_path_join, \
    get_dataset_shape, construct_model, \
    get_model_file, check_values, check_shape


def get_res(model_file, x_adv, epsilon,
            model, dset,
            num_eval_examples=10000,
            eval_batch_size=64,
            is_ignore_misclassified=True):
    saver = tf.train.Saver()

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_corr = 0
    num_corr_nat = 0 if is_ignore_misclassified else num_eval_examples

    x_nat, _ = dset.get_eval_data(0, num_eval_examples)
    l_inf = np.amax(np.abs(x_nat - x_adv))

    if l_inf > epsilon + 0.0001:
        print('maximum perturbation found: {}'.format(l_inf))
        print('maximum perturbation allowed: {}'.format(epsilon))
        return

    y_pred = []  # label accumulator

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = x_adv[bstart:bend, :]
            _, y_batch = dset.get_eval_data(bstart, bend)

            cur_corr, y_pred_batch = sess.run([model.num_correct, model.y_pred],
                                              feed_dict={model.x_input: x_batch,
                                                         model.y_input: y_batch})

            # measure accuracy only for those whose x_nat is not misclassified.
            x_batch = x_nat[bstart:bend, :]

            if is_ignore_misclassified:
                cur_corr_nat, y_pred_batch_nat = sess.run([model.num_correct, model.y_pred],
                                                          feed_dict={model.x_input: x_batch,
                                                                     model.y_input: y_batch})

                total_corr += sum((y_batch == y_pred_batch_nat) & (y_batch == y_pred_batch))
                num_corr_nat += cur_corr_nat
            else:
                total_corr += cur_corr

            y_pred.append(y_pred_batch)

        accuracy = total_corr / num_corr_nat
        accuracy_nat = num_corr_nat / num_eval_examples

        print('Accuracy (x_adv): {:.2f}% for {} pts'.format(100.0 * accuracy, num_corr_nat))
        print('Accuracy (x_nat): {:.2f}% for {} pts'.format(100.0 * accuracy_nat, num_eval_examples))
        y_pred = np.concatenate(y_pred, axis=0)
        np.save(data_path_join('pred.npy'), y_pred)
        print('Output saved at {}'.format(data_path_join('pred.npy')))

        return accuracy, accuracy_nat


def main(config_file):
    """
    :param config_file:
    :return:
    """
    # deallocate memory if any
    tf.reset_default_graph()
    # free_gpus()

    # load configs.
    with open(config_file) as config_file:
        config = json.load(config_file)

    # load dataset
    dset = Dataset(config['dset_name'], config['dset_config'])

    with tf.device(config['device']):
        model = construct_model(config['dset_name'])

    x_adv = np.load(data_path_join(config['store_adv_path']))

    model_file = get_model_file(config)

    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    target_shape = (num_eval_examples,) + get_dataset_shape(config['dset_name'])

    check_values(x_adv, dset.min_value, dset.max_value)
    check_shape(x_adv, target_shape)

    res = get_res(model_file, x_adv, config['attack_config']['epsilon'],
                  model, dset,
                  num_eval_examples=num_eval_examples,
                  eval_batch_size=eval_batch_size)

    return res


if __name__ == '__main__':
    main(config_path_join('imagenet_topk_config.json'))
