"""
Runner of attack methods. Running this file as a program will
apply the chosen attack to the model specified by the config file and store
the examples in an .npy file.
"""

import json
import math

import numpy as np
import tensorflow as tf

from datasets.dataset import Dataset
from utils.helper_fcts import config_path_join, data_path_join, \
    construct_model, get_model_file, construct_attack


def main(config_file):
    """
    :param config_file:
    :return:
    """
    tf.reset_default_graph()

    with open(config_file) as config_file:
        config = json.load(config_file)

    dset = Dataset(config['dset_name'], config['dset_config'])

    model_file = get_model_file(config)

    with tf.device(config['device']):
        model = construct_model(config['dset_name'])
        attack = construct_attack(model, config, dset)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_adv = []  # adv accumulator

        print('Iterating over {} batches'.format(num_batches))

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch, y_batch = dset.get_eval_data(bstart, bend)

            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            x_adv.append(x_batch_adv)

        print('Storing examples')
        path = data_path_join(config['store_adv_path'])
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(path, x_adv)
        print('Examples stored in {}'.format(path))


if __name__ == '__main__':
    main(config_path_join('imagenet_topk_config.json'))
