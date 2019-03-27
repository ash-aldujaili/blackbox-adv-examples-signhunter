"""
A wrapper for datasets
    mnist, cifar10, imagenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import datasets.cifar10_input as cifar10_input
from datasets.imagenet_val_set import ImagenetValidData
from utils.helper_fcts import data_path_join


class Dataset(object):
    def __init__(self, name, config):
        """
        :param name: dataset name
        :param config: dictionary whose keys are dependent on the dataset created
         (see source code below)
        """
        assert name in ['mnist', 'cifar10', 'cifar10aug', 'imagenet'], "Invalid dataset"
        self.name = name

        if self.name == 'cifar10':
            data_path = data_path_join('cifar10_data')
            self.data = cifar10_input.CIFAR10Data(data_path)
        elif self.name == 'cifar10aug':
            data_path = data_path_join('cifar10_data')
            raw_cifar = cifar10_input.CIFAR10Data(data_path)
            sess = config['sess']
            model = config['model']
            self.data = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
        elif self.name == 'mnist':
            self.data = input_data.read_data_sets(data_path_join('mnist_data'), one_hot=False)
        elif self.name == 'imagenet':
            self.data = ImagenetValidData(data_dir=config['data_dir'])

    def get_next_train_batch(self, batch_size):
        """
        Returns a tuple of (x_batch, y_batch)
        """
        if self.name in ['cifar10', 'cifar10aug']:
            return self.data.train_data.get_next_batch(batch_size, multiple_passes=True)
        elif self.name == 'mnist':
            return self.data.train.next_batch(batch_size)
        elif self.name == 'imagenet':
            raise Exception(
                'No training data for imagenet is needed (provided), the models are assumed to be pretrained!')

    def get_eval_data(self, bstart, bend):
        """
        :param bstart: batch start index
        :param bend: batch end index
        """
        if self.name in ['cifar10', 'cifar10aug']:
            return self.data.eval_data.xs[bstart:bend, :], \
                   self.data.eval_data.ys[bstart:bend]
        elif self.name == 'mnist':
            return self.data.test.images[bstart:bend, :], \
                   self.data.test.labels[bstart:bend]
        elif self.name == 'imagenet':
            return self.data.get_eval_data(bstart, bend)

    @property
    def min_value(self):
        if self.name in ['cifar10', 'cifar10aug', 'mnist', 'imagenet']:
            return 0

    @property
    def max_value(self):
        if self.name in ['cifar10', 'cifar10aug']:
            return 255
        if self.name in ['mnist', 'imagenet']:
            return 1
