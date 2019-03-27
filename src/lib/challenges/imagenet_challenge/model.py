"""
This model is adapted from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3_test.py
"""

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets.inception import inception_v3, inception_v3_arg_scope


class Model(object):
    """
    Inception_v3 Constructor.
    Args:
        mode: 'train' or 'eval'
    """

    def __init__(self, mode='eval'):
        if mode == 'train':
            raise Exception("The current implementation was not intended for training,"
                            "use it at your own risk!")
        self.mode = mode
        self._build_model()

    def _build_model(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        self.y_input = tf.placeholder(tf.int64, shape=[None])

        with slim.arg_scope(inception_v3_arg_scope()):
            logits, _ = inception_v3(self.x_input, num_classes=1001,
                                     is_training=self.mode == 'train')

        self.pre_softmax = logits
        self.y_pred = tf.argmax(self.pre_softmax, 1)

        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax
        )
        self.xent = tf.reduce_sum(self.y_xent)

        self.correct_prediction = tf.equal(self.y_pred, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


if __name__ == '__main__':
    print("I am just a module to be imported by others, testing here")
    from datasets.imagenet_val_set import ImagenetValidData

    data = ImagenetValidData()
    images, labels = data.get_eval_data(0, 50)

    model = Model()
    model_file = '/home/ash/Downloads/inception_v3.ckpt'

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_file)
        logits_out, acc = sess.run([model.pre_softmax, model.accuracy], feed_dict={
            model.x_input: images,
            model.y_input: labels
        })

    print(acc)
