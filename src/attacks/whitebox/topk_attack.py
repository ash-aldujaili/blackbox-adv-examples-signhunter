"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils.compute_fcts import lp_step
from utils.compute_fcts import noisy_sign


class LinfTopKAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func,
                 lb, ub, crit='top', retain_p=0.2, is_ns_sign=True, p="inf"):
        """Attack parameter initialization. The attack performs `k` steps of
           size `a` starting from a (`random_start`) point around the point under test
           or the point itself, while always staying within `epsilon` from the initial
           point. The attack randomizes the gradient sign according to `crit` for `1-retain_p`
           fraction of the point's coordinate.
           """
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.crit = crit
        self.retain_p = retain_p
        self.is_ns_sign = is_ns_sign
        self.lb = lb
        self.ub = ub
        self.p = p

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1 - label_mask) * model.pre_softmax, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, self.lb, self.ub)
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            # x = np.add(x,
            #            self.step_size * noisy_sign(grad,
            #                                        crit=self.crit,
            #                                        retain_p=self.retain_p,
            #                                        is_ns_sign=self.is_ns_sign),
            #            casting='unsafe')
            g = noisy_sign(grad,
                           crit=self.crit,
                           retain_p=self.retain_p,
                           is_ns_sign=self.is_ns_sign)

            x = lp_step(x, g, self.step_size, self.p)

            # x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, self.lb, self.ub)  # ensure valid pixel range
        return x
