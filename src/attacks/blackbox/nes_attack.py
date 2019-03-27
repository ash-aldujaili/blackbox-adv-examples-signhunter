"""
Implements NES attacks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch
from torch import Tensor as t

from attacks.blackbox.black_box_attack import BlackBoxAttack
from utils.compute_fcts import lp_step


class NESAttack(BlackBoxAttack):
    """
    NES Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lr, q, lb, ub):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param q: number of noise samples per NES step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_crit_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub)
        self.q = q
        self.fd_eta = fd_eta
        self.lr = lr

    def _suggest(self, xs_t, loss_fct, metric_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = ch.zeros_like(xs_t)
        for i in range(self.q):
            exp_noise = ch.randn_like(xs_t) / (dim ** 0.5)
            fxs_t = xs_t + self.fd_eta * exp_noise
            bxs_t = xs_t - self.fd_eta * exp_noise
            est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / (2. * self.fd_eta)
            gs_t += t(est_deriv.reshape(-1, *[1] * num_axes)) * exp_noise
        # compute the cosine similarity
        cos_sims, ham_sims = metric_fct(xs_t.cpu().numpy(), gs_t.view(_shape[0], -1).cpu().numpy())
        # perform the step
        new_xs = lp_step(xs_t, gs_t, self.lr, self.p)
        # print(np.linalg.norm(new_xs.numpy().reshape(_shape[0], -1)), np.linalg.norm(xs_t.numpy().reshape(_shape[0], -1), axis=1))
        return new_xs, 2 * self.q * np.ones(_shape[0]), cos_sims, ham_sims

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_crit_queries": "inf" if np.isinf(self.max_crit_queries) else self.max_crit_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "lr": self.lr,
            "q": self.q,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }
