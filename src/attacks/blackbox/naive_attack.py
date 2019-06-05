"""
Implements the binary based attack with sequential bit flipping
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch

from attacks.blackbox.black_box_attack import BlackBoxAttack
from utils.compute_fcts import lp_step, sign


class NaiveAttack(BlackBoxAttack):
    """
    Naive Attack: sequentially flip signs till at the boundary
    """

    def __init__(self, max_loss_queries, epsilon, p, lb, ub):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_crit_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub)
        self.xo_t = None
        self.sgn_t = None
        self.best_est_deriv = None
        self.i = 0

    def _suggest(self, xs_t, loss_fct, metric_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        add_queries = 0
        if self.is_new_batch:
            self.xo_t = xs_t.clone()
            self.i = 0
        if self.i == 0:
            self.sgn_t = sign(ch.ones(_shape[0], dim))
            fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            bxs_t = self.xo_t
            est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon
            self.best_est_deriv = est_deriv
            add_queries = 2
        self.sgn_t[:, self.i] *= -1
        fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        bxs_t = self.xo_t
        est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon
        self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], self.i] *= -1.
        self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (
                est_deriv < self.best_est_deriv) * self.best_est_deriv
        # compute the cosine similarity
        cos_sims, ham_sims = metric_fct(self.xo_t.cpu().numpy(), self.sgn_t.cpu().numpy())
        # perform the step
        new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        self.i += 1
        if self.i == dim:
            self.xo_t = new_xs.clone()
            self.i = 0
        return new_xs, np.ones(_shape[0]) + add_queries, cos_sims, ham_sims

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_crit_queries": "inf" if np.isinf(self.max_crit_queries) else self.max_crit_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "attack_name": self.__class__.__name__
        }
