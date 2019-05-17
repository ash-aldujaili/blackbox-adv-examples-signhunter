"""
Implements SignHunter
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch

from attacks.blackbox.black_box_attack import BlackBoxAttack
from utils.compute_fcts import lp_step, sign


class SignAttack(BlackBoxAttack):
    """
    SignHunter
    """

    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lb, ub):
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
        self.fd_eta = fd_eta
        self.best_est_deriv = None
        self.xo_t = None
        self.sgn_t = None
        self.h = 0
        self.i = 0

    def _suggest(self, xs_t, loss_fct, metric_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        # additional queries at the start
        add_queries = 0
        if self.is_new_batch:
            self.xo_t = xs_t.clone()
            self.h = 0
            self.i = 0
        if self.i == 0 and self.h == 0:
            self.sgn_t = sign(ch.ones(_shape[0], dim))
            fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            bxs_t = self.xo_t  
            est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon
            self.best_est_deriv = est_deriv
            add_queries = 3  # because of bxs_t and the 2 evaluations in the i=0, h=0, case.
        chunk_len = np.ceil(dim / (2 ** self.h)).astype(int)
        istart = self.i * chunk_len
        iend = min(dim, (self.i + 1) * chunk_len)
        self.sgn_t[:, istart:iend] *= - 1.
        fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        bxs_t = self.xo_t
        est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon
        self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], istart: iend] *= -1.
        self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (
                est_deriv < self.best_est_deriv) * self.best_est_deriv
        # compute the cosine similarity
        cos_sims, ham_sims = metric_fct(self.xo_t.cpu().numpy(), self.sgn_t.cpu().numpy())
        # perform the step
        new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        # update i and h for next iteration
        self.i += 1
        if self.i == 2 ** self.h or iend == dim:
            self.h += 1
            self.i = 0
            # if h is exhausted, set xo_t to be xs_t
            if self.h == np.ceil(np.log2(dim)).astype(int) + 1:
                self.xo_t = xs_t.clone()
                self.h = 0
                print("new change")
        return new_xs, np.ones(_shape[0]) + add_queries, cos_sims, ham_sims

    def get_gs(self):
        """
        return the current estimated of the gradient sign
        :return:
        """
        return self.sgn_t

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_crit_queries": "inf" if np.isinf(self.max_crit_queries) else self.max_crit_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }
