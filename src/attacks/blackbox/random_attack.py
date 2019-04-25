"""
Implements the binary based attack
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch

from attacks.blackbox.black_box_attack import BlackBoxAttack
from utils.compute_fcts import lp_step, sign


class RandAttack(BlackBoxAttack):
    """
    Random Attack: randomly flip signs till at the boundary
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


    def _suggest(self, xs_t, loss_fct, metric_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        if self.is_new_batch:
            self.xo_t = xs_t.clone()
        sgn_t = sign(ch.rand(_shape[0], dim) - 0.5)


        # compute the cosine similarity
        cos_sims, ham_sims = metric_fct(self.xo_t.cpu().numpy(), sgn_t.cpu().numpy())
        # perform the step
        new_xs = lp_step(self.xo_t, sgn_t.view(_shape), self.epsilon, self.p)
        return new_xs, np.ones(_shape[0]) , cos_sims, ham_sims

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
