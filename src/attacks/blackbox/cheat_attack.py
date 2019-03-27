"""
Implement a random attacker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch import Tensor as t

from attacks.blackbox.black_box_attack import BlackBoxAttack
from utils.compute_fcts import lp_step


class CheatAttack(BlackBoxAttack):
    """
    Attacker with access to gradient orcale (cheater)
    This attacker is hypothetical and meant to compare metrics with other
    """

    def __init__(self,
                 max_loss_queries,
                 epsilon,
                 lr,
                 p,
                 lb,
                 ub):
        """
        :param max_crit_queries: max number of calls to early stopping criterion  per data point
         (the model does not do any call to the loss oracle)
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        super().__init__(max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub)

        self.lr = lr
        # Attention: this should be set by the user prior to calling `self.run()` method
        self.grad_fct = None

    def set_grad_fct(self, grad_fct):
        """

        :param grad_fct:
        :return:
        """
        self.grad_fct = grad_fct

    def _suggest(self, xs_t, loss_fct, metric_fct):
        _shape = list(xs_t.shape)
        _gs = self.grad_fct(xs_t.cpu().numpy())
        # compute the cosine similarity
        cos_sims, ham_sims = metric_fct(xs_t.cpu().numpy(), _gs)
        # perform the step
        new_xs = lp_step(xs_t, t(_gs.reshape(_shape)), self.lr, self.p)
        return new_xs, 2 * np.ones(_shape[0]), cos_sims, ham_sims

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_crit_queries": self.max_crit_queries,
            "max_loss_queries": self.max_loss_queries,
            "attack_name": self.__class__.__name__,
            "lr": self.lr
        }
