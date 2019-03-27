"""
Implements the base class for black-box attacks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch
from torch import Tensor as t

from utils.compute_fcts import l2_proj_maker, linf_proj_maker


class BlackBoxAttack(object):
    def __init__(self, max_loss_queries=np.inf,
                 max_crit_queries=np.inf,
                 epsilon=0.5, p='inf', lb=0., ub=1.):
        """
        :param max_loss_queries: max number of calls to model per data point
        :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)
        assert not (np.isinf(max_loss_queries) and np.isinf(max_crit_queries)), "one of the budgets has to be finite!"

        self.epsilon = epsilon
        self.p = p
        self.max_loss_queries = max_loss_queries
        self.max_crit_queries = max_crit_queries
        self.total_loss_queries = 0
        self.total_crit_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.lb = lb
        self.ub = ub
        # the _proj method takes pts and project them into the constraint set:
        # which are
        #  1. epsilon lp-ball around xs
        #  2. valid data pt range [lb, ub]
        # it is meant to be used within `self.run` and `self._suggest`
        self._proj = None
        # a handy flag for _suggest method to denote whether the provided xs is a
        # new batch (i.e. the first iteration within `self.run`)
        self.is_new_batch = False

    def summary(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        self.total_loss_queries = int(self.total_loss_queries)
        self.total_crit_queries = int(self.total_crit_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_loss_queries": self.total_loss_queries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "average_num_loss_queries": "NaN" if self.total_successes == 0 else self.total_loss_queries / self.total_successes,
            "failure_rate": self.total_failures / (self.total_successes + self.total_failures),
            "total_crit_queries": self.total_crit_queries,
            "average_num_crit_queries": "NaN" if self.total_successes == 0 else self.total_crit_queries / self.total_successes,
            "config": self._config()
        }

    def _config(self):
        """
        return the attack's parameter configurations as a dict
        :return:
        """
        raise NotImplementedError

    def _suggest(self, xs_t, loss_fct, metric_fct):
        """
        :param xs_t: batch_size x dim x .. (torch tensor)
        :param loss_fct: function to query (the attacker would like to maximize) (batch_size data pts -> R^{batch_size}
        :param metric_fct: returns the cosine similarity of the suggested step with true gradient
                and the normalized hamming similarity
        :return: suggested xs as a (torch tensor)and the used number of queries per data point
            i.e. a tuple of (batch_size x dim x .. tensor, batch_size array of number queries used)
        Note
        1. this method is assumed to be used only within run, one implication is that
        `self._proj` is redefined every time `self.run` is called which might be used by `self._suggest`
        """
        raise NotImplementedError

    def proj_replace(self, xs_t, sugg_xs_t, dones_mask_t):
        sugg_xs_t = self._proj(sugg_xs_t)
        # replace xs only if not done
        xs_t = sugg_xs_t * (1. - dones_mask_t) + xs_t * dones_mask_t
        return xs_t

    def run(self, xs, loss_fct, early_stop_crit_fct, metric_fct):
        """
        attack with `xs` as data points using the oracle `l` and
        the early stopping criterion `early_stop_crit_fct`
        :param xs: data points to be perturbed adversarially (numpy array)
        :param loss_fct: loss function (m data pts -> R^m)
        :param early_stop_crit_fct: early stop function (m data pts -> {0,1}^m)
                ith entry is 1 if the ith data point is misclassified
        :param metric_fct: function to compute the cosine similarity / hamming similarity of the suggested step with the true
                        gradient to be used within `_suggest`
        :return: a dict of logs whose length is the number of iterations
        """
        # convert to tensor
        xs_t = t(xs)
        batch_size = xs.shape[0]
        num_axes = len(xs.shape[1:])
        num_loss_queries = np.zeros(batch_size)
        num_crit_queries = np.zeros(batch_size)

        dones_mask = early_stop_crit_fct(xs)
        correct_classified_mask = np.logical_not(dones_mask)

        # list of logs to be returned
        logs_dict = {
            'total_loss': [],
            'total_cos_sim': [],
            'total_ham_sim': [],
            'total_successes': [],
            'total_failures': [],
            'iteration': [],
            'total_loss_queries': [],
            'total_crit_queries': [],
            'num_loss_queries_per_iteration': [],
            'num_crit_queries_per_iteration': []
        }

        # ignore this batch of xs if all are misclassified
        if sum(correct_classified_mask) == 0:
            return logs_dict

        # init losses and cosine similarity for performance tracking
        losses = np.zeros(batch_size)
        cos_sims = np.zeros(batch_size)
        ham_sims = np.zeros(batch_size)

        # make a projector into xs lp-ball and within valid pixel range
        if self.p == '2':
            _proj = l2_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: ch.clamp(_proj(_), self.lb, self.ub)
        elif self.p == 'inf':
            _proj = linf_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: ch.clamp(_proj(_), self.lb, self.ub)
        else:
            raise Exception('Undefined l-p!')

        # iterate till model evasion or budget exhaustion
        # to inform self._suggest this is  a new batch
        self.is_new_batch = True
        its = 0
        while True:
            if np.any(num_loss_queries >= self.max_loss_queries):
                print("#loss queries exceeded budget, exiting")
                break
            if np.any(num_crit_queries >= self.max_crit_queries):
                print("#crit_queries exceeded budget, exiting")
                break
            if np.all(dones_mask):
                print("all data pts are misclassified, exiting")
                break
            # propose new perturbations
            sugg_xs_t, num_loss_queries_per_step, cur_cos_sims, cur_ham_sims = self._suggest(xs_t, loss_fct, metric_fct)
            # project around xs and within pixel range and
            # replace xs only if not done
            xs_t = self.proj_replace(xs_t, sugg_xs_t, t(dones_mask.reshape(-1, *[1] * num_axes).astype(np.float32)))
            # print(np.linalg.norm(xs_t.view(xs_t.shape[0], -1).cpu(), axis=1))
            # update number of queries (note this is done before updating dones_mask)
            num_loss_queries += num_loss_queries_per_step * (1. - dones_mask)
            num_crit_queries += (1. - dones_mask)
            # update total loss and total cos_sim (these two values are used for performance monitoring)
            cos_sims = cur_cos_sims * (1. - dones_mask) + cos_sims * dones_mask
            ham_sims = cur_ham_sims * (1. - dones_mask) + ham_sims * dones_mask
            losses = loss_fct(xs_t.cpu().numpy()) * (1. - dones_mask) + losses * dones_mask
            # update dones mask
            dones_mask = np.logical_or(dones_mask, early_stop_crit_fct(xs_t.cpu().numpy()))
            its += 1
            self.is_new_batch = False

            # update logs
            logs_dict['total_loss'].append(sum(losses))
            logs_dict['total_cos_sim'].append(sum(cos_sims))
            logs_dict['total_ham_sim'].append(sum(ham_sims))
            logs_dict['total_successes'].append(sum(dones_mask * correct_classified_mask))
            logs_dict['total_failures'].append(sum(np.logical_not(dones_mask) * correct_classified_mask))
            logs_dict['iteration'].append(its)
            # assuming all data pts consume the same number of queries per step
            logs_dict['num_loss_queries_per_iteration'].append(num_loss_queries_per_step[0])
            logs_dict['num_crit_queries_per_iteration'].append(1)
            logs_dict['total_loss_queries'].append(sum(num_loss_queries * dones_mask * correct_classified_mask))
            logs_dict['total_crit_queries'].append(sum(num_crit_queries * dones_mask * correct_classified_mask))

        success_mask = dones_mask * correct_classified_mask
        self.total_loss_queries += (num_loss_queries * success_mask).sum()
        self.total_crit_queries += (num_crit_queries * success_mask).sum()
        self.total_successes += success_mask.sum()
        self.total_failures += (np.logical_not(dones_mask) * correct_classified_mask).sum()

        # set self._proj to None to ensure it is intended use
        self._proj = None

        return logs_dict
