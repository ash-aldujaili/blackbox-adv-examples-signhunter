# experimentation in signhunter for continuous optmization
import itertools

import cma
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
FONTSIZE = 12.5
COLORS = itertools.cycle(["#a80000", "#00a8a8", "#5400a8", "#54a800",
                          '#dc00dc', '#dc6e00', '#00dc00', '#006edc'])
MARKERS = itertools.cycle(['.', '+', 'o'])  # , '*', 'v', '>', '<', 'd'])
plt.rcParams["mathtext.fontset"] = "cm"


def zscore(confidence_level=0.95):
    """
    return the zscore
    """
    return st.norm.ppf((1 + confidence_level) / 2)


def sign(x):
    return np.sign(np.sign(x - 0.5))


def cos_sim(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def plot_curve_with_conf(logs, label):
    ys = np.array(logs)
    ms = np.mean(logs, axis=0)
    stds = np.std(logs, axis=0)
    ns = ys.shape[0]
    xs = np.arange(ys.shape[1])

    # marg_err = zscore(0.50) * stds / np.sqrt(ns)
    marg_err = stds
    plt.plot(xs, ms, label=r'\texttt{{{}}}'.format(label), marker=MARKERS.__next__(), markevery=ys.shape[1] // 3)
    if ns > 1:
        plt.fill_between(xs, ms - marg_err,
                         ms + marg_err,
                         alpha=0.2)


def hamming_dist(a, b):
    return sum([_a != _b for _a, _b in zip(a, b)])


def hamming_dist_to_sdf(x, g):
    """
    x : current point
    g : estimated gradient
    """
    return hamming_dist(np.sign(g), np.sign(df(x)))


def sign_hunter(score, max_evals, s):
    """
    score function to assess the suggested sign
    max_evals: query budgt
    s: proposed sign from previous iterations (could be used if the function is well-behaved)
    """
    s = np.ones(n)
    best_val = score(s)
    num_evals = 1
    for h in range(np.ceil(np.log2(n) + 1).astype(int)):
        for i in range(2 ** h):
            chunk_len = np.ceil(n / 2 ** h).astype(int)
            istart = i * chunk_len
            iend = min(n, (i + 1) * chunk_len)
            s[istart: iend] *= -1
            # print("Flipping {}-{}:".format(istart, iend))
            # print(s)
            _val = score(s)
            num_evals += 1
            if best_val < _val:
                best_val = _val
            else:
                s[istart: iend] *= -1
                # print("revert")
            if num_evals >= max_evals:
                return s.copy()
            if iend == n:
                break
    # in case max_evals was more than SignHunter needs
    print("SignHunter flipped all bits")
    return s.copy()


class SignHunter(object):
    def __init__(self, n):
        self.n = n
        self.reset()
        self.s = np.ones(self.n)

    def reset(self):
        self.i = 0
        self.h = 0
        self.is_done = False
        self.best_val = - np.float("inf")

    def step(self, score, num_steps=1):
        if self.i == 0 and self.h == 0:
            self.best_val = score(self.s)
        for _ in range(num_steps):
            chunk_len = np.ceil(n / (2. ** self.h)).astype(int)
            istart = self.i * chunk_len
            iend = min(self.n, istart + chunk_len)
            # print(istart, iend, self.s)
            self.s[istart: iend] *= -1
            val = score(self.s)
            # print(val, self.best_val)
            if val >= self.best_val:
                self.best_val = val
            else:
                self.s[istart: iend] *= -1
            self.i += 1
            if iend == self.n:
                self.i = 0
                self.h += 1
                if self.h == np.ceil(np.log2(self.n)) - 1:
                    # print( 'it is done')
                    self.is_done = True
                    return self.s, _ + 1
        return self.s, num_steps


def cma_optmize(f, num_evals, x):
    es = cma.CMAEvolutionStrategy(x, 0.5)
    num_iters = num_evals // es.popsize
    # print(es.popsize)
    ham_dists = []
    y_vals = [f(x)]
    for _ in range(num_iters):
        solutions = es.ask()
        es.tell(solutions, [f(x) for x in solutions])
        y_vals.append(min(y_vals[0], es.best.f))
        ham_dists.append(hamming_dist_to_sdf(es.mean_old, es.mean - es.mean_old))
    return y_vals, ham_dists


def sign_optimize(f, num_iters, num_samples, x, mu, eta=0.001):
    sh = SignHunter(len(x))
    y_vals = [f(x)]
    ham_dists = []
    num_evals = 0
    prev_g = None
    while num_evals < num_iters * num_samples:
        score = lambda _: (f(x + eta * _) - f(x)) / eta
        g, cur_num_evals = sh.step(score, num_steps=num_samples)
        if sh.is_done: sh.reset()
        num_evals += cur_num_evals
        ham_dists.append(hamming_dist_to_sdf(x, g))
        # if prev_g is not None:
        #    mu = mu * (g == prev_g) + mu * 0.1 * (g != prev_g)
        x = x - mu * g
        prev_g = g
        y_vals.append(f(x))
    return y_vals, ham_dists


def optimize(f, num_evals, n, alg='signhunter', mu=0.0001, eta=0.001, is_ones_init=False):
    one_norm = np.sqrt(n)
    num_samples = 5  # 4 + int(3 * np.log(n)) # if you want to compare set the number of samples similar to cmaes
    num_iters = num_evals // num_samples
    x = np.ones(n) if is_ones_init else np.random.rand() * np.ones(n)
    if alg == 'cma': return cma_optmize(f, num_evals, x)
    # if alg == 'signhunter': return sign_optimize(f, num_iters, num_samples, x, mu, eta)
    y_vals = [f(x)]
    ham_dists = []
    g = np.ones(n)
    for _ in range(num_iters):
        prev_g = g
        if alg == 'signhunter':
            score = lambda _: f(x + eta * _)  # / one_norm)
            g = sign_hunter(score, num_samples, g)
            # mu = mu  * (g == prev_g) +  mu * 0.33 * (g != prev_g)
            # print(mu)
        else:
            g = np.zeros(n)
            for _ in range(num_samples):
                _v = np.random.randn(n) / (n ** 0.5)
                est_deriv = (f(x + eta * _v) - f(x)) / eta
                g += _v * est_deriv
            if alg == 'zo':
                g = np.sign(g)
                # g = sign(g)
        # print("est g: {}".format(g))
        # print("df:{}".format(df(x)))
        # ham_dists.append(hamming_dist_to_sdf(x, g))
        x = x - mu * g

        # print("ham_dists:{}".format(hamming_dist(g, prev_g)))
        y_vals.append(f(x))
    return y_vals, ham_dists


if __name__ == '__main__':
    # Note CMA-ES is included here for comparison in small dimensions
    # it does not scale to higher dimensions and therefore it was not
    # considered in the paper's discussion. 
    is_nes = True
    is_zo = True
    is_cma = False
    is_show = False
    np.random.seed(1)
    num_runs = 30
    # problem setup
    for is_ones_init in [True]:  # [True, False]:
        for n in [10, 100, 1000, 100000]:
            plt.clf()
            ys_nes = []
            ys_zo = []
            ys_sign = []
            ys_cma = []
            num_evals = 3000
            sign_lr = 0.01
            nes_lr = 0.01
            zo_lr = 0.01
            for _ in range(num_runs):
                print(n, _)
                np.random.seed(_)
                # instantiate a new obj function
                x_opt = np.random.rand(n)


                def f(x):
                    return (x - x_opt).T.dot((x - x_opt))


                # for debugging
                def df(x):
                    return 2 * (x - x_opt)


                np.random.seed(_)
                y_sign, h_sign = optimize(f, num_evals, n, alg='signhunter', mu=sign_lr, is_ones_init=is_ones_init)

                if is_nes:
                    np.random.seed(_)  # to make sure all initialized to the same point
                    y_nes, h_nes = optimize(f, num_evals, n, alg='nes', mu=nes_lr, is_ones_init=is_ones_init)
                    ys_nes.append(y_nes)

                ys_sign.append(y_sign)
                if is_zo:
                    np.random.seed(_)
                    y_zo, h_zo = optimize(f, num_evals, n, alg='zo', mu=zo_lr, is_ones_init=is_ones_init)
                    ys_zo.append(y_zo)
                if is_cma:
                    np.random.seed(_)
                    y_cma, h_cma = optimize(f, num_evals, n, alg='cma', is_ones_init=is_ones_init)
                    ys_cma.append(y_cma)

            plot_curve_with_conf(ys_sign, label='SignHunter')
            if is_nes: plot_curve_with_conf(ys_nes, label='NES')
            if is_zo: plot_curve_with_conf(ys_zo, label='ZO-SignSGD')
            if is_cma: plot_curve_with_conf(ys_cma, label='CMA-ES')
            plt.xlabel("iteration")
            plt.ylabel("loss value")
            plt.legend()
            if is_show:
                plt.show()
            else:
                plt.savefig("plt_n-{}_fevals-{}_one-init-{}.pdf".format(n, num_evals, is_ones_init))

            # plt.plot(h_nes, label='NES')
            # plt.plot(h_sign, label='SignHunter')
            # plt.plot(h_zo, label='ZO')
            # plt.plot(h_cma, label='CMA-ES')
            # plt.title("Norm. Hamming Dist.")
            # plt.legend()
            # plt.show()
