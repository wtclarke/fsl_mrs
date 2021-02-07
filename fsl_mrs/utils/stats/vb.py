#!/usr/bin/env python

# core.py - main MRS class definition
#
# Authors: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT


import numpy as np
from scipy.special import digamma
from scipy.special import loggamma

from scipy.stats import norm, gamma
import matplotlib.pyplot as plt


class OptimizeResult(dict):
    """
    'borrowed' from scipy v1.4.1 source code
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class NonlinVB(object):
    """
    Usage:
    model = NonlinVB()
    model.forward = forward_model
    results = model.fit(y,x0)

    forward_model function must be defined as having the free parameters
    as its first argument

    if parameters are positive this must be dealt with in the forward model
    by exponentiating them

    """

    def __init__(self, forward=None):
        self.forward = forward
        self.priors = None
        self.jac = None
        self._lam = 1E-4

    def set_priors(self, M0, P0, c0, s0):
        """
        theta = N(M0,P0^-1)
        Phi   = G(shape=c0,scale=s0)
        """
        self.priors = [M0, P0, c0, s0]

    def __set_priors(self, x0):
        """
        choose priors
        """
        M0 = np.zeros_like(x0)
        var0 = np.ones_like(x0) * 1e5
        var0[var0 == 0] = 1e5
        C0 = np.diag(var0)
        P0 = np.linalg.inv(C0)
        beta_mean0 = 1e-2
        beta_var0 = 1e-2
        # c=shape, s=scale parameters for Gamma distribution
        s0 = beta_var0 / beta_mean0
        c0 = beta_mean0**2 / beta_var0

        self.priors = [M0, P0, c0, s0]

        return M0, P0, c0, s0

    def __calc_jacobian(self, x, args=None):
        """
        Numerical differentiation to calculate Jacobian matrix
        of partial derivatives of model prediction with respect to
        parameters
        """
        # if jacobian provided use it
        if self.jac is not None:
            return self.jac(x, *args)

        J = None
        for param_idx, param_value in enumerate(x):
            xL = np.array(x)
            xU = np.array(x)
            delta = param_value * 1e-5
            if delta < 0:
                delta = -delta
            if delta < 1e-10:
                delta = 1e-10
            xU[param_idx] += delta
            xL[param_idx] -= delta

            yU = self.forward(xU, *args)
            yL = self.forward(xL, *args)
            if J is None:
                J = np.zeros([len(yU), len(x)], dtype=np.float32)
            J[:, param_idx] = (yU - yL) / (2 * delta)
        return J

    def calc_jacobian(self, x, args=None):
        return self.__calc_jacobian(x, args)

    def __lam_up(self):
        MAXIMUM = 1e3
        self._lam = np.minimum(10 * self._lam, MAXIMUM)

    def __lam_down(self):
        MINIMUM = 1e-10
        self._lam = np.maximum(.1 * self._lam, MINIMUM)

    def __update_model_params(self, k, M, P, s, c, J):
        """
        Update model parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        k = data - prediction
        M = means (prior = M0)
        P = precision (prior=P0)
        s = noise shape (prior = s0)
        c = noise scale (prior = c0)
        J = Jacobian
        """
        M0, P0, _, _ = self.priors

        P_new = s * c * np.dot(J.transpose(), J) + P0
        C_new = np.linalg.inv(P_new)

        # L-M update for the mean
        M_old = M.copy()
        delta = s * c * np.dot(J.transpose(), (k + np.dot(J, M_old))) + np.dot(P0, M0) - np.dot(P_new, M_old)

        # use current lambda to calc F
        # if F goes up, accept step and decrease lambda
        # if F goes down reject step and increase lambda
        F_old = self.__calc_free_energy(len(k), C_new, M_old, P_new, s, c)
        mat = np.linalg.inv((P_new + self._lam * np.diag(P)))
        M_test = M_old + np.dot(mat, delta)
        F_new = self.__calc_free_energy(len(k), C_new, M_test, P_new, s, c)
        if F_new < F_old:
            M_new = M_old
            self.__lam_up()
        else:
            M_new = M_test
            self.__lam_down()

        return M_new, P_new

    def __update_noise(self, N, k, P, J):
        """
        Update noise parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        k = data - prediction
        P = precision (prior=P0)
        J = Jacobian
        """
        _, _, c0, s0 = self.priors
        C = np.linalg.inv(P)
        c_new = N / 2 + c0
        s_new = 1 / (1 / s0 + 1 / 2 * np.dot(k.transpose(), k) + 1 / 2 * np.trace(np.dot(C, np.dot(J.transpose(), J))))
        return c_new, s_new

    def __calc_free_energy(self, N, C, M, P, s, c):
        """
        free energy is maximized in VB so this it a good thing
        to monitor especially with nonlinear models
        """
        M0, P0, c0, s0 = self.priors
        F = (N / 2 + c0 - c) * (np.log(s) + digamma(c)) + s * c / 2 * (1 / s - 1 / s0)
        F += c * np.log(s) + loggamma(c)
        _, logdetP = np.linalg.slogdet(P)
        F += logdetP / 2
        F -= 1 / 2 * (np.dot(np.dot((M - M0).transpose(), P0), M - M0) + np.trace(np.dot(C, P0)))
        return F

    def fit(self, y, x0, niter=20, monitor=True, verbose=False, args=None):
        """
        Run VB updates
        Parameters
        ----------
        y : array-like
            data to be fitted
        x0 : array-like
            initial guess
        niter : int
             number of VB updates
        monitor : bool
             monitor Free Energy
        verbose : bool
             print information while fitting
        args : list
             arguments to be passed to the forward model
        """
        if self.forward is None:
            raise(Exception('You must define a forward model before fitting'))

        # admin
        N = len(y)
        x0 = np.asarray(x0)

        # Set priors
        if self.priors is None:
            M0, P0, c0, s0 = self.__set_priors(x0)
        else:
            M0, P0, c0, s0 = self.priors

        # Initial posterior parameters
        M = x0
        C = np.eye(len(x0))
        P = np.linalg.inv(C)
        c = c0
        s = s0
        if verbose:
            print(f'Iteration 0: x={M}, noise={c*s}')

        if monitor:
            Ms = [M]
            Ps = [P]
            ss = [s]
            cs = [c]
            FE = [self.__calc_free_energy(N, C, M, P, s, c)]

        # Update model and noise parameters iteratively
        for idx in range(niter):
            # calc error
            k = y - self.forward(M, *args)
            # calc Jacobian
            J = self.__calc_jacobian(M, args)
            # update model params
            M, P = self.__update_model_params(k, M, P, s, c, J)
            # update noise params
            c, s = self.__update_noise(N, k, P, J)
            if verbose:
                print(f'Iteration {idx+1}: x={M}, noise={c*s}')
            if monitor:
                FE.append(self.__calc_free_energy(N, C, M, P, s, c))
                Ms.append(M)
                Ps.append(P)
                ss.append(s)
                cs.append(c)

        # Pick maximum F
        FE = np.asarray(FE)
        idx = np.argmax(FE)
        M = Ms[idx]
        P = Ps[idx]
        s = ss[idx]
        c = cs[idx]

        # Collect results
        results = self.collect_results([M, P, s, c])
        if monitor:
            results.xs = np.asarray(Ms)
            results.FE = np.asarray(FE)
            results.J = self.__calc_jacobian(M, args)
            results.residuals = y - self.forward(M, *args)

        # keep copy of model
        results.forward = self.forward

        return results

    def collect_results(self, variables):
        """
        variables = Mean,PrecisionMatrix,scale,shape
        """
        M, P, s, c = variables
        mu = M
        var = np.diag(np.linalg.inv(P))
        results = OptimizeResult(x=mu,
                                 noise_params=[s, c],
                                 cov=np.linalg.inv(P),
                                 var=var,
                                 noise_std=1. / np.sqrt(c * s),
                                 phi=s * c)
        return results


def plot_posterior(means, cov, labels=None, samples=None, actual=None):
    """
    helper function for plotting posterior distribution

    Parameters
    ----------
    means : array like
    cov   : matrix
    labels : list
    samples : 2D (samples x params)
              as ouput by MH
    actual : array like
             true parameter values if known

    Returns
    -------
    matplotlib figure

    """
    fig = plt.figure(figsize=(10, 10))
    n = means.size
    nbins = 50
    k = 1
    for i in range(n):
        for j in range(n):
            if i == j:
                x = np.linspace(means[i] - 5 * np.sqrt(cov[i, i]),
                                means[i] + 5 * np.sqrt(cov[i, i]), nbins)
                y = norm.pdf(x, means[i], np.sqrt(cov[i, i]))

                plt.subplot(n, n, k)
                plt.plot(x, y)
                if samples is not None:
                    plt.hist(samples[:, i], histtype='step', density=True)
                if labels is not None:
                    plt.title(labels[i])
                if actual is not None:
                    plt.axvline(x=actual[i], c='r')

            else:
                m = np.asarray([means[i], means[j]])
                v = np.asarray([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])
                xi = np.linspace(means[i] - 5 * np.sqrt(cov[i, i]),
                                 means[i] + 5 * np.sqrt(cov[i, i]), nbins)
                xj = np.linspace(means[j] - 5 * np.sqrt(cov[j, j]),
                                 means[j] + 5 * np.sqrt(cov[j, j]), nbins)
                x = np.asarray([(a, b) for a in xi for b in xj])
                x = x - m
                h = np.sum(-.5 * (x * (x @ np.linalg.inv(v).T)), axis=1)

                h = np.exp(h - h.max())
                h = np.reshape(h, (nbins, nbins))
                plt.subplot(n, n, k)

                plt.contour(xi, xj, h)

                if samples is not None:
                    plt.plot(samples[:, i], samples[:, j], 'k.', alpha=.1)
            k = k + 1

    plt.show()
    return fig


def plot_noise_posterior(noise_params, true_sigma=None):
    fig = plt.figure()
    s, c = noise_params

    beta_mean = c * s
    beta_std = np.sqrt(c) * s
    x = np.linspace(beta_mean - 5 * beta_std, beta_mean + 5 * beta_std, 100)
    y = gamma.pdf(x, c, 0, s)

    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.axvline(x=s * c, c='r')
    if true_sigma is not None:
        plt.axvline(x=1 / true_sigma**2, c='g')

    plt.xlabel('precision')

    plt.subplot(1, 2, 2)
    plt.plot(1 / np.sqrt(x), y)
    plt.axvline(x=1 / np.sqrt(s * c), c='r')
    if true_sigma is not None:
        plt.axvline(x=true_sigma, c='g')
    plt.xlabel('sigma')
    plt.show()

    return fig


# Demo
def run_example(do_plot=True):
    """
    Fit M0,R1,R2
    """
    TEs = np.array([10, 40, 50, 60, 80])  # TE values in ms
    TRs = np.array([.8, 1, 1.5, 2])     # TR in seconds (I know this is bad)

    # All combinations of TEs/TRs
    comb = np.array([(x, y) for x in TEs for y in TRs])
    TEs, TRs = comb[:, 0], comb[:, 1]
    args = [TEs, TRs]

    # function for our model
    def forward(p, TEs, TRs):
        M0, R1, R2 = p
        return M0 * np.exp(-R2 * TEs) * (1 - np.exp(-R1 * TRs))

    # simulate data using model
    true_p = [100, 1 / .8, 1 / 50]   # M0, R1=1/T1,R2=1/T2
    data = forward(true_p, *args)
    snr = 50
    noise_std = true_p[0] / snr
    noise = np.random.randn(data.size) * noise_std
    data = data + noise

    model = NonlinVB()
    model.forward = forward
    res = model.fit(data, x0=[200, 1 / 1, 1 / 70], args=args)

    # visualise results
    if do_plot:
        _ = plot_posterior(res.x, res.cov, actual=true_p, labels=['M0', 'R1', 'R2'])
        _ = plot_noise_posterior(res.noise_params, noise_std)

    return res
