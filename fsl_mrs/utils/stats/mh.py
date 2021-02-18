#!/usr/bin/env python

# mh.py - main MH class and functionality definition
#
# Authors: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#          Michiel Cottaar <michiel.cottaar@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT


import numpy as np
from . import dist


def sse(y, pred):
    """
    SumSquaredError

    Parameters
    ----------

    y : array-like
      Data
    pred : array-like
      Prediction

    Returns
    -------
    array


    """
    return np.sum((y - pred)**2)


def mh_example(do_plot=True, verbose=False):
    """
    Example use of MH for nonlinear fitting

    Model:
      Generative likelihood
        Y = a*exp(-b*X) + N(0,sig^2)
      Priors:
        a ~ N(0,1)
        b ~ Gamma(1,1)

      Where:
        X: array of regressors
        a,b: parameters


    Returns
    -------
    array (nsamples x nparams)

    """
    np.random.seed(123)

    x = np.linspace(0, 10, 100)
    p = [1, 2, 0.05]
    y = p[0] * np.exp(-p[1] * x)
    y_noise = y + p[2] * np.random.randn(y.size)

    # model
    # a ~ N(0,10^2)
    # b ~ N(0,10^2)
    # sig ~ Gamma(1,1)
    # y ~ a*exp(-b*x) + N(0,sig^2)

    # loglik
    def forward(p):
        return p[0] * np.exp(-p[1] * x)

    def loglik(p):
        pred = forward(p)
        return sse(y_noise, pred) / 2 / p[2]**2

    # logpr
    def logpr(p):
        pr = dist.gauss_logpdf(p[0], loc=0, scale=10)
        pr += dist.gauss_logpdf(p[1], loc=0, scale=10)
        pr += dist.gamma_logpdf(p[2], shape=1, scale=1)
        return pr

    p0 = [1, 1, 0.05]
    mask = [1, 1, 0]
    LB = [-np.inf, 0.0001, 0.0001]
    UB = [np.inf, np.inf, np.inf]
    mh = MH(loglik, logpr)
    samples = mh.fit(p0, mask=mask, verbose=verbose, LB=LB, UB=UB)

    if do_plot:
        mh.plot_samples(samples, labels=['a', 'b', 'sig'])
        mh.plot_fit(forward, x, y_noise, samples.mean(axis=0))

    return samples


def test_mh_example():
    samples = mh_example(do_plot=False)
    assert 0 < samples.mean(axis=0)[0] < 2
    assert 0 < samples.mean(axis=0)[1] < 3


def plot_samples(samples, labels=None, plot_type='matrix'):
    """
    Plot summary of the sampling

    samples : array-like (num_samples x num_params)
    labels  : list
    plot_type : one of: 'matrix', 'vector', 'corr'
    """

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.figure()
    sns.set()

    if plot_type == 'matrix':
        df = pd.DataFrame(data=samples, columns=labels)
        g = sns.PairGrid(df)
        g.map_diag(plt.hist)
        g.map_offdiag(sns.kdeplot)
    elif plot_type == 'vector':
        mean = np.mean(samples, axis=0)

        std_pos = np.sqrt(np.sum(np.maximum(0, samples - mean)**2, axis=0) / np.sum((samples - mean) > 0, axis=0))
        std_neg = np.sqrt(np.sum(np.maximum(0, mean - samples)**2, axis=0) / np.sum((mean - samples) > 0, axis=0))

        fig, ax = plt.subplots(1)
        plt.errorbar(y=range(samples.shape[1]),
                     x=mean,
                     xerr=[std_neg, std_pos], fmt='o')
        if labels is not None:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels=labels)
    elif plot_type == 'corr':
        df = pd.DataFrame(data=samples, columns=labels)
        corr = df.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 8}, fmt='.1g')

    else:
        raise Exception('Unknown plot_type')


def plot_fit(forward, x, y, params):
    """
        Plot data fit

        forward : function
        data : array
        params : array
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set()
    plt.figure()
    sns.scatterplot(x, y)
    sns.lineplot(x, forward(params))


class MH:

    def __init__(self, loglik, logpr, burnin=1000, sampleevery=10, njumps=5000, update=20):
        """
        Initialise MH object

        Parameters
        ----------

        loglik: function
            Maps parameters to minus log-likelihood
        logpr:  function
            Maps parameters to minus log-prior
        burnin: int
            Number of iterations before actual sampling starts
        sampleevery: int
            Sampling rate
        njumps: int
            Number of sampling iterations
        update: int
            Rate of update of proposal distribution


        """

        self.burnin = burnin
        self.sampleevery = sampleevery
        self.njumps = njumps
        self.update = update
        self.loglik = loglik
        self.logpr = logpr

    def bounds_from_list(self, n, bounds):
        """
        Get bounds from list to two lists
        Args:
            n: num params
            bounds: sciipy-optimize-style bounds

        Returns:
        numpy 1D array (Lower bounds)
        numpy 1D array (Upper bounds)
        """
        LB = -np.inf * np.ones(n)
        UB = np.inf * np.ones(n)
        if bounds is None:
            return LB, UB
        if not isinstance(bounds, list):
            raise(Exception('bounds must either be a list or None'))
        for i, b in enumerate(bounds):
            LB[i] = b[0] if b[0] is not None else -np.inf
            UB[i] = b[1] if b[1] is not None else np.inf
        return LB, UB

    def fit(self, p0, mask=None, verbose=False, LB=None, UB=None):
        """
        Run Metropolis Hastings algorithm to fit data

        Parameters
        ----------

        p0 : array-like
            Initial values for the parameters to be fitted
        mask : array-like
            Mask for fixed parameters. Has the same size as p0, contains zero for fixed parameters
        verbose : boolean
        LB: array-like
            Lower bounds on parameters
        UB array-like
            Upper bounds on parameters

        Returns
        -------
        array
            Samples from the posterior distribution (nsamples X nparams)

        """
        # Convert to numpy array
        p0 = np.array(p0, dtype=float)

        if verbose:
            print("Initialisation")

        # Bounds
        LB = np.full(p0.size, -np.inf) if LB is None else LB
        UB = np.full(p0.size, np.inf) if UB is None else UB

        for idx in range(p0.size):
            if not LB[idx] <= p0[idx] <= UB[idx]:
                raise Exception("Initial values outside of range!!!")

        # Initialise p,e,acc,rej,prop
        p = np.array(p0, dtype=float)
        e = self.loglik(p) + self.logpr(p)
        acc = np.zeros(p.size)
        rej = np.zeros(p.size)
        prop = np.abs(p0) / 10  # np.ones(p.size)
        prop[prop == 0] = 1

        samples = np.zeros((self.njumps + self.burnin, p.size))

        # Mask
        if mask is None:
            mask = np.ones(p0.size)

        # Main loop
        maxiter = self.burnin + self.njumps
        if verbose:
            print("Begin MH sampling")
        for iter in range(maxiter):
            if verbose:
                print(".... Iter {}/{}".format(iter, maxiter))
            # Loop through params
            for idx in range(p.size):
                if mask[idx] != 0:
                    oldp = p[idx]
                    p[idx] = p[idx] + np.random.randn() * prop[idx]
                    if not LB[idx] <= p[idx] <= UB[idx]:
                        p[idx] = oldp
                        rej[idx] += 1
                    else:
                        olde = e
                        e = self.loglik(p) + self.logpr(p)
                        if np.exp(olde - e) > np.random.rand():
                            acc[idx] += 1
                        else:
                            p[idx] = oldp
                            rej[idx] += 1
                            e = olde
            # end loop over params
            samples[iter, :] = p
            if iter % self.update == 0:
                if verbose:
                    print(".... >>> Update Proposal ")
                prop *= np.sqrt((1 + acc) / (1 + rej))
                acc *= 0
                rej *= 0

        samples = samples[self.burnin::self.sampleevery]
        return samples

    def marglik_HM(self, samples):
        """
        Approximate Marginal Likelihood using Harmonic Mean estimator

        Parameters
        ----------
        samples : array-like
        """
        LL = np.zeros(samples.shape[0])
        for i in range(samples.shape[0]):
            LL[i] = self.loglik(samples[i, :])

        M = LL.max()
        ML = np.log(1 / np.sum(np.exp(LL - M))) - M - np.log(samples.shape[0])

        return ML

    def marglik_Laplace(self, samples):
        """
        Approximate Marginal Likelihood using Laplace approx

        Parameters
        ----------
        samples : array-like
        """
        mean = samples.mean(axis=0)
        detcov = np.linalg.det(np.cov(samples.T))

        LL = -self.loglik(mean)
        LP = -self.logpr(mean)

        ML = LL + LP + .5 * np.log(detcov) + samples.shape[1] / 2.0 * np.log(np.pi)

        return ML
