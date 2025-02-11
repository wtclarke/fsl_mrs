#!/usr/bin/env python

# fitting.py - Fit MRS models
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from scipy.optimize import minimize

from fsl_mrs import models
from fsl_mrs.utils.results import FitRes
from fsl_mrs.utils.constants import nucleus_constants
from fsl_mrs.utils import baseline as bline


def fit_FSLModel(mrs,
                 method: str = 'Newton',
                 ppmlim=None,
                 baseline: str = 'polynomial, 2',
                 baseline_order: int | None = None,
                 metab_groups=None,
                 model: str = 'voigt',
                 x0=None,
                 MHSamples=500,
                 disable_mh_priors=False,
                 fit_baseline_mh=False):
    """Run linear combination fitting on the passed mrs object.

    Can run either with a truncated Newton (method='Newton') or Metropolis Hastings (method='MH') optimiser.

    :param mrs: MRS object containing the data, the basis set and optionally the water reference
    :type mrs: fsl_mrs.core.MRS
    :param method: 'Newton' or 'MH', defaults to 'Newton'
    :type method: str, optional
    :param ppmlim: ppm range over which to fit, defaults to nucleus standard (via None) e.g. (.2, 4.2) for 1H.
    :type ppmlim: tuple, optional
    :param baseline_order: Polynomial baseline order, defaults to 2, -1 disables.
    :type baseline_order: int, optional
    :param metab_groups: List of metabolite groupings, defaults to None
    :type metab_groups: List, optional
    :param model: 'lorentzian' or 'voigt', defaults to 'voigt'
    :type model: str, optional
    :param x0: Initialisation values, defaults to None
    :type x0: List, optional
    :param MHSamples: Number of MH samples to run, defaults to 500
    :type MHSamples: int, optional
    :param disable_mh_priors: If True all priors are disabled for MH fitting, defaults to False
    :type disable_mh_priors: bool, optional
    :param fit_baseline_mh: If true baseline parameters are also fit using MH, defaults to False
    :type fit_baseline_mh: bool, optional

    :return: Fit results object
    :rtype: fsl_mrs.utils.FitRes
    """

    err_func, grad_func, forward, x2p, p2x = models.getModelFunctions(model)

    init_func = models.getInit(model)         # initialisation of params

    data = mrs.get_spec().copy()              # data copied to keep it safe

    # Find appropriate ppm limit for nucleus
    if ppmlim is None:
        ppmlim = nucleus_constants(mrs.nucleus).ppm_range
    if ppmlim is None:
        raise ValueError(
            'Please specify a fitting range (ppmlim): '
            f'No ppmlim specified and no default found for nucleus {mrs.nucleus}.')

    first, last = mrs.ppmlim_to_range(ppmlim)  # data range

    if metab_groups is None:
        metab_groups = [0] * len(mrs.names)

    # shorter names for some of the useful stuff
    freq, time, basis = mrs.frequencyAxis, mrs.timeAxis, mrs.basis

    # Prepare baseline
    baseline_obj = bline.Baseline(
        mrs,
        ppmlim,
        baseline,
        baseline_order)

    # Constants
    if metab_groups is None:
        g = 1
    else:
        g = max(metab_groups) + 1
    constants = (freq, time, basis, baseline_obj.regressor, metab_groups, g, data, first, last)

    if x0 is None:
        # Initialise all params
        x0 = init_func(mrs, metab_groups, baseline_obj.regressor, ppmlim)

    # Fitting
    if method == 'Newton':
        # Bounds
        bounds = models.FSLModel_bounds(
            model,
            mrs.numBasis,
            g,
            baseline_obj.n_basis,
            method,
            disableBaseline=baseline_obj.disabled)

        err_func, grad_func = baseline_obj.prepare_penalised_fit_functions(
            err_func,
            grad_func,
            lambda x: x2p(x, mrs.numBasis, g)[-1])

        res = minimize(
            err_func,
            x0,
            args=constants,
            method='TNC',
            jac=grad_func,
            bounds=bounds,
            options=dict(maxfun=1E5))
        # Results
        results = FitRes(mrs, res.x, model, method, metab_groups, baseline_obj, ppmlim)

    elif method == 'init':
        results = FitRes(mrs, x0, model, method, metab_groups, baseline_obj, ppmlim)

    elif method == 'MH':
        from fsl_mrs.utils.stats import mh, dist

        def forward_mh(p):
            return forward(p, freq, time, basis, baseline_obj.regressor, metab_groups, g)[first:last]
        numPoints_over_2 = (last - first) / 2.0
        y = data[first:last]

        def loglik(p):
            return np.log(np.linalg.norm(y - forward_mh(p))) * numPoints_over_2

        if disable_mh_priors:
            def logpr(p):
                return np.sum(dist.gauss_logpdf(p, loc=np.zeros_like(p), scale=np.ones_like(p) * 1E2))
        else:
            from fsl_mrs.utils.constants import MCMC_PRIORS

            def logpr(p):
                def make_prior(param, loc, scale):
                    return np.sum(dist.gauss_logpdf(param,
                                                    loc=loc * np.ones_like(param),
                                                    scale=scale * np.ones_like(param)))
                prior = 0
                if model.lower() == 'lorentzian':
                    con, gamma, eps, phi0, phi1, b = x2p(p, mrs.numBasis, g)
                    PRIORS = MCMC_PRIORS['lorentzian']

                    prior += make_prior(con, PRIORS['conc_loc'], PRIORS['conc_scale'])
                    prior += make_prior(gamma,
                                        PRIORS['gamma_loc'] * np.pi,
                                        PRIORS['gamma_scale'] * np.pi)
                    prior += make_prior(eps,
                                        PRIORS['eps_loc'] * 2 * np.pi * mrs.centralFrequency / 1E6,
                                        PRIORS['eps_scale'] * 2 * np.pi * mrs.centralFrequency / 1E6)
                    prior += make_prior(phi0,
                                        PRIORS['phi0_loc'] * np.pi / 180,
                                        PRIORS['phi0_scale'] * np.pi / 180)
                    prior += make_prior(phi1,
                                        PRIORS['phi1_loc'] * 2 * np.pi,
                                        PRIORS['phi1_scale'] * 2 * np.pi)

                elif model.lower() == 'voigt':
                    con, gamma, sigma, eps, phi0, phi1, b = x2p(p, mrs.numBasis, g)
                    PRIORS = MCMC_PRIORS['voigt']

                    prior += make_prior(con, PRIORS['conc_loc'], PRIORS['conc_scale'])
                    prior += make_prior(gamma,
                                        PRIORS['gamma_loc'] * np.pi,
                                        PRIORS['gamma_scale'] * np.pi)
                    prior += make_prior(sigma,
                                        PRIORS['sigma_loc'] * np.pi,
                                        PRIORS['sigma_scale'] * np.pi)
                    prior += make_prior(eps,
                                        PRIORS['eps_loc'] * 2 * np.pi * mrs.centralFrequency / 1E6,
                                        PRIORS['eps_scale'] * 2 * np.pi * mrs.centralFrequency / 1E6)
                    prior += make_prior(phi0,
                                        PRIORS['phi0_loc'] * np.pi / 180,
                                        PRIORS['phi0_scale'] * np.pi / 180)
                    prior += make_prior(phi1,
                                        PRIORS['phi1_loc'] * 2 * np.pi,
                                        PRIORS['phi1_scale'] * 2 * np.pi)
                return prior

        # Setup the fitting
        # Init with nonlinear fit
        res = fit_FSLModel(
            mrs,
            method='Newton',
            ppmlim=ppmlim,
            metab_groups=metab_groups,
            baseline=baseline,
            baseline_order=baseline_order,
            model=model)
        # Create masks and bounds for MH fit
        p0 = res.params

        LB, UB = models.FSLModel_bounds(
            model,
            mrs.numBasis,
            g,
            baseline_obj.n_basis,
            method,
            disableBaseline=baseline_obj.disabled)
        mask = models.FSLModel_mask(
            model,
            mrs.numBasis,
            g,
            baseline_obj.n_basis,
            fit_baseline=fit_baseline_mh)

        # Check that the values initialised by the newton
        # method don't exceed these bounds (unlikely but possible with bad data)
        for i, (p, u, l) in enumerate(zip(p0, UB, LB)):
            if p > u:
                p0[i] = u
            elif p < l:
                p0[i] = l

        # Do the fitting
        mcmc = mh.MH(loglik, logpr, burnin=100, njumps=MHSamples)
        samples = mcmc.fit(p0, LB=LB, UB=UB, verbose=False, mask=mask)

        # collect results
        results = FitRes(mrs, samples, model, method, metab_groups, baseline_obj, ppmlim)

    else:
        raise Exception('Unknown optimisation method.')

    # End of fitting

    return results
