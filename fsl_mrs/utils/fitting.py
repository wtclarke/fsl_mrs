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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fsl_mrs.core.mrs import MRS


def fit_FSLModel(mrs: "MRS",
                 method: str = 'Newton',
                 ppmlim: tuple[float, float] | None = None,
                 baseline: str | bline.Baseline = 'polynomial, 2',
                 baseline_order: int | None = None,
                 metab_groups: list[int] | None = None,
                 model: models.model_strings = 'voigt',
                 x0: list[float] | None = None,
                 MHSamples: int = 500,
                 disable_mh_priors: bool = False,
                 fit_baseline_mh: bool = False,
                 scipy_min_options_dict: dict | None = dict(maxfun=1E5),
                 capture_minimize_output: bool = False):
    """Run linear combination fitting on the passed mrs object.

    Can run either with a truncated Newton (method='Newton') or Metropolis Hastings (method='MH') optimiser.

    :param mrs: MRS object containing the data, the basis set and optionally the water reference
    :type mrs: fsl_mrs.core.MRS
    :param method: 'Newton' or 'MH', defaults to 'Newton'
    :type method: str, optional
    :param ppmlim: ppm range over which to fit, defaults to nucleus standard (via None) e.g. (.2, 4.2) for 1H.
    :type ppmlim: tuple, optional
    :param baseline: Baseline specification string or reusable Baseline object, defaults to 'polynomial, 2'
    :type baseline: str or fsl_mrs.utils.baseline.Baseline, optional
    :param baseline_order: Polynomial baseline order, defaults to 2, -1 disables.
    :type baseline_order: int, optional
    :param metab_groups: List of metabolite groupings, defaults to None
    :type metab_groups: List, optional
    :param model: 'lorentzian', 'voigt', 'free_shift', 'free_shift_lorentzian', 'negativevoigt', defaults to 'voigt'
    :type model: str, optional
    :param x0: Initialisation values, defaults to None
    :type x0: List, optional
    :param MHSamples: Number of MH steps to run, defaults to 500 (will produce 50 samples)
    :type MHSamples: int, optional
    :param disable_mh_priors: If True all priors are disabled for MH fitting, defaults to False
    :type disable_mh_priors: bool, optional
    :param fit_baseline_mh: If true baseline parameters are also fit using MH, defaults to False
    :type fit_baseline_mh: bool, optional
    :param scipy_min_options_dict: Options dict passed to scipy.minimise TNC function.
    :type scipy_min_options_dict: dict or None, optional
    :param capture_minimize_output: Attach the scipy.optimize.minimize result object for diagnostics,
        defaults to False.
    :type capture_minimize_output: bool, optional

    :return: Fit results object
    :rtype: fsl_mrs.utils.FitRes
    """

    err_func, grad_func, forward, x2p, p2x = models.getModelFunctions(model)

    init_func = models.getInit(model)         # initialisation of params

    data = mrs.get_spec().copy()              # data copied to keep it safe

    # A supplied Baseline object defines both the baseline basis and, if
    # ppmlim is omitted, the fit range it was constructed for.
    if isinstance(baseline, bline.Baseline):
        if baseline_order is not None:
            raise ValueError('baseline_order cannot be used when baseline is a Baseline object.')
        baseline_obj = baseline
        if ppmlim is None:
            ppmlim = baseline_obj.ppmlim
    elif isinstance(baseline, str):
        baseline_obj = None
    else:
        raise TypeError('baseline must be a string or fsl_mrs.utils.baseline.Baseline object.')

    # Find appropriate ppm limit for nucleus
    if ppmlim is None and baseline_obj is None:
        ppmlim = nucleus_constants(mrs.nucleus).ppm_range
    if ppmlim is None and baseline_obj is None:
        raise ValueError(
            'Please specify a fitting range (ppmlim): '
            f'No ppmlim specified and no default found for nucleus {mrs.nucleus}.')

    indices = mrs.axes.ppmShiftIndices(ppmlim)

    if metab_groups is None:
        metab_groups = [0] * len(mrs.names)

    # shorter names for some of the useful stuff
    freq, time, basis = mrs.frequencyAxis, mrs.timeAxis, mrs.get_basis(copy=False)

    # Prepare baseline
    if baseline_obj is None:
        baseline_obj = bline.Baseline(
            mrs,
            ppmlim,
            baseline,
            baseline_order)
    else:
        baseline_obj.validate_mrs(mrs, ppmlim)

    # Constants
    if metab_groups is None:
        g = 1
    else:
        g = max(metab_groups) + 1
    constants = (freq, time, basis, baseline_obj.regressor, metab_groups, g, data, indices)

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
            options=scipy_min_options_dict)
        # Results
        results = FitRes(mrs, res.x, model, method, metab_groups, baseline_obj, ppmlim)
        if capture_minimize_output:
            results.scipy_minimize_result = res

    elif method == 'init':
        results = FitRes(mrs, x0, model, method, metab_groups, baseline_obj, ppmlim)

    elif method == 'MH':
        from fsl_mrs.utils.stats import mh, dist

        def forward_mh(p):
            return forward(p, freq, time, basis, baseline_obj.regressor, metab_groups, g)[indices]
        numPoints_over_2 = (indices.stop - indices.start) / 2.0
        y = data[indices]

        if fit_baseline_mh and baseline_obj.mode == 'spline':
            penalty_function = baseline_obj.mh_penalty_term()

            def loglik(p):
                lik = np.linalg.norm(y - forward_mh(p)) + penalty_function(p)
                return np.log(lik) * numPoints_over_2
        else:
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
            model=model,
            capture_minimize_output=capture_minimize_output)
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
        if capture_minimize_output and hasattr(res, 'scipy_minimize_result'):
            results.scipy_minimize_result = res.scipy_minimize_result

    else:
        raise Exception('Unknown optimisation method.')

    # End of fitting

    return results
