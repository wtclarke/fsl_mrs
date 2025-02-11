"""FSL-MRS fitting model functions

Will Clarke & Saad Jbabdi, University of Oxford, 2022
"""

import numpy as np
import fsl_mrs.models.model_freeshift as freeshift
import fsl_mrs.models.model_voigt as voigt
import fsl_mrs.models.model_lorentzian as lorentzian
import fsl_mrs.models.model_negativevoigt as negativevoigt


def getModelFunctions(model):
    """ Return the err, grad, forward and conversion functions appropriate for the model."""
    if model == 'lorentzian':
        err_func = lorentzian.err          # error function
        grad_func = lorentzian.grad         # gradient
        forward = lorentzian.forward      # forward model
        x2p = lorentzian.x2param
        p2x = lorentzian.param2x
    elif model == 'voigt':
        err_func = voigt.err     # error function
        grad_func = voigt.grad    # gradient
        forward = voigt.forward  # forward model
        x2p = voigt.x2param
        p2x = voigt.param2x
    elif model == 'free_shift':
        err_func = freeshift.err     # error function
        grad_func = freeshift.grad    # gradient
        forward = freeshift.forward  # forward model
        x2p = freeshift.x2param
        p2x = freeshift.param2x
    elif model == 'negativevoigt':
        err_func = negativevoigt.err     # error function
        grad_func = negativevoigt.grad    # gradient
        forward = negativevoigt.forward  # forward model
        x2p = negativevoigt.x2param
        p2x = negativevoigt.param2x
    else:
        raise ValueError('Unknown model {}.'.format(model))
    return err_func, grad_func, forward, x2p, p2x


def getModelForward(model):
    """Return the model forward function

    :param model: fitting model name: 'lorentzian', 'voigt',
     or 'free_shift' or 'negativevoigt'
    :type model: str
    :return: forward function
    :rtype: _type_
    """
    _, _, fwd, _, _ = getModelFunctions(model)
    return fwd


def getModelJac(model):
    """Return the model jacobian function

    :param model: fitting model name: 'lorentzian', 'voigt',
     or 'free_shift' or 'negativevoigt'
    :type model: str
    :return: Jacobian function
    :rtype: function
    """
    if model == 'lorentzian':
        jac = lorentzian.jac
    elif model == 'voigt':
        jac = voigt.jac
    elif model == 'free_shift':
        jac = freeshift.jac
    elif model == 'negativevoigt':
        jac = negativevoigt.jac        
    else:
        raise ValueError('Unknown model {}.'.format(model))
    return jac


def getInit(model):
    """Return the initilisation function

    :param model: fitting model name: 'lorentzian', 'voigt',
     or 'free_shift' or 'negativevoigt'
    :type model: str
    :return: Init function
    :rtype: function
    """
    if model == 'lorentzian':
        return lorentzian.init
    elif model == 'voigt':
        return voigt.init
    elif model == 'free_shift':
        return freeshift.init
    elif model == 'negativevoigt':
        return negativevoigt.init
    else:
        raise ValueError('Unknown model {}.'.format(model))


def FSLModel_vars(model, n_basis=None, n_groups=1, n_baseline=0):
    """
    Print out parameter names as a list of strings
    Args:
        model: str
        n_basis: int, number of basis spectra
        n_groups: int, number of metabolite groups
        n_baseline: int, number of baseline bases
    Returns:
        list of strings
        list of int
    """
    if model == 'lorentzian':
        return lorentzian.vars(n_basis, n_groups, n_baseline)
    elif model == 'voigt':
        return voigt.vars(n_basis, n_groups, n_baseline)
    elif model == 'free_shift':
        return freeshift.vars(n_basis, n_groups, n_baseline)
    elif model == 'negativevoigt':
        return negativevoigt.vars(n_basis, n_groups, n_baseline)
    else:
        raise ValueError('Unknown model {}.'.format(model))


def FSLModel_bounds(model, n_basis, n_groups, n_baseline, method, disableBaseline=False):
    """Return fitting parameter bounds associated with each model

    :param model: Name of model used
    :type model: str
    :param n_basis: Number of basis spectra
    :type n_basis: int
    :param n_groups: Number of metabolite groups
    :type n_groups: int
    :param n_baseline: Number of baseline bases
    :type n_baseline: int
    :param method: Fitting optimisation method. 'Newton' or 'MH'
    :type method: str
    :param disableBaseline: Disable baseline fit by setting bounds to zero, defaults to False
    :type disableBaseline: bool, optional
    :return: For Newton method a list of (lowr bound, upper bound) tuples.
        For MH method a 2-tuple of lower and upper bounds lists.
    :rtype: List or tuple
    """

    if model == 'lorentzian':
        return lorentzian.bounds(n_basis, n_groups, n_baseline, method, disableBaseline)
    elif model == 'voigt':
        return voigt.bounds(n_basis, n_groups, n_baseline, method, disableBaseline)
    elif model == 'free_shift':
        return freeshift.bounds(n_basis, n_groups, n_baseline, method, disableBaseline)
    elif model == 'negativevoigt':
        return negativevoigt.bounds(n_basis, n_groups, n_baseline, method, disableBaseline)
    else:
        raise ValueError('Unknown model {}.'.format(model))


def FSLModel_mask(
        model,
        n_basis,
        n_groups,
        n_baseline,
        fit_conc=True,
        fit_shape=True,
        fit_phase=True,
        fit_baseline=False):
    """Return parameter mask for MH and VB fitting methods

    :param model: Name of model used
    :type model: str
    :param n_basis: Number of basis spectra
    :type n_basis: int
    :param n_groups: Number of metabolite groups
    :type n_groups: int
    :param n_baseline: Number baseline bases
    :type n_baseline: int
    :param fit_conc: Whether to fit the concentrations, defaults to True
    :type fit_conc: bool, optional
    :param fit_shape: Whether to fit the lineshapes, defaults to True
    :type fit_shape: bool, optional
    :param fit_phase: Whether to fit the phase, defaults to True
    :type fit_phase: bool, optional
    :param fit_baseline: Whether to fit the baseline, defaults to False
    :type fit_baseline: bool, optional
    :return: Parameter mask
    :rtype: list
    """
    if model == 'lorentzian':
        return lorentzian.mask(
            n_basis,
            n_groups,
            n_baseline,
            fit_conc=fit_conc,
            fit_shape=fit_shape,
            fit_phase=fit_phase,
            fit_baseline=fit_baseline)
    elif model == 'voigt':
        return voigt.mask(
            n_basis,
            n_groups,
            n_baseline,
            fit_conc=fit_conc,
            fit_shape=fit_shape,
            fit_phase=fit_phase,
            fit_baseline=fit_baseline)
    elif model == 'free_shift':
        return freeshift.mask(
            n_basis,
            n_groups,
            n_baseline,
            fit_conc=fit_conc,
            fit_shape=fit_shape,
            fit_phase=fit_phase,
            fit_baseline=fit_baseline)
    elif model == 'negativevoigt':
        return negativevoigt.mask(
            n_basis,
            n_groups,
            n_baseline,
            fit_conc=fit_conc,
            fit_shape=fit_shape,
            fit_phase=fit_phase,
            fit_baseline=fit_baseline)
    else:
        raise ValueError('Unknown model {}.'.format(model))


def getFittedModel(model, resParams, base_poly, metab_groups, mrs,
                   basisSelect=None, baselineOnly=False, noBaseline=False,
                   no_phase=False):
    """ Return the predicted model given some fitting parameters

        model     (str)  : Model string
        resParams (array):
        base_poly
        metab_groups
        mrs   (class obj):

    """
    numBasis = len(mrs.names)
    numGroups = max(metab_groups) + 1

    _, _, forward, x2p, p2x = getModelFunctions(model)

    if noBaseline:
        bp = np.zeros(base_poly.shape)
    else:
        bp = base_poly

    if no_phase:
        p = x2p(resParams, numBasis, numGroups)
        p = p[0:-3] + (0.0, 0.0) + p[-1:]
        params = p2x(*p)
    else:
        params = resParams

    if basisSelect is None and not baselineOnly:
        return forward(params,
                       mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis,
                       bp,
                       metab_groups,
                       numGroups)
    elif baselineOnly:
        p = x2p(params, numBasis, numGroups)
        p = (np.zeros(numBasis),) + p[1:]
        xx = p2x(*p)
        return forward(xx,
                       mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis,
                       bp,
                       metab_groups,
                       numGroups)
    elif basisSelect is not None:
        p = x2p(params, numBasis, numGroups)
        tmp = np.zeros(numBasis)
        basisIdx = mrs.names.index(basisSelect)
        tmp[basisIdx] = p[0][basisIdx]
        p = (tmp,) + p[1:]
        xx = p2x(*p)
        return forward(xx,
                       mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis,
                       bp,
                       metab_groups,
                       numGroups)
