"""Voigt fitting model

Voigt model: Lorentzian (gamma) and Gaussian (sigma) broadening terms
Linewidths and frequency parameters constrained by metabolite group.
Will Clarke & Saad Jbabdi, University of Oxford, 2022
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq as sp_lstsq

from fsl_mrs.utils.misc import FIDToSpec


def vars(n_basis, n_groups, n_baseline):
    """Return the names and sizes of each parameter in the model
    :param n_basis: Number of basis spectra
    :type n_basis: int
    :param n_groups: Number of metabolite groups
    :type n_groups: int
    :param n_baseline: Number baseline bases
    :type n_baseline: int
    :return: List of parameter names
    :rtype: List
    :return: List of parameter sizes
    :rtype: List
    """
    var_names = ['conc', 'gamma', 'sigma', 'eps', 'Phi_0', 'Phi_1', 'baseline']
    sizes = [
        n_basis,  # Number of metabs
        n_groups,  # gamma
        n_groups,  # sigma
        n_groups,  # eps
        1,  # Phi_0
        1,  # Phi_1
        2 * n_baseline]  # baseline
    return var_names, sizes


def bounds(n_basis, n_groups, n_baseline, method, disableBaseline=False):
    """Return bounds for the methods

    :param n_basis: Number of basis spectra
    :type n_basis: int
    :param n_groups: Number of metabolite groups
    :type n_groups: int
    :param n_baseline: Number baseline bases
    :type n_baseline: int
    :param method: Fitting optimisation method. 'Newton' or 'MH'
    :type method: str
    :param disableBaseline: Disable the baseline by setting bounds to 0, defaults to False
    :type disableBaseline: bool, optional
    :return: For Newton method a list of (lower bound, upper bound) tuples.
        For MH method a 2-tuple of lower and upper bounds lists.
    :rtype: List or tuple
    """
    if method == 'Newton':
        # conc
        bnds = [(0, None)] * n_basis
        # gamma
        bnds.extend([(0, None)] * n_groups)
        # sigma
        bnds.extend([(0, None)] * n_groups)
        # eps
        bnds.extend([(None, None)] * n_groups)
        # phi0, phi1
        bnds.extend([(None, None)] * 2)
        # baseline
        n = n_baseline * 2
        if disableBaseline:
            bnds.extend([(0.0, 0.0)] * n)
        else:
            bnds.extend([(None, None)] * n)
        return bnds

    elif method == 'MH':
        MAX = 1e10
        MIN = -1e10
        # conc
        LB = [0] * n_basis
        UB = [MAX] * n_basis
        # gamma
        LB.extend([0] * n_groups)
        UB.extend([MAX] * n_groups)
        # sigma
        LB.extend([0] * n_groups)
        UB.extend([MAX] * n_groups)
        # eps
        LB.extend([MIN] * n_groups)
        UB.extend([MAX] * n_groups)
        # phi0, phi1
        LB.extend([MIN] * 2)
        UB.extend([MAX] * 2)
        # baseline
        n = n_baseline * 2
        if disableBaseline:
            LB.extend([0.0] * n)
            UB.extend([0.0] * n)
        else:
            LB.extend([MIN] * n)
            UB.extend([MAX] * n)

        return LB, UB


def mask(
        n_basis,
        n_groups,
        n_baseline,
        fit_conc=True,
        fit_shape=True,
        fit_phase=True,
        fit_baseline=False):
    """Generate parameter mask for MH and VB fitting methods

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

    if fit_conc:
        mask = [1] * n_basis
    else:
        mask = [0] * n_basis
    n = 3 * n_groups
    if fit_shape:
        mask.extend([1] * n)
    else:
        mask.extend([0] * n)
    if fit_phase:
        mask.extend([1] * 2)
    else:
        mask.extend([0] * 2)
    n = n_baseline * 2
    if fit_baseline:
        mask.extend([1] * n)
    else:
        mask.extend([0] * n)
    return mask


def x2param(x, n, g):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1,baselineparams]

    n  : number of metabiltes
    g  : number of metabolite groups
    """
    con = x[:n]           # concentrations
    gamma = x[n:n + g]        # lineshape scale parameter θ
    sigma = x[n + g:n + 2 * g]    # lineshape shape parameter k
    eps = x[n + 2 * g:n + 3 * g]    # frequency shift
    phi0 = x[n + 3 * g]        # global phase shift
    phi1 = x[n + 3 * g + 1]      # global phase ramp
    b = x[n + 3 * g + 2:]     # baseline params

    return con, gamma, sigma, eps, phi0, phi1, b


def param2x(con, gamma, sigma, eps, phi0, phi1, b):
    x = np.r_[con, gamma, sigma, eps, phi0, phi1, b]

    return x


def forward(x, nu, t, m, B, G, g):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1,baselineparams]

    nu : array-like - frequency axis
    t  : array-like - time axis
    m  : basis time course
    B  : baseline functions
    G  : metabolite groups
    g  : number of metab groups

    Returns forward prediction in the frequency domain
    """

    n = m.shape[1]    # get number of basis functions

    con, gamma, sigma, eps, phi0, phi1, b = x2param(x, n, g)

    E = np.zeros((m.shape[0], g), dtype=complex)
    for gg in range(g):
        E[:, gg] = np.exp(-(1j * eps[gg] + gamma[gg] + t * sigma[gg]**2) * t).flatten()

    tmp = np.zeros(m.shape, dtype=complex)
    for i, gg in enumerate(G):
        tmp[:, i] = m[:, i] * E[:, gg]

    M = FIDToSpec(tmp, axis=0)
    S = np.exp(-1j * (phi0 + phi1 * nu)) * (M @ con[:, None])

    # add baseline
    if B is not None:
        S += B @ b[:, None]

    return S.flatten()


def err(x, nu, t, m, B, G, g, data, first, last):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1,baselineparams]

    nu : array-like - frequency axis
    t  : array-like - time axis
    m  : basis time course
    B  : baseline functions
    G  : metabolite groups
    g  : number of metab groups
    data : array like - frequency domain data
    first,last : range for the fitting is data[first:last]

    returns scalar error
    """
    pred = forward(x, nu, t, m, B, G, g)
    err = data[first:last] - pred[first:last]
    sse = np.real(np.sum(err * np.conj(err)))
    return sse


def grad(x, nu, t, m, B, G, g, data, first, last):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1,baselineparams]

    nu : array-like - frequency axis
    t  : array-like - time axis
    m  : basis time course
    B  : baseline functions
    G  : metabolite groups
    g  : number of metab groups
    data : array like - frequency domain data
    first,last : range for the fitting is data[first:last]

    returns gradient vector
    """
    n = m.shape[1]    # get number of basis functions
    # g     = max(G)+1       # get number of metabolite groups

    con, gamma, sigma, eps, phi0, phi1, b = x2param(x, n, g)

    # Start
    E = np.zeros((m.shape[0], g), dtype=complex)
    SIG = np.zeros((m.shape[0], g), dtype=complex)
    for gg in range(g):
        E[:, gg] = np.exp(-(1j * eps[gg] + gamma[gg] + t * sigma[gg]**2) * t).flatten()
        SIG[:, gg] = sigma[gg]

    e_term = np.zeros(m.shape, dtype=complex)
    sig_term = np.zeros(m.shape, dtype=complex)
    c = np.zeros((con.size, g))
    for i, gg in enumerate(G):
        e_term[:, i] = E[:, gg]
        sig_term[:, i] = SIG[:, gg]
        c[i, gg] = con[i]
    m_term = m * e_term

    phi_term = np.exp(-1j * (phi0 + phi1 * nu))
    Fmet = FIDToSpec(m_term)
    Ftmet = FIDToSpec(t * m_term)
    Ft2sigmet = FIDToSpec(t * t * sig_term * m_term)
    Ftmetc = Ftmet @ c
    Ft2sigmetc = Ft2sigmet @ c
    Fmetcon = Fmet @ con[:, None]

    Spec = data[first:last, None]

    # Forward model
    S = (phi_term * Fmetcon)
    if B is not None:
        S += B @ b[:, None]

    # Gradients
    dSdc = phi_term * Fmet
    dSdgamma = phi_term * (-Ftmetc)
    dSdsigma = phi_term * (-2 * Ft2sigmetc)
    dSdeps = phi_term * (-1j * Ftmetc)
    dSdphi0 = -1j * phi_term * (Fmetcon)
    dSdphi1 = -1j * nu * phi_term * (Fmetcon)
    dSdb = B

    # Only compute within a range
    S = S[first:last]
    dSdc = dSdc[first:last, :]
    dSdgamma = dSdgamma[first:last, :]
    dSdsigma = dSdsigma[first:last, :]
    dSdeps = dSdeps[first:last, :]
    dSdphi0 = dSdphi0[first:last]
    dSdphi1 = dSdphi1[first:last]
    dSdb = dSdb[first:last]

    dS = np.concatenate((dSdc, dSdgamma, dSdsigma, dSdeps, dSdphi0, dSdphi1, dSdb), axis=1)

    grad = np.real(np.sum(S * np.conj(dS) + np.conj(S) * dS - np.conj(Spec) * dS - Spec * np.conj(dS), axis=0))

    return grad


def jac(x, nu, t, m, B, G, g, first, last):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1,baselineparams]

    nu : array-like - frequency axis
    t  : array-like - time axis
    m  : basis time course
    B  : baseline functions
    G  : metabolite groups
    g  : number of metab groups
    data : array like - frequency domain data
    first,last : range for the fitting is data[first:last]

    returns gradient vector
    """
    n = m.shape[1]  # get number of basis functions

    con, gamma, sigma, eps, phi0, phi1, b = x2param(x, n, g)

    # Start
    E = np.zeros((m.shape[0], g), dtype=complex)
    SIG = np.zeros((m.shape[0], g), dtype=complex)
    for gg in range(g):
        E[:, gg] = np.exp(-(1j * eps[gg] + gamma[gg] + t * sigma[gg] ** 2) * t).flatten()
        SIG[:, gg] = sigma[gg]

    e_term = np.zeros(m.shape, dtype=complex)
    sig_term = np.zeros(m.shape, dtype=complex)
    c = np.zeros((con.size, g))
    for i, gg in enumerate(G):
        e_term[:, i] = E[:, gg]
        sig_term[:, i] = SIG[:, gg]
        c[i, gg] = con[i]
    m_term = m * e_term

    phi_term = np.exp(-1j * (phi0 + phi1 * nu))
    Fmet = FIDToSpec(m_term)
    Ftmet = FIDToSpec(t * m_term)
    Ft2sigmet = FIDToSpec(t * t * sig_term * m_term)
    Ftmetc = Ftmet @ c
    Ft2sigmetc = Ft2sigmet @ c
    Fmetcon = Fmet @ con[:, None]

    # Gradients
    dSdc = phi_term * Fmet
    dSdgamma = phi_term * (-Ftmetc)
    dSdsigma = phi_term * (-2 * Ft2sigmetc)
    dSdeps = phi_term * (-1j * Ftmetc)
    dSdphi0 = -1j * phi_term * (Fmetcon)
    dSdphi1 = -1j * nu * phi_term * (Fmetcon)
    dSdb = B

    # Only compute within a range
    dSdc = dSdc[first:last, :]
    dSdgamma = dSdgamma[first:last, :]
    dSdsigma = dSdsigma[first:last, :]
    dSdeps = dSdeps[first:last, :]
    dSdphi0 = dSdphi0[first:last]
    dSdphi1 = dSdphi1[first:last]
    dSdb = dSdb[first:last]

    dS = np.concatenate((dSdc, dSdgamma, dSdsigma, dSdeps, dSdphi0, dSdphi1, dSdb), axis=1)

    return dS


def modify_basis(mrs, gamma, sigma, eps, first, last):
    bs = mrs.basis * np.exp(-(gamma + (sigma**2 * mrs.timeAxis) + 1j * eps) * mrs.timeAxis)
    bs = FIDToSpec(bs, axis=0)
    bs = bs[first:last, :]
    return np.concatenate((np.real(bs), np.imag(bs)), axis=0)


def loss(p, mrs, B, y, first, last):
    gamma, sigma, eps = np.exp(p[0]), np.exp(p[1]), p[2]
    basis = modify_basis(mrs, gamma, sigma, eps, first, last)
    desmat = np.concatenate((basis, B), axis=1)
    # pinv = np.linalg.pinv(desmat)
    # beta = np.real(pinv @ y)
    # beta = np.linalg.lstsq(desmat, y)[0]
    beta = sp_lstsq(desmat, y, lapack_driver='gelsy', check_finite=False)[0]
    # project onto >0 concentration
    beta[:mrs.numBasis] = np.clip(beta[:mrs.numBasis], 0, None)
    pred = np.matmul(desmat, beta)
    val = np.mean(np.abs(pred - y)**2)
    return val


# Initilisation functions
def _init_params_voigt(mrs, baseline, ppmlim):
    first, last = mrs.ppmlim_to_range(ppmlim)
    y = mrs.get_spec(ppmlim=ppmlim)
    y = np.concatenate((np.real(y), np.imag(y)), axis=0).flatten()
    B = baseline[first:last, :].copy()
    B = np.concatenate((np.real(B), np.imag(B)), axis=0)

    def local_loss(x):
        return loss(x, mrs, B, y, first, last)

    x0 = np.array([np.log(1), np.log(1), 0])
    bounds = (
        (None, np.log(100)),
        (None, np.log(100)),
        (-mrs.centralFrequency / 1E6, mrs.centralFrequency / 1E6))
    res = minimize(local_loss, x0, bounds=bounds)

    g, s, e = np.exp(res.x[0]), np.exp(res.x[1]), res.x[2]

    # get concentrations and baseline params
    basis = modify_basis(mrs, g, s, e, first, last)
    desmat = np.concatenate((basis, B), axis=1)
    beta = np.real(np.linalg.pinv(desmat) @ y)
    con = np.clip(beta[:mrs.numBasis], 0, None)
    # con    = beta[:mrs.numBasis]
    b = beta[mrs.numBasis:]

    return g, s, e, con, b


def init(mrs, metab_groups, baseline, ppmlim):
    """
       Initialise params of FSLModel for Voigt linesahapes
    """
    gamma, sigma, eps, con, b0 = _init_params_voigt(mrs, baseline, ppmlim)

    # Append
    x0 = con
    g = max(metab_groups) + 1                  # number of metab groups
    x0 = np.append(x0, [gamma] * g)              # gamma[0]..
    x0 = np.append(x0, [sigma] * g)              # sigma[0]..
    x0 = np.append(x0, [eps] * g)                # eps[0]..
    x0 = np.append(x0, [0, 0])                  # phi0 and phi1
    x0 = np.append(x0, b0)                     # baseline

    return x0
