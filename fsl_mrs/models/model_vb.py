'''
Utils for VB implementation
Only needs forward model
does exponentiate positive parameters (i.e. they are in log-transform)
ensures prediction is real by concatenating real and imag signals
'''

import numpy as np

from fsl_mrs.utils.misc import FIDToSpec
from model_lorentzian import x2param
from model_voigt import x2param as x2param_voigt


# ########### For VB
# Exponentiate positive params
def FSLModel_forward_vb(x, nu, t, m, B, G, g, first, last):
    n = m.shape[1]    # get number of basis functions

    logcon, loggamma, eps, phi0, phi1, b = x2param(x, n, g)
    con = np.exp(logcon)
    gamma = np.exp(loggamma)

    E = np.zeros((m.shape[0], g), dtype=complex)
    for gg in range(g):
        E[:, gg] = np.exp(-(1j * eps[gg] + gamma[gg]) * t).flatten()

    tmp = np.zeros(m.shape, dtype=complex)
    for i, gg in enumerate(G):
        tmp[:, i] = m[:, i] * E[:, gg]

    M = FIDToSpec(tmp)
    S = np.exp(-1j * (phi0 + phi1 * nu)) * (M @ con[:, None])

    # add baseline
    if B is not None:
        S += B @ b[:, None]

    S = S.flatten()[first:last]

    return np.concatenate((np.real(S), np.imag(S)))

# Gradient of the forward model (not the error)
# !!! grad wrt logparam (for those that are logged)
#  dfdlogx = x*dfdx


def FSLModel_grad_vb(x, nu, t, m, B, G, g, first, last):
    n = m.shape[1]    # get number of basis functions

    logcon, loggamma, eps, phi0, phi1, b = x2param(x, n, g)
    con = np.exp(logcon)
    gamma = np.exp(loggamma)

    # Start
    E = np.zeros((m.shape[0], g), dtype=complex)
    for gg in range(g):
        E[:, gg] = np.exp(-(1j * eps[gg] + gamma[gg]) * t).flatten()

    e_term = np.zeros(m.shape, dtype=complex)
    c = np.zeros((con.size, g))
    for i, gg in enumerate(G):
        e_term[:, i] = E[:, gg]
        c[i, gg] = con[i]
    m_term = m * e_term

    phi_term = np.exp(-1j * (phi0 + phi1 * nu))

    Fmet = FIDToSpec(m_term)
    Ftmet = FIDToSpec(t * m_term)
    Ftmetc = Ftmet @ c
    Fmetcon = Fmet @ con[:, None]

    # Forward model
    S = (phi_term * Fmetcon)
    if B is not None:
        S += B @ b[:, None]

    # Gradients
    dSdc = phi_term * Fmet
    dSdgamma = phi_term * (-Ftmetc)
    dSdeps = phi_term * (-1j * Ftmetc)
    dSdphi0 = -1j * phi_term * (Fmetcon)
    dSdphi1 = -1j * nu * phi_term * (Fmetcon)
    dSdb = B

    # Only compute within a range
    dSdc = dSdc[first:last, :]
    dSdgamma = dSdgamma[first:last, :]
    dSdeps = dSdeps[first:last, :]
    dSdphi0 = dSdphi0[first:last]
    dSdphi1 = dSdphi1[first:last]
    dSdb = dSdb[first:last]

    dS = np.concatenate((dSdc * con[None, :],
                         dSdgamma * gamma[None, :],
                         dSdeps,
                         dSdphi0,
                         dSdphi1,
                         dSdb), axis=1)

    dS = np.concatenate((np.real(dS), np.imag(dS)), axis=0)

    return dS


def FSLModel_forward_vb_voigt(x, nu, t, m, B, G, g, first, last):
    n = m.shape[1]    # get number of basis functions

    logcon, loggamma, logsigma, eps, phi0, phi1, b = x2param_voigt(x, n, g)
    con = np.exp(logcon)
    gamma = np.exp(loggamma)
    sigma = np.exp(logsigma)

    E = np.zeros((m.shape[0], g), dtype=complex)
    for gg in range(g):
        E[:, gg] = np.exp(-(1j * eps[gg] + gamma[gg] + t * sigma[gg]**2) * t).flatten()

    tmp = np.zeros(m.shape, dtype=complex)
    for i, gg in enumerate(G):
        tmp[:, i] = m[:, i] * E[:, gg]

    M = FIDToSpec(tmp)
    S = np.exp(-1j * (phi0 + phi1 * nu)) * (M @ con[:, None])

    # add baseline
    if B is not None:
        S += B @ b[:, None]

    S = S.flatten()[first:last]

    return np.concatenate((np.real(S), np.imag(S)))
