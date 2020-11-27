#!/usr/bin/env python

# models.py - MRS forward models and helper functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <will.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.utils.misc import FIDToSpec,SpecToFID

# Helper functions for LCModel fitting


# ##################### FSL MODEL
def FSLModel_vars(model='voigt'):
    """
    Print out parameter names as a list of strings
    Args:
        model: str (either 'lorientzian' or 'voigt'

    Returns:
        list of strings
    """
    if model == 'lorentzian':
        var_names = ['conc', 'gamma', 'eps', 'Phi_0', 'Phi_1', 'baseline']
    elif model == 'voigt':
        var_names = ['conc', 'gamma', 'sigma', 'eps', 'Phi_0', 'Phi_1', 'baseline']
    else:
        raise(Exception('model must be either "voigt" or "lorentzian"'))
    return var_names

def FSLModel_x2param(x,n,g):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1,baselineparams]

    n  : number of metabolites
    g  : number of metabolite groups
    """
    con   = x[:n]           # concentrations
    gamma = x[n:n+g]        # lorentzian blurring
    eps   = x[n+g:n+2*g]    # frequency shift
    phi0  = x[n+2*g]        # global phase shift
    phi1  = x[n+2*g+1]      # global phase ramp
    b     = x[n+2*g+2:]     # baseline params

    return con,gamma,eps,phi0,phi1,b

def FSLModel_param2x(con,gamma,eps,phi0,phi1,b):
    x = np.r_[con,gamma,eps,phi0,phi1,b]
    
    return x


def FSLModel_transform_basis(x,nu,t,m,G,g):
    """
       Transform basis by applying frequency shifting/blurring
    """
    n     = m.shape[1]    # get number of basis functions

    con,gamma,eps,phi0,phi1,b = FSLModel_x2param(x,n,g)

    E = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg])*t).flatten()
    
    tmp = np.zeros(m.shape,dtype=np.complex)
    for i,gg in enumerate(G):
        tmp[:,i] = m[:,i]*E[:,gg]
    
    M     = FIDToSpec(tmp)
    
    return SpecToFID(np.exp(-1j*(phi0+phi1*nu))*M)
    
    #return tmp
    
def FSLModel_forward(x,nu,t,m,B,G,g):
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
    
    n     = m.shape[1]    # get number of basis functions

    con,gamma,eps,phi0,phi1,b = FSLModel_x2param(x,n,g)

    E = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg])*t).flatten()
    # E = np.exp(-(1j*eps+gamma)*t) # THis is actually slower! But maybe more optimisable longterm with numexpr or numba

    #tmp = np.zeros(m.shape,dtype=np.complex)
    #for i,gg in enumerate(G):
    #    tmp[:,i] = m[:,i]*E[:,gg]
    tmp = m*E[:,G]
    
    M     = FIDToSpec(tmp)
    S     = np.exp(-1j*(phi0+phi1*nu)) * (M@con[:,None])

    # add baseline
    if B is not None:                
        S += B@b[:,None]
    
    return S.flatten()

def FSLModel_err(x,nu,t,m,B,G,g,data,first,last):
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
    pred = FSLModel_forward(x,nu,t,m,B,G,g)
    err  = data[first:last]-pred[first:last]
    sse  = np.real(np.sum(err*np.conj(err)))
    return sse



def FSLModel_forward_and_jac(x,nu,t,m,B,G,g,data,first,last):
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

    returns jacobian matrix
    """
    n     = m.shape[1]    # get number of basis functions
    #g     = max(G)+1       # get number of metabolite groups

    con,gamma,eps,phi0,phi1,b = FSLModel_x2param(x,n,g)

    # Start 
    E = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg])*t).flatten()
    
    e_term   = np.zeros(m.shape,dtype=np.complex)
    c        = np.zeros((con.size,g))
    for i,gg in enumerate(G):
        e_term[:,i] = E[:,gg]
        c[i,gg] = con[i]
    m_term = m*e_term
    
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    
    Fmet     = FIDToSpec(m_term)
    Ftmet    = FIDToSpec(t*m_term)
    Ftmetc   = Ftmet@c
    Fmetcon = Fmet@con[:,None]
    
    Spec     = data[first:last,None]

    # Forward model 
    S     = (phi_term*Fmetcon)
    if B is not None:
        S += B@b[:,None]
        
    # Gradients        
    dSdc     = phi_term*Fmet
    dSdgamma = phi_term*(-Ftmetc)
    dSdeps   = phi_term*(-1j*Ftmetc)
    dSdphi0  = -1j*phi_term*(Fmetcon)
    dSdphi1  = -1j*nu*phi_term*(Fmetcon)
    dSdb     = B
    
    # Only compute within a range            
    S         = S[first:last]
    dSdc      = dSdc[first:last,:]
    dSdgamma  = dSdgamma[first:last,:]    
    dSdeps    = dSdeps[first:last,:]
    dSdphi0   = dSdphi0[first:last]
    dSdphi1   = dSdphi1[first:last]
    dSdb      = dSdb[first:last]
    
    jac  = np.concatenate((dSdc,dSdgamma,dSdeps,dSdphi0,dSdphi1,dSdb),axis=1)
    
    return S,jac

def FSLModel_grad(x,nu,t,m,B,G,g,data,first,last):
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

    S,dS = FSLModel_forward_and_jac(x,nu,t,m,B,G,g,data,first,last)
    Spec = data[first:last,None]
    grad = np.real(np.sum(S*np.conj(dS)+np.conj(S)*dS - np.conj(Spec)*dS - Spec*np.conj(dS),axis=0))
    
    return grad


# CODE FOR CHECKING THE GRADIENTS
# cf = lambda p : FSLModel_err(p,nu,t,m,B,G,data,first,last)
# x0 = xx+np.random.normal(0, .01, xx.shape)
# grad_num = misc.gradient(x0,cf)
# grad_ana = FSLModel_grad(x0,nu,t,m,B,G,data,first,last)

# print(grad_num.shape)
# print(grad_ana.shape)

# %matplotlib notebook
# plt.plot(grad_num,grad_ana,'.')


# ##################### FSL MODEL including voigt distribution lineshape
def FSLModel_x2param_Voigt(x,n,g):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1,baselineparams]

    n  : number of metabiltes
    g  : number of metabolite groups
    """
    con   = x[:n]           # concentrations
    gamma = x[n:n+g]        # lineshape scale parameter θ
    sigma = x[n+g:n+2*g]    # lineshape shape parameter k
    eps   = x[n+2*g:n+3*g]    # frequency shift
    phi0  = x[n+3*g]        # global phase shift
    phi1  = x[n+3*g+1]      # global phase ramp
    b     = x[n+3*g+2:]     # baseline params

    return con,gamma,sigma,eps,phi0,phi1,b

def FSLModel_param2x_Voigt(con,gamma,sigma,eps,phi0,phi1,b):
    x = np.r_[con,gamma,sigma,eps,phi0,phi1,b]
    
    return x

def FSLModel_forward_Voigt(x,nu,t,m,B,G,g):
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
    
    n     = m.shape[1]    # get number of basis functions

    con,gamma,sigma,eps,phi0,phi1,b = FSLModel_x2param_Voigt(x,n,g)

    E = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg]+t*sigma[gg]**2)*t).flatten()
    
    tmp = np.zeros(m.shape,dtype=np.complex)
    for i,gg in enumerate(G):
        tmp[:,i] = m[:,i]*E[:,gg]
    
    M     = FIDToSpec(tmp,axis=0)
    S     = np.exp(-1j*(phi0+phi1*nu)) * (M@con[:,None])

    # add baseline
    if B is not None:
        S += B@b[:,None]
    
    return S.flatten()

def FSLModel_err_Voigt(x,nu,t,m,B,G,g,data,first,last):
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
    pred = FSLModel_forward_Voigt(x,nu,t,m,B,G,g)
    err  = data[first:last]-pred[first:last]
    sse  = np.real(np.sum(err*np.conj(err)))
    return sse


def FSLModel_grad_Voigt(x,nu,t,m,B,G,g,data,first,last):
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
    n     = m.shape[1]    # get number of basis functions
    #g     = max(G)+1       # get number of metabolite groups

    con,gamma,sigma,eps,phi0,phi1,b = FSLModel_x2param_Voigt(x,n,g)

    # Start 
    E = np.zeros((m.shape[0],g),dtype=np.complex)
    SIG = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg]+t*sigma[gg]**2)*t).flatten()
        SIG[:,gg] = sigma[gg]
    
    e_term   = np.zeros(m.shape,dtype=np.complex)
    sig_term   = np.zeros(m.shape,dtype=np.complex)
    c        = np.zeros((con.size,g))
    for i,gg in enumerate(G):
        e_term[:,i] = E[:,gg]
        sig_term[:,i] = SIG[:,gg]
        c[i,gg] = con[i]
    m_term = m*e_term
    
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    Fmet     = FIDToSpec(m_term)
    Ftmet    = FIDToSpec(t*m_term)
    Ft2sigmet   = FIDToSpec(t*t*sig_term*m_term)
    Ftmetc   = Ftmet@c
    Ft2sigmetc  = Ft2sigmet@c
    Fmetcon = Fmet@con[:,None]
    
    Spec     = data[first:last,None]

    # Forward model 
    S     = (phi_term*Fmetcon)
    if B is not None:
        S += B@b[:,None]
        
    # Gradients        
    dSdc     = phi_term*Fmet
    dSdgamma = phi_term*(-Ftmetc)
    dSdsigma = phi_term*(-2*Ft2sigmetc)
    dSdeps   = phi_term*(-1j*Ftmetc)
    dSdphi0  = -1j*phi_term*(Fmetcon)
    dSdphi1  = -1j*nu*phi_term*(Fmetcon)
    dSdb     = B
    
    # Only compute within a range            
    S         = S[first:last]
    dSdc      = dSdc[first:last,:]
    dSdgamma  = dSdgamma[first:last,:]
    dSdsigma  = dSdsigma[first:last,:]
    dSdeps    = dSdeps[first:last,:]
    dSdphi0   = dSdphi0[first:last]
    dSdphi1   = dSdphi1[first:last]
    dSdb      = dSdb[first:last]

    dS  = np.concatenate((dSdc,dSdgamma,dSdsigma,dSdeps,dSdphi0,dSdphi1,dSdb),axis=1)

    grad = np.real(np.sum(S*np.conj(dS)+np.conj(S)*dS - np.conj(Spec)*dS - Spec*np.conj(dS),axis=0))
    
    return grad


def getModelFunctions(model):
    """ Return the err, grad, forward and conversion functions appropriate for the model."""
    if model == 'lorentzian':
        err_func   = FSLModel_err          # error function
        grad_func  = FSLModel_grad         # gradient
        forward    = FSLModel_forward      # forward model
        x2p        = FSLModel_x2param
        p2x        = FSLModel_param2x            
    elif model == 'voigt':
        err_func   = FSLModel_err_Voigt     # error function
        grad_func  = FSLModel_grad_Voigt    # gradient
        forward    = FSLModel_forward_Voigt # forward model
        x2p        = FSLModel_x2param_Voigt
        p2x        = FSLModel_param2x_Voigt 
    else:
        raise Exception('Unknown model {}.'.format(model))
    return err_func,grad_func,forward,x2p,p2x

def getFittedModel(model,resParams,base_poly,metab_groups,mrs,basisSelect=None,baselineOnly = False,noBaseline = False):
    """ Return the predicted model given some fitting parameters
        
        model     (str)  : Model string
        resParams (array):
        base_poly
        metab_groups
        mrs   (class obj):
        
    """
    numBasis = len(mrs.names)
    numGroups = max(metab_groups)+1

    _,_,forward,x2p,p2x = getModelFunctions(model)

    if noBaseline:
        bp = np.zeros(base_poly.shape)
    else:
        bp = base_poly

    if basisSelect is None and not baselineOnly:
        return forward(resParams,
                       mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis,
                       bp,
                       metab_groups,
                       numGroups)
    elif baselineOnly:
        p = x2p(resParams,numBasis,numGroups)        
        p = (np.zeros(numBasis),)+p[1:]    
        xx  = p2x(*p)
        return forward(xx,
                       mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis,
                       bp,
                       metab_groups,
                       numGroups)
    elif basisSelect is not None:
        p = x2p(resParams,numBasis,numGroups)
        tmp = np.zeros(numBasis)
        basisIdx = mrs.names.index(basisSelect)
        tmp[basisIdx] = p[0][basisIdx]
        p = (tmp,)+p[1:]             
        xx  = p2x(*p)
        return forward(xx,
                       mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis,
                       bp,
                       metab_groups,
                       numGroups)




# Utils for VB implementation
# Only needs forward model
# does exponentiate positive parameters (i.e. they are in log-transform)
# ensures prediction is real by concatenating real and imag signals

# ########### For VB
# Exponentiate positive params
def FSLModel_forward_vb(x,nu,t,m,B,G,g,first,last):
    n     = m.shape[1]    # get number of basis functions

    logcon,loggamma,eps,phi0,phi1,b = FSLModel_x2param(x,n,g)
    con   = np.exp(logcon)
    gamma = np.exp(loggamma)
    
    E = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg])*t).flatten()
    
    tmp = np.zeros(m.shape,dtype=np.complex)
    for i,gg in enumerate(G):
        tmp[:,i] = m[:,i]*E[:,gg]
    
    M     = FIDToSpec(tmp)
    S     = np.exp(-1j*(phi0+phi1*nu)) * (M@con[:,None])

    # add baseline
    if B is not None:                
        S += B@b[:,None]
    
    S = S.flatten()[first:last]

    return np.concatenate((np.real(S),np.imag(S)))

# Gradient of the forward model (not the error)
# !!! grad wrt logparam (for those that are logged)
#  dfdlogx = x*dfdx
def FSLModel_grad_vb(x,nu,t,m,B,G,g,first,last):
    n     = m.shape[1]    # get number of basis functions

    logcon,loggamma,eps,phi0,phi1,b = FSLModel_x2param(x,n,g)
    con   = np.exp(logcon)
    gamma = np.exp(loggamma)
    
    # Start 
    E = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg])*t).flatten()
    
    e_term   = np.zeros(m.shape,dtype=np.complex)
    c        = np.zeros((con.size,g))
    for i,gg in enumerate(G):
        e_term[:,i] = E[:,gg]
        c[i,gg] = con[i]
    m_term = m*e_term
    
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    
    Fmet     = FIDToSpec(m_term)
    Ftmet    = FIDToSpec(t*m_term)
    Ftmetc   = Ftmet@c
    Fmetcon = Fmet@con[:,None]
    
    # Forward model 
    S     = (phi_term*Fmetcon)
    if B is not None:
        S += B@b[:,None]
        
    # Gradients        
    dSdc     = phi_term*Fmet
    dSdgamma = phi_term*(-Ftmetc)
    dSdeps   = phi_term*(-1j*Ftmetc)
    dSdphi0  = -1j*phi_term*(Fmetcon)
    dSdphi1  = -1j*nu*phi_term*(Fmetcon)
    dSdb     = B
    
    # Only compute within a range            
    dSdc      = dSdc[first:last,:]
    dSdgamma  = dSdgamma[first:last,:]
    dSdeps    = dSdeps[first:last,:]
    dSdphi0   = dSdphi0[first:last]
    dSdphi1   = dSdphi1[first:last]
    dSdb      = dSdb[first:last]

    dS  = np.concatenate((dSdc*con[None,:],
                          dSdgamma*gamma[None,:],
                          dSdeps,
                          dSdphi0,
                          dSdphi1,
                          dSdb),axis=1)

    dS = np.concatenate((np.real(dS),np.imag(dS)),axis=0)
    
    return dS



def FSLModel_forward_vb_voigt(x,nu,t,m,B,G,g,first,last):
    n     = m.shape[1]    # get number of basis functions

    logcon,loggamma,logsigma,eps,phi0,phi1,b = FSLModel_x2param_Voigt(x,n,g)
    con   = np.exp(logcon)
    gamma = np.exp(loggamma)
    sigma = np.exp(logsigma)

    E = np.zeros((m.shape[0],g),dtype=np.complex)
    for gg in range(g):
        E[:,gg] = np.exp(-(1j*eps[gg]+gamma[gg]+t*sigma[gg]**2)*t).flatten()

    tmp = np.zeros(m.shape,dtype=np.complex)
    for i,gg in enumerate(G):
        tmp[:,i] = m[:,i]*E[:,gg]
    
    M     = FIDToSpec(tmp)
    S     = np.exp(-1j*(phi0+phi1*nu)) * (M@con[:,None])

    # add baseline
    if B is not None:                
        S += B@b[:,None]
    
    S = S.flatten()[first:last]

    return np.concatenate((np.real(S),np.imag(S)))
