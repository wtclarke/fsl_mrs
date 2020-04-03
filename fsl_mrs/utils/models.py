#!/usr/bin/env python

# models.py - MRS forward models and helper functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.utils.misc import FIDToSpec,SpecToFID

# Helper functions for LCModel fitting

# faster than utils.misc.FIDtoSpec
#def FIDToSpec(FID):
#    return np.fft.fft(FID,axis=0)

#def SpecToFID(FID):
#    return np.fft.ifft(FID,axis=0)



################ TIME DOMAIN FUNCTIONS
def LCModel_forward(x,nu,t,m):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1]
    """
    n     = m.shape[1]
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    M    = np.fft.fft(m*np.exp(-(gamma+1j*eps)*t),axis=0)
    Y_nu = np.exp(-1j*(phi0+phi1*nu)) * (M @ con[:,None])
    Y_t  = np.fft.ifft(Y_nu,axis=0)
    
    return Y_t.flatten()

def LCModel_jac(x,nu,t,m,FID,first=None,last=None):
    n     = m.shape[1]
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    
    m_term   = m*np.exp(-(gamma+1j*eps)*t)
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    
    Fmet  = np.fft.fft(m_term,axis=0)
    cFmet = Fmet@con[:,None]
    
    Y        = LCModel_forward(x,nu,t,m)
    Y        = Y[:,None]
    dYdc     = np.fft.ifft(phi_term*Fmet,axis=0)
    dYdgamma = np.fft.ifft(phi_term*(np.fft.fft(   -t*m_term,axis=0)@con[:,None]),axis=0)
    dYdeps   = np.fft.ifft(phi_term*(np.fft.fft(-1j*t*m_term,axis=0)@con[:,None]),axis=0)
    dYdphi0  = np.fft.ifft(-1j*   phi_term*cFmet,axis=0)
    dYdphi1  = np.fft.ifft(-1j*nu*phi_term*cFmet,axis=0)    
    
    dY  = np.concatenate((dYdc,dYdgamma,dYdeps,dYdphi0,dYdphi1),axis=1)

    jac = np.real(np.sum(Y*np.conj(dY)+np.conj(Y)*dY - np.conj(FID[:,None])*dY - FID[:,None]*np.conj(dY),axis=0))
    
    return jac

def LCModel_err(x,nu,t,m,y,first,last):
    pred = LCModel_forward(x,nu,t,m)
    return np.sum(np.absolute(y[first:last]-pred[first:last])**2)


def LCModel_approxCovariance(x,n,nu,t,m,FID):
    noise   = error_fun(x,n,nu,t,m,y)    
    J     = LCModel_jac(x,n,nu,t,m,FID)
    H     = J[:,None]@J[:,None].T
    C     = np.diag(np.linalg.inverse(H))*noise
    return C
    
################ FREQUENCY DOMAIN FUNCTIONS
def LCModel_forward_freq(x,nu,t,m):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1]
    """
    n     = m.shape[1]
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    M    = FIDToSpec(m*np.exp(-(gamma+1j*eps)*t))
    Y_nu = np.exp(-1j*(phi0+phi1*nu)) * (M@con[:,None])
    
    return Y_nu.flatten()

def LCModel_jac_freq(x,nu,t,m,Spec,first,last):
    n     = m.shape[1]
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    
    m_term   = m*np.exp(-(gamma+1j*eps)*t)    
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    
    Fmet  = FIDToSpec(m_term)
    Ftmet = FIDToSpec(t*m_term)
    cFmet = Fmet@con[:,None]
    
    Y        = LCModel_forward_freq(x,nu,t,m)
    dYdc     = phi_term*Fmet
    dYdgamma = phi_term*(-Ftmet@con[:,None])
    dYdeps   = phi_term*(-1j*Ftmet@con[:,None])
    dYdphi0  = -1j*   phi_term*cFmet
    dYdphi1  = -1j*nu*phi_term*cFmet   

    # Only compute within a range
    Y         = Y[first:last,None]
    Spec      = Spec[first:last,None]
    dYdc      = dYdc[first:last,:]
    dYdgamma  = dYdgamma[first:last]
    dYdeps    = dYdeps[first:last]
    dYdphi0   = dYdphi0[first:last]
    dYdphi1   = dYdphi1[first:last]
    
    dY  = np.concatenate((dYdc,dYdgamma,dYdeps,dYdphi0,dYdphi1),axis=1)

    jac = np.real(np.sum(Y*np.conj(dY)+np.conj(Y)*dY - np.conj(Spec)*dY - Spec*np.conj(dY),axis=0))

    # bit quicker?

    #err = Y-Spec 
    #jac = np.sum(dY*np.conj(err) + np.conj(dY)*err,axis=0)

    return jac #np.real(jac)

def LCModel_err_freq(x,nu,t,m,data,first,last):
    pred = LCModel_forward_freq(x,nu,t,m)                                         
    err  = data[first:last]-pred[first:last]
    sse  = np.real(np.sum(err*np.conj(err))) 
    return sse

    


# ##################### FSL MODEL
# Modifications on LCModel to be listed here
#

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
    
    tmp = np.zeros(m.shape,dtype=np.complex)
    for i,gg in enumerate(G):
        tmp[:,i] = m[:,i]*E[:,gg]
    
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
    
    dS  = np.concatenate((dSdc,dSdgamma,dSdeps,dSdphi0,dSdphi1,dSdb),axis=1)

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

