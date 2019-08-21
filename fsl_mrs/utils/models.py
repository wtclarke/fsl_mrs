#!/usr/bin/env python

# models.py - MRS forward models and helper functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np


# Helper functions for LCModel fitting

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
    M    = np.fft.fft(m*np.exp(-(gamma+1j*eps)*t),axis=0)
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
    
    Fmet  = np.fft.fft(m_term,axis=0)
    Ftmet = np.fft.fft(t*m_term,axis=0)
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
def FSLModel_forward(x,nu,t,m,B):
    """
    x = [con[0],...,con[n-1],gamma,eps,delta,phi0,phi1,baselineparams]

    nu : array-like - frequency axis
    t  : array-like - time axis
    m  : basis time course
    B  : baseline functions
    """
    n     = m.shape[1]  # get number of basis functions
    con   = x[:n]       # concentrations
    gamma = x[n]        # lorentzian blurring
    eps   = x[n+1]      # frequency shift
    delta = x[n+2]      # gaussian blurring
    phi0  = x[n+3]      # global phase shift
    phi1  = x[n+4]      # global phase ramp
    b     = x[n+5:]     # baseline params
    
    M     = np.fft.fft(m*np.exp(-(1j*eps+gamma+delta/2*t)*t),axis=0)
    S     = np.exp(-1j*(phi0+phi1*nu)) * (M@con[:,None])

    # add baseline
    if B is not None:
        S     += np.matmul(B,b)
    
    return S

def FSLModel_err(x,nu,t,m,B,data,first,last):
    pred = FSLModel_forward(x,n,nu,t,m,B)
    err  = data[first:last]-pred[first:last]
    sse  = np.real(np.sum(err*np.conj(err)))
    return sse


def FSLModel_jac(x,nu,t,m,B,data,first,last):
    n     = m.shape[1]
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1]
    delta = x[n+2]
    phi0  = x[n+3]  
    phi1  = x[n+4]
    
    m_term   = m*np.exp(-(gamma+1j*eps+delta/2*t)*t)
    tm_term  = t*m_term
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    
    Fmet  = np.fft.fft(m_term,axis=0)
    Ftmet = np.fft.fft(tm_term,axis=0)
    cFmet = Fmet@con[:,None]
    
    S        = FSLModel_forward_freq(x,n,nu,t,m,B)
    Spec     = data[first:last,None]
    
    dSdc     = phi_term*Fmet
    dSdgamma = phi_term*(-Ftmet@con[:,None])
    dSdeps   = phi_term*(-1j*Ftmet@con[:,None])
    dSddelta = phi_term*np.matmul(np.fft.fft(-.5*t*tm_term,axis=0),con)
    dSdphi0  = -1j*phi_term*cFmet
    dSdphi1  = -1j*nu*phi_term*cFmet
    dSdb     = B

    
    # Only compute within a range
    dSdc      = dSdc[first:last,:]
    dSdgamma  = dSdgamma[first:last]    
    dSdeps    = dSdeps[first:last]
    dSddelta  = dSddelta[first:last]
    dSdphi0   = dSdphi0[first:last]
    dSdphi1   = dSdphi1[first:last]
    
    dS  = np.concatenate((dSdc,dSdgamma,dSdeps,dSddelta,dSdphi0,dSdphi1,dSdb),axis=1)

    jac = np.real(np.sum(S*np.conj(dS)+np.conj(S)*dS - np.conj(Spec)*dS - Spec*np.conj(dS),axis=0))

    return jac
