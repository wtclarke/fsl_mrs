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
def LCModel_forward(x,n,nu,t,m):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1]
    """
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    M    = np.fft.fft(m*np.exp(-(gamma+1j*eps)*t),axis=0)
    Y_nu = np.exp(-1j*(phi0+phi1*nu)) * (M @ con[:,None])
    Y_t  = np.fft.ifft(Y_nu,axis=0)
    
    return Y_t.flatten()

def LCModel_jac(x,n,nu,t,m,FID):
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    
    m_term   = m*np.exp(-(gamma+1j*eps)*t)
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    
    Fmet  = np.fft.fft(m_term,axis=0)
    cFmet = Fmet@con[:,None]
    
    Y        = LCModel_forward(x,n,nu,t,m)
    Y        = Y[:,None]
    dYdc     = np.fft.ifft(phi_term*Fmet,axis=0)
    dYdgamma = np.fft.ifft(phi_term*(np.fft.fft(   -t*m_term,axis=0)@con[:,None]),axis=0)
    dYdeps   = np.fft.ifft(phi_term*(np.fft.fft(-1j*t*m_term,axis=0)@con[:,None]),axis=0)
    dYdphi0  = np.fft.ifft(-1j*   phi_term*cFmet,axis=0)
    dYdphi1  = np.fft.ifft(-1j*nu*phi_term*cFmet,axis=0)    
    
    dY  = np.concatenate((dYdc,dYdgamma,dYdeps,dYdphi0,dYdphi1),axis=1)

    jac = np.real(np.sum(Y*np.conj(dY)+np.conj(Y)*dY - np.conj(FID[:,None])*dY - FID[:,None]*np.conj(dY),axis=0))
    
    return jac

def LCModel_err(x,n,nu,t,m,y):
    pred = LCModel_forward(x,n,nu,t,m)
    return np.sum(np.absolute(y-pred)**2)


def LCModel_approxCovariance(x,n,nu,t,m,FID):
    noise   = error_fun(x,n,nu,t,m,y)    
    J     = LCModel_jac(x,n,nu,t,m,FID)
    H     = J[:,None]@J[:,None].T
    C     = np.diag(np.linalg.inverse(H))*noise
    return C
    
################ FREQUENCY DOMAIN FUNCTIONS
def LCModel_forward_freq(x,n,nu,t,m):
    """
    x = [con[0],...,con[n-1],gamma,eps,phi0,phi1]
    """
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    M    = np.fft.fft(m*np.exp(-(gamma+1j*eps)*t),axis=0)
    Y_nu = np.exp(-1j*(phi0+phi1*nu)) * (M @ con[:,None])
    #Y_t  = np.fft.ifft(Y_nu,axis=0)
    
    return Y_nu.flatten()

def LCModel_jac_freq(x,n,nu,t,m,Spec):
    con   = x[:n] 
    gamma = x[n] 
    eps   = x[n+1] 
    phi0  = x[n+2]  
    phi1  = x[n+3] 
    
    m_term   = m*np.exp(-(gamma+1j*eps)*t)
    phi_term = np.exp(-1j*(phi0+phi1*nu)) 
    
    Fmet  = np.fft.fft(m_term,axis=0)
    cFmet = Fmet@con[:,None]
    
    Y        = LCModel_forward_freq(x,n,nu,t,m)
    Y        = Y[:,None]
    dYdc     = phi_term*Fmet
    dYdgamma = phi_term*(np.fft.fft(   -t*m_term,axis=0)@con[:,None])
    dYdeps   = phi_term*(np.fft.fft(-1j*t*m_term,axis=0)@con[:,None])
    dYdphi0  = -1j*   phi_term*cFmet
    dYdphi1  = -1j*nu*phi_term*cFmet   
    
    dY  = np.concatenate((dYdc,dYdgamma,dYdeps,dYdphi0,dYdphi1),axis=1)

    jac = np.real(np.sum(Y*np.conj(dY)+np.conj(Y)*dY - np.conj(Spec[:,None])*dY - Spec[:,None]*np.conj(dY),axis=0))
    
    return jac

def LCModel_err_freq(x,n,nu,t,m,y):
    pred = LCModel_forward_freq(x,n,nu,t,m)
    return np.sum(np.absolute(y-pred)**2)


    
