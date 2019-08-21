#!/usr/bin/env python

# misc.py - Various utils
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from scipy.signal import butter, lfilter

H2O_PPM_TO_TMS = 4.65  # Shift of water to Tetramethylsilane


# Convention:
#  freq in Hz
#  ppm = freq/1e6
#  ppm_shift = ppm - 4.65
#  why is there a minus sign here? 

def ppm2hz(cf,ppm,shift=True):
    if shift:
        return (ppm-H2O_PPM_TO_TMS)*cf*1E-6
    else:
        return (ppm)*cf*1E-6

def hz2ppm(cf,hz,shift=True):
    if shift:
        return 1E6 *hz/cf + H2O_PPM_TO_TMS
    else:
        return 1E6 *hz/cf

def filter(mrs,FID,ppmlim,filter_type='bandpass'):
    """
       Filter in/out frequencies defined in ppm
       
       Parameters
       ----------
       mrs    : MRS Object
       FID    : array-like
              temporal signal to filter
       ppmlim : float or tuple              
       filter_type: {'lowpass','highpass','bandpass', 'bandstop'}
              default type is 'bandstop')

       Outputs
       -------
       numpy array 
    """
    
    # Sampling frequency (Hz)
    fs     = 1/mrs.dwellTime
    nyq    = 0.5 * fs
    
    wn = [ppm2hz(mrs.centralFrequency,ppmlim[0])/ nyq,
          ppm2hz(mrs.centralFrequency,ppmlim[1])/ nyq]     
    
    order = 6
    
    b,a = butter(order, wn, btype=filter_type)
    y = lfilter(b, a, FID)
    return y


# Numerical differentiation (light)
import numpy as np
#Gradient Function
def gradient(x, f):
    """
      Calculate f'(x): the numerical gradient of a function

      Parameters:
      -----------
      x : array-like 
      f : function

      Returns:
      --------
      array-like
    """
    x = x.astype(float)
    N = x.size
    gradient = []
    for i in range(N):
        eps = abs(x[i]) *  np.finfo(np.float32).eps 
        if eps==0: 
            eps = 1e-5
        xx0 = 1. * x[i]
        f0 = f(x)
        x[i] = x[i] + eps
        f1 = f(x)
        #gradient.append(np.asscalar(np.array([f1 - f0]))/eps)
        gradient.append((f1-f0)/eps)
        x[i] = xx0
    return np.array(gradient).reshape(x.shape)



#Hessian Matrix
def hessian (x, f):
    """
       Calculate numerical Hessian of f at x
       
       Parameters:
       -----------
       x : array-like
       f : function

       Returns:
       --------
       matrix
    """
    N = x.size
    hessian = np.zeros((N,N)) 
    gd_0 = gradient( x, f)
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps 
    if eps==0: 
        eps = 1e-5
    for i in range(N):
        xx0 = 1.*x[i]
        x[i] = xx0 + eps
        gd_1 =  gradient(x, f)
        hessian[:,i] = ((gd_1 - gd_0)/eps).reshape(x.shape[0])
        x[i] =xx0
    return hessian



# Little bit of code for checking the gradients
# m = np.linspace(0,10,100)
# cf = lambda p : np.sum(p[0]*np.exp(-p[1]*m))
# x0 = np.random.randn(2)*.1
# grad_num = misc.gradient(x0,cf)
# E = lambda x : np.sum(np.exp(-x[1]*m))
# grad_anal = np.array([E(x0),-x0[0]*np.sum(m*np.exp(-x0[1]*m))])
# hess_anal = np.zeros((2,2))
# hess_anal[0,1] = -np.sum(m*np.exp(-x0[1]*m))
# hess_anal[1,0] = -np.sum(m*np.exp(-x0[1]*m))
# hess_anal[1,1] = x0[0]*np.sum(m**2*np.exp(-x0[1]*m))
# hess_num = misc.hessian(x0,cf)
# print('x0 = {}, f(x0) = {}'.format(x0,cf(x0)))
# print('Grad Analytic : {}'.format(grad_anal))
# print('Grad Numrical : {}'.format(grad_num))
# print('Hess Analytic : {}'.format(hess_anal))
# print('Hess Numrical : {}'.format(hess_num))

