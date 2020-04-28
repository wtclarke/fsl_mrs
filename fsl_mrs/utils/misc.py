#!/usr/bin/env python

# misc.py - Various utils
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
import scipy.fft
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
import itertools as it
from copy import deepcopy

from .constants import H2O_PPM_TO_TMS 


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

def FIDToSpec(FID,axis=0):
    """ Convert FID to spectrum
    
        Performs fft along indicated axis
        Args:
            FID (np.array)      : array of FIDs
            axis (int,optional) : time domain axis

        Returns:
            x (np.array)        : array of spectra
    """
    # By convention the first point of the fid is special cased   
    FID[0] *=0.5
    out = scipy.fft.fftshift(scipy.fft.fft(FID,axis=axis,norm='ortho'),axes=axis)
    FID[0] *=2
    return out

def SpecToFID(spec,axis=0):
    """ Convert spectrum to FID
    
        Performs fft along indicated axis
        Args:
            spec (np.array)     : array of spectra
            axis (int,optional) : freq domain axis

        Returns:
            x (np.array)        : array of FIDs
    """    
    fid = scipy.fft.ifft(scipy.fft.ifftshift(spec,axes=axis),axis=axis,norm='ortho')
    fid[0] *= 2
    return fid

def calculateAxes(bandwidth,centralFrequency,points):
    dwellTime = 1/bandwidth
    timeAxis         = np.linspace(dwellTime,
                                    dwellTime*points,
                                    points)  
    frequencyAxis    = np.linspace(-bandwidth/2,
                                    bandwidth/2,
                                    points)        
    ppmAxis          = hz2ppm(centralFrequency,
                                    frequencyAxis,shift=False)
    ppmAxisShift     = hz2ppm(centralFrequency,
                                    frequencyAxis,shift=True)

    return {'time':timeAxis,'freq':frequencyAxis,'ppm':ppmAxis,'ppmshift':ppmAxisShift}

def checkCFUnits(cf,units='Hz'):
    """ Check the units of central frequency and adjust if required."""
    # Assume cf in Hz > 1E5, if it isn't assume that user has passed in MHz
    if cf<1E5:
        if units.lower()=='hz':
            cf *= 1E6
        elif units.lower()=='mhz':
            pass
        else:
            raise ValueError('Only Hz or MHz defined')
    else:
        if units.lower()=='hz':
            pass
        elif units.lower()=='mhz':
            cf /= 1E6
        else:
            raise ValueError('Only Hz or MHz defined')
    return cf

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

    #first,last = mrs.ppmlim_to_range(ppmlim)
    #f1,f2 = np.abs(mrs.frequencyAxis[first]),np.abs(mrs.frequencyAxis[last])
    #if f1>f2:
    #    f1,f2=f2,f1
    #wn = [f1/nyq,f2/nyq]

    f1 = np.abs(ppm2hz(mrs.centralFrequency,ppmlim[0])/ nyq)
    f2 = np.abs(ppm2hz(mrs.centralFrequency,ppmlim[1])/ nyq)

    if f1>f2:
        f1,f2=f2,f1
    wn = [f1,f2]

    #print(wn)
    
    order = 6
    
    b,a = butter(order, wn, btype=filter_type)
    y = lfilter(b, a, FID)
    return y


def resample_ts(ts,dwell,new_dwell):
    """
      Temporal resampling

      Parameters
      ----------

     ts : matrix (rows=time axis)
     
     dwell and new_dwell : float 

      Returns
      -------

      matrix

    """
    numPoints = ts.shape[0]
    
    t      = np.linspace(dwell,dwell*numPoints,numPoints)-dwell
    new_t  = np.linspace(new_dwell,new_dwell*numPoints,numPoints)-new_dwell
    
    f      = interp1d(t,ts,axis=0)
    new_ts = f(new_t)
        
    return new_ts

def ts_to_ts(old_ts,old_dt,new_dt,new_n):
    """
    Temporal resampling where the new time series has a smaller number of points
    """
    old_n    = old_ts.shape[0]    
    old_t    = np.linspace(old_dt,old_dt*old_n,old_n)-old_dt
    new_t    = np.linspace(new_dt,new_dt*new_n,new_n)-new_dt
    
    f      = interp1d(old_t,old_ts,axis=0)
    new_ts = f(new_t)
    
    return new_ts
        

# Numerical differentiation (light)
import numpy as np
#Gradient Function
def gradient(x, f):
    """
      Calculate f'(x): the numerical gradient of a function

      Parameters:
      -----------
      x : array-like 
      f : scalar function

      Returns:
      --------
      array-like
    """
    N = len(x)
    gradient = []
    for i in range(N):
        eps = abs(x[i]) *  np.finfo(np.float32).eps 
        if eps==0: 
            eps = 1e-5
        xl = np.array(x)
        xu = np.array(x)
        xl[i] -= eps
        xu[i] += eps
        fl = f(xl)
        fu = f(xu)        
        gradient.append((fu-fl)/(2*eps))
    return np.array(gradient)


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
    N = len(x)
    hessian = []
    gd_0 = gradient( x, f)
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps 
    if eps==0: 
        eps = 1e-5
    for i in range(N):
        hessian.append([])
        xx0 = 1.*x[i]
        x[i] = xx0 + eps
        gd_1 =  gradient(x, f)
        for j in range(N):
            hessian[i].append((gd_1[j,:] - gd_0[j,:])/eps)
        x[i] =xx0
    return np.asarray(hessian)

def hessian_diag(x,f):
    """
       Calculate numerical second order derivative of f at x
       (the diagonal of the Hessian)
       
       Parameters:
       -----------
       x : array-like
       f : function

       Returns:
       --------
       array-like
    """
    N = x.size
    hess = np.zeros((N,1)) 
    gd_0 = gradient( x, f)    
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps

    if eps==0: 
        eps = 1e-5
    for i in range(N):
        xx0 = 1.*x[i]
        x[i] = xx0 + eps
        gd_1 =  gradient(x, f)
        hess[i] = ((gd_1[i] - gd_0[i])/eps)
        x[i] =xx0

    return hess



# Little bit of code for checking the gradients
def check_gradients():
    m = np.linspace(0,10,100)
    cf = lambda p : np.sum(p[0]*np.exp(-p[1]*m))
    x0 = np.random.randn(2)*.1
    grad_num = gradient(x0,cf)
    E = lambda x : np.sum(np.exp(-x[1]*m))
    grad_anal = np.array([E(x0),-x0[0]*np.sum(m*np.exp(-x0[1]*m))])
    hess_anal = np.zeros((2,2))
    hess_anal[0,1] = -np.sum(m*np.exp(-x0[1]*m))
    hess_anal[1,0] = -np.sum(m*np.exp(-x0[1]*m))
    hess_anal[1,1] = x0[0]*np.sum(m**2*np.exp(-x0[1]*m))
    hess_num = hessian(x0,cf)
    hess_diag = hessian_diag(x0,cf)
    print('x0 = {}, f(x0)  = {}'.format(x0,cf(x0)))
    print('Grad Analytic   : {}'.format(grad_anal))
    print('Grad Numerical  : {}'.format(grad_num))
    print('Hess Analytic   : {}'.format(hess_anal))
    print('Hess Numreical  : {}'.format(hess_num))
    print('Hess Diag       : {}'.format(hess_diag))
    

def calculate_crlb(x,f,data):
    """
       Calculate Cramer-Rao Lower Bound
       This assumes a model of the form data = f(x) + noise
       where noise ~ N(0,sig^2)
       In which case the CRLB is sum( |f'(x)|^2 )/sig^2
       It uses numerical differentiation to get f'(x)

      Parameters:
       x : array-like
       f : function
       data : array-like

      Returns:
        array-like
    """
    # estimate noise variance empirically
    sig2 = np.var(data-f(x))
    grad = gradient(x,f)        
    crlb = 1/(np.sum(np.abs(grad)**2,axis=1)/sig2)
    
    return crlb

def calculate_lap_cov(x,f,data,sig2=None,method='fisher'):
    """
      Calculate approximate covariance using
      Fisher information matrix

      Assumes forward model is data=f(x)+N(0,sig^2)
      

      Parameters:
       x : array-like
       f : function
       data : array-like
       sig2 : optional noise variance 
       method : 'fisher' or 'hessian'

      Returns:
        2D array    
    """
    x = np.asarray(x)
    N = x.size
    C = np.zeros((N,N)) # covariance
    if sig2 is None:        
        sig2 = np.var(data-f(x))
    grad = gradient(x,f)
    
    # if method == 'hessian':
    #     hess = hessian(x,f)
    #     err  = data-f(x)
    # for i in range(N):
    #     gi = grad[i]
    #     for j in range(N):
    #         gj = grad[j]
    #         gigj = np.abs(gi*np.conj(gj)+np.conj(gi)*gj)
    #         if method == 'hessian':                
    #             C[i,j] = np.sum(gigj + 2*err*hess[i,j])
    #         else:
    #             C[i,j] = np.sum(gigj)
            
    # C = np.linalg.pinv(C/2/sig2)

    J = np.concatenate((np.real(grad),np.imag(grad)),axis=1)
    P0 = np.diag(np.ones(N)*1E-5)
    P = np.dot(J,J.transpose()) / sig2
    C = np.linalg.inv(P+P0)

    
    return C

def calculate_lap_cov_with_grad(x,f,df,args,sig2=None):
    grad = df(x,*args)
    if sig2 is None:        
        sig2 = np.var(data-f(x,*args))
    J = np.concatenate((np.real(grad),np.imag(grad)),axis=1)
    P0 = np.diag(np.ones(N)*1E-5)
    P = np.dot(J,J.transpose()) / sig2
    C = np.linalg.inv(P+P0)
    
    return C


# Various utilities
def multiply(x,y):
    """
     Elementwise multiply numpy arrays x and y 
     
     Returns same shape as x
    """
    shape = x.shape
    r = x.flatten()*y.flatten()
    return np.reshape(r,shape)

def shift_FID(mrs,FID,eps):
    """
       Shift FID in spectral domain

    Parameters:
       mrs : MRS object
       FID : array-like
       eps : shift factor (Hz)

    Returns:
       array-like
    """
    t           = mrs.timeAxis
    FID_shifted = multiply(FID,np.exp(-1j*2*np.pi*t*eps))
    
    return FID_shifted
 
def blur_FID(mrs,FID,gamma):
    """
       Blur FID in spectral domain

    Parameters:
       mrs   : MRS object
       FID   : array-like
       gamma : blur factor in Hz

    Returns:
       array-like
    """
    t           = mrs.timeAxis
    FID_blurred = multiply(FID,np.exp(-t*gamma))
    return FID_blurred

def blur_FID_Voigt(mrs,FID,gamma,sigma):
    """
       Blur FID in spectral domain

    Parameters:
       mrs   : MRS object
       FID   : array-like
       gamma : Lorentzian line broadening 
       sigma : Gaussian line broadening 

    Returns:
       array-like
    """
    t = mrs.timeAxis
    FID_blurred = multiply(FID,np.exp(-t*(gamma+t*sigma**2/2)))
    return FID_blurred

def rescale_FID(x,scale=100):
    """
    Useful for ensuring values are within nice range

    Forces norm of 1D arrays to be = scale
    Forces norm of column-mean of 2D arrays to be = scale (i.e. preserves relative norms of the columns)
    
    Parameters
    ----------
    x : 1D or 2D array
    scale : float            
    """
    
    y =  x.copy()

    if type(y) is list:
        factor = np.linalg.norm(sum(y)/len(y))
        return [yy/factor*scale for yy in y],1/factor * scale
    
    if y.ndim == 1:
        factor = np.linalg.norm(y)
    else:
        factor = np.linalg.norm(np.mean(y,axis=1),axis=0)        
    y =  y / factor * scale
    return y,1/factor * scale


def create_peak(mrs,ppm,gamma=0,sigma=0):
    """
        creates FID for peak at specific ppm
        
    Parameters
    ----------
    mrs : MRS object (contains time information)
    ppm : float
    gamma : float
            Peak Lorentzian dispersion
    sigma : float
            Peak Gaussian dispersion
    
    Returns
    -------
    array-like FID
    """
    
    freq = ppm2hz(mrs.centralFrequency,ppm)
    t    = mrs.timeAxis 
    x    = np.exp(-1j*2*np.pi*freq*t).flatten()
    
    if gamma>0 or sigma>0:
        x = blur_FID_Voigt(mrs,x,gamma,sigma)

    # dephase
    x = x*np.exp(-1j*np.angle(x[0]))
    
    return x

def extract_spectrum(mrs,FID,ppmlim=(0.2,4.2),shift=True):
    """
       Extracts spectral interval
    
    Parameters:
       mrs : MRS object
       FID : array-like
       ppmlim : tuple
       
    Returns:
       array-like
    """
    spec        = FIDToSpec(FID)
    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim,shift=shift)
    spec        = spec[first:last]
    
    return spec
       
def normalise(x,axis=0):
    """
       Devides x by norm of x
    """
    return x/np.linalg.norm(x,axis=axis)

def ztransform(x,axis=0):
    """
       Demeans x and make norm(x)=1
    """
    return (x-np.mean(x,axis=axis))/np.std(x,axis)/np.sqrt(x.size)
    
def correlate(x,y):
    """
       Computes correlation between complex signals x and y
       Uses formula : sum( conj(z(x))*z(y)) where z() is the ztransform
    """
    return np.real(np.sum(np.conjugate(ztransform(x))*ztransform(y)))

def phase_correct(mrs,FID,ppmlim=(1,3)):
    """
       Apply phase correction to FID
    """
    first,last = mrs.ppmlim_to_range(ppmlim)
    phases = np.linspace(0,2*np.pi,1000)
    x = []
    for phase in phases:
        f = np.real(np.fft.fft(FID*np.exp(1j*phase),axis=0))
        x.append(np.sum(f[first:last]<0))
    phase = phases[np.argmin(x)]
    return FID*np.exp(1j*phase)    




def detrend(data,deg=1,keep_mean=True):
    """
    remove polynomial trend from data
    works along first dimension
    """
    n = data.shape[0]
    x = np.arange(n)
    M = np.zeros((n,deg+1))
    for i in range(deg+1):        
        M[:,i] = x**i

    beta = np.linalg.pinv(M) @ data

    pred = M @ beta
    m = 0
    if keep_mean:
        m = np.mean(data,axis=0)
    return data - pred + m


def regress_out(x,conf,keep_mean=True):
    """
    Linear deconfounding
    """
    if type(conf) is list:
        confa = np.squeeze(np.asarray(conf)).T
    else:
        confa = conf
    if keep_mean:
        m = np.mean(x,axis=0)
    else:
        m = 0
    return x - confa@(np.linalg.pinv(confa)@x) + m



# ----- MRSI stuff ---- #
def volume_to_list(data,mask):
    """
       Turn voxels within mask into list of data

    Parameters
    ----------
    
    data : 4D array

    mask : 3D array

    Returns
    -------

    list

    """
    nx, ny, nz = data.shape[:3]
    voxels = []
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x, y, z]:
            voxels.append((x, y, z))       
    voxdata = [data[x, y, z, :] for (x, y, z) in voxels]
    return voxdata

def list_to_volume(data_list,mask,dtype=float):
    """
       Turn list of voxelwise data into 4D volume

    Parameters
    ----------
    voxdata : list
    mask    : 3D volume
    dtype   : force output data type

    Returns
    -------
    4D or 3D volume
    """

    nx,ny,nz = mask.shape
    nt       = data_list[0].size
    if nt>1:
        data     = np.zeros((nx,ny,nz,nt),dtype=dtype)
    else:
        data     = np.zeros((nx,ny,nz,),dtype=dtype)
    i=0
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x, y, z]:            
            if nt>1:
                data[x, y, z, :] = data_list[i]
            else:
                data[x, y, z] = data_list[i]
            i+=1

    return data

def unravel(idx,mask):
    nx,ny,nz=mask.shape
    counter = 0
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x,y,z]:
            if counter==idx:
                return np.array([x,y,z])
            counter +=1


def ravel(arr,mask):
    nx,ny,nz=mask.shape
    counter = 0
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x,y,z]:
            if arr==[x,y,z]:
                return counter
            counter += 1
    


#### FMRS Stuff

def smooth_FIDs(FIDlist,window):
    """
    Smooth a list of FIDs (makes sense if acquired one after the other as the smoothing is done along the "time" dimension

    Note: at the edge of the list of FIDs the smoothing wraps around the list so make sure that the beginning and the end are 'compatible'

    Parameters:
    -----------
    FIDlist : list of FIDs
    window  : int (preferably odd number)

    Returns:
    --------
    list of FIDs
    """
    sFIDlist = []
    for idx,FID in enumerate(FIDlist):
        fid = 0
        n   = 0
        for i in range(-int(window/2),int(window/2)+1,1):
            fid = fid + FIDlist[(idx+i)%len(FIDlist)]
            n   = n+1
        fid = fid/n
        sFIDlist.append(fid)
    return sFIDlist
