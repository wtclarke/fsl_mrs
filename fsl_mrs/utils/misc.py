#!/usr/bin/env python

# misc.py - Various utils
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
import itertools as it
from copy import deepcopy

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
    def scaleFID(x):
        x[0,...] *= 0.5
        return x

    # move indicated axis into first dimension
    # copy so we don't modify the first fid point
    FID = np.moveaxis(FID,axis,0).copy()
    x = np.fft.fftshift(np.fft.fft(scaleFID(FID),axis=0),axes=0)/FID.shape[0]
    x = np.moveaxis(x,0,axis)
    return x

def SpecToFID(spec,axis=0):
    """ Convert spectrum to FID
    
        Performs fft along indicated axis
        Args:
            spec (np.array)     : array of spectra
            axis (int,optional) : freq domain axis

        Returns:
            x (np.array)        : array of FIDs
    """
    def scaleFID(x):
        x[0] *= 2
        return x
    spec = np.moveaxis(spec,axis,0).copy()
    x = scaleFID(np.fft.ifft(np.fft.ifftshift(spec,axes=0),axis=0)*spec.shape[0])
    x = np.moveaxis(x,0,axis)
    return x

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

def calculate_lap_cov(x,f,data):
    """
       Calculate approximate covariance using
       Fisher information matrix

      Parameters:
       x : array-like
       f : function
       data : array-like

      Returns:
        2D array    
    """
    N = x.size
    C = np.zeros((N,N))

    sig2 = np.var(data-f(x))
    grad = gradient(x,f)
    for i in range(N):
        for j in range(N):
            fij = np.real(grad[i])*np.real(grad[j]) + np.imag(grad[i])*np.imag(grad[j])
            C[i,j] = np.sum(fij)/sig2

    C = np.linalg.pinv(C)
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
