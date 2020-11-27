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
from scipy.optimize import minimize
from functools import partial

from .constants import H2O_PPM_TO_TMS


def ppm2hz(cf, ppm, shift=True, shift_amount=H2O_PPM_TO_TMS):
    if shift:
        return (ppm-shift_amount)*cf*1E-6
    else:
        return (ppm)*cf*1E-6


def hz2ppm(cf, hz, shift=True, shift_amount=H2O_PPM_TO_TMS):
    if shift:
        return 1E6 * hz/cf + shift_amount
    else:
        return 1E6 * hz/cf


def FIDToSpec(FID, axis=0):
    """ Convert FID to spectrum

        Performs fft along indicated axis
        Args:
            FID (np.array)      : array of FIDs
            axis (int,optional) : time domain axis

        Returns:
            x (np.array)        : array of spectra
    """
    # By convention the first point of the fid is special cased
    ss = [slice(None) for i in range(FID.ndim)]
    ss[axis] = slice(0, 1)
    ss = tuple(ss)
    FID[ss] *= 0.5
    out = scipy.fft.fftshift(scipy.fft.fft(FID,
                                     axis=axis,
                                     norm='ortho'),
                             axes=axis)
    FID[ss] *= 2
    return out


def SpecToFID(spec, axis=0):
    """ Convert spectrum to FID

        Performs fft along indicated axis
        Args:
            spec (np.array)     : array of spectra
            axis (int,optional) : freq domain axis

        Returns:
            x (np.array)        : array of FIDs
    """
    fid = scipy.fft.ifft(scipy.fft.ifftshift(spec,
                                            axes=axis),
                        axis=axis, norm='ortho')
    ss = [slice(None) for i in range(fid.ndim)]
    ss[axis] = slice(0, 1)
    ss = tuple(ss)
    fid[ss] *= 2
    return fid


def calculateAxes(bandwidth, centralFrequency, points, shift):
    dwellTime = 1/bandwidth
    timeAxis = np.linspace(dwellTime,
                           dwellTime * points,
                           points)
    frequencyAxis = np.linspace(-bandwidth/2,
                                bandwidth/2,
                                points)
    ppmAxis = hz2ppm(centralFrequency,
                     frequencyAxis,
                     shift=False)
    ppmAxisShift = hz2ppm(centralFrequency,
                          frequencyAxis,
                          shift=True,
                          shift_amount=shift)

    return {'time': timeAxis,
            'freq': frequencyAxis,
            'ppm': ppmAxis,
            'ppmshift': ppmAxisShift}


def checkCFUnits(cf, units='Hz'):
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

def calculate_lap_cov(x,f,data,sig2=None):
    """
      Calculate approximate covariance using
      Fisher information matrix

      Assumes forward model is data=f(x)+N(0,sig^2)
      

      Parameters:
       x : array-like
       f : function
       data : array-like
       sig2 : optional noise variance 

      Returns:
        2D array    
    """
    x = np.asarray(x)
    N = x.size
    if sig2 is None:        
        sig2 = np.var(data-f(x))
    grad = gradient(x,f)
    
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


def create_peak(mrs,ppm,amp,gamma=0,sigma=0):
    """
        creates FID for peak at specific ppm
        
    Parameters
    ----------
    mrs : MRS object (contains time information)
    ppm : list of floats
    amp : list of floats
    gamma : float
            Peak Lorentzian dispersion
    sigma : float
            Peak Gaussian dispersion
    
    Returns
    -------
    array-like FID
    """
    
    if isinstance(ppm,(float,int)):
        ppm = [float(ppm),]
    if isinstance(amp,(float,int)):
        amp = [float(amp),]

    t    = mrs.timeAxis
    out = np.zeros(t.shape[0],dtype=np.complex128)

    for p,a in zip(ppm,amp):
        freq = ppm2hz(mrs.centralFrequency,p)
         
        x    = a*np.exp(1j*2*np.pi*freq*t).flatten()
        
        if gamma>0 or sigma>0:
            x = blur_FID_Voigt(mrs,x,gamma,sigma)

        # dephase
        x = x*np.exp(-1j*np.angle(x[0]))
        out+= x

    return out

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



def parse_metab_groups(mrs,metab_groups):
    """
    Creates list of indices per metabolite group

    Parameters:
    -----------
    metab_groups :
       - A single index    : output is a list of 0's
       - A single string   : corresponding metab in own group 
       - The strings 'separate_all' or 'combine_all'
       - A list:
        - list of integers : output same as input
        - list of strings  : each string is interpreted as metab name and has own group
       Entries in the lists above can also be lists, in which case the corresponding metabs are grouped
    
    mrs : MRS Object

    Returns
    -------
    list of integers
    """
    if isinstance(metab_groups,list) and len(metab_groups)==1:
        metab_groups = metab_groups[0]
    
    out = [0]*mrs.numBasis
    
    if isinstance(metab_groups,int):
        return out

    if isinstance(metab_groups,str):
        if metab_groups.lower() == 'separate_all':
            return list(range(mrs.numBasis))
        
        if metab_groups.lower() == 'combine_all':
            return [0]*mrs.numBasis
            
        out = [0]*mrs.numBasis
        out[mrs.names.index(metab_groups)] = 1
        return out
    

    if isinstance(metab_groups,list):
        if isinstance(metab_groups[0],int):
            assert(len(metab_groups) == mrs.numBasis)
            return metab_groups
        
        grpcounter = 0
        for entry in metab_groups:
            if isinstance(entry,str):
                entry = entry.split('+')
            grpcounter += 1
            if isinstance(entry,str):
                out[mrs.names.index(entry)] = grpcounter
            elif isinstance(entry,list):
                for n in entry:
                    assert(isinstance(n,str))
                    out[mrs.names.index(n)] = grpcounter
            else:
                raise(Exception('entry must be string or list of strings'))
    
    m = min(out)
    if m > 0:
        out = [x-m for x in out]
                
    return out
                
            
        



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


#### Dynamic MRS utils
# THE MAIN MAPPING CLASS


# Class responsible for variable mapping
class VariableMapping(object):
    def __init__(self,
                 param_names,
                 param_sizes,
                 time_variable,
                 config_file):
        """
        Variable Mapping Class Constructor
        
        Mapping betwee free and mapped:
        Mapped = TxN matrix
        Mapped[i,j] = float or 1D-array of floats with size param_sizes[j]
        
        
        
        Parameters
        ----------
        param_names  : list
        param_sizes  : list
        time_variale : array-like
        config_file  : string
        """
        
        self.time_variable  = np.asarray(time_variable)
        self.ntimes         = self.time_variable.shape[0]

        self.mapped_names   = param_names                
        self.mapped_nparams = len(self.mapped_names)
        self.mapped_sizes   = param_sizes

        from runpy import run_path
        settings = run_path(config_file)

        self.Parameters     = settings['Parameters']
        for name in self.mapped_names:
            if name not in self.Parameters:
                self.Parameters[name] = 'fixed'
        self.fcns = {}
        for key in settings:
            if callable(settings[key]):
                self.fcns[key] = settings[key]
        if 'Bounds' in settings:
            self.Bounds = self.create_constraints(settings['Bounds'])
        else:
            self.Bounds         = self.create_constraints(None)
        self.nfree          = self.calc_nfree()
        
    def __str__(self):
        OUT  = '-----------------------\n'
        OUT += 'Variable Mapping Object\n'
        OUT += '-----------------------\n'
        OUT += f'Number of Mapped param groups  = {len(self.mapped_names)}\n'
        OUT += f'Number of Mapped params        = {sum(self.mapped_sizes)}\n'
        OUT += f'Number of Free params          = {self.nfree}\n'
        OUT += f'Number of params if all indep  = {sum(self.mapped_sizes)*self.ntimes}\n'

        OUT += 'Dynamic functions\n'
        for param_name in self.mapped_names:
            beh = self.Parameters[param_name]
            OUT += f'{param_name} \t  {beh}\n'
        
        return OUT


    def calc_nfree(self):
        """
        Calculate number of free parameters based on mapped behaviour
        
        Returns
        -------
        int
        """
        N = 0
        for index,param in enumerate(self.mapped_names):
            beh = self.Parameters[param]
            if (beh == 'fixed'):
                N += self.mapped_sizes[index]
            elif (beh == 'variable'):
                N += self.ntimes*self.mapped_sizes[index]
            else:
                if 'dynamic' in beh:
                    N+= len(beh['params'])*self.mapped_sizes[index]
        return N
    
    
    def create_constraints(self,bounds):
        """
        Create list of constraints to be used in optimization
        
        Parameters:
        -----------
        bounds : dict   {param:bounds}
        
        Returns
        -------
        list
        """
        
        if bounds is None:
            return [(None,None)]*self.calc_nfree()
        
        if not isinstance(bounds,dict):
            raise(Exception('Input should either be a dict or None'))
        
        b = []  # list of bounds
        for index,name in enumerate(self.mapped_names):
            psize = self.mapped_sizes[index]
            
            if (self.Parameters[name] == 'fixed'):
                # check if there are bound on this param
                if name in bounds:
                    for s in range(psize):
                        b.append(bounds[name])
                else:
                    for s in range(psize):
                        b.append((None,None))
                
            elif (self.Parameters[name] == 'variable'):
                for t in range(self.ntimes):
                    for s in range(psize):
                        if name in bounds:
                            b.append(bounds[name])
                        else:
                            b.append((None,None))                    
            else:
                if 'dynamic' in self.Parameters[name]:
                    pnames = self.Parameters[name]['params']
                    for s in range(psize):
                        for p in pnames:
                            if p in bounds:
                                b.append(bounds[p])
                            else:
                                b.append((None,None))  
                    
        return b
 
        
    def mapped_from_list(self,p):
        """
        Converts list of params into Mapped by repeating over time
        
        Parameters
        ----------
        p : list
        
        Returns
        -------
        2D array
        """
        if isinstance(p,list):
            p = np.asarray(p)
        if (p.ndim==1):
            p = np.repeat(p[None,:],self.ntimes,0)
        return p

        
    def create_free_names(self):
        """
        create list of names for free params
        
        Returns
        -------
        list of strings
        """
        names = []
        for index,param in enumerate(self.mapped_names):
            beh = self.Parameters[param]
            if (beh == 'fixed'):
                name = [f'{param}_{x}' for x in range(self.mapped_sizes[index])]
                names.extend(name)
            elif (beh == 'variable'):
                name = [f'{param}_{x}_t{t}' for x in range(self.mapped_sizes[index]) for t in range(self.ntimes)]
                names.extend(name)
            else:
                if 'dynamic' in beh:
                    dyn_name = self.Parameters[param]['params']
                    name = [f'{param}_{y}_{x}' for x in range(self.mapped_sizes[index]) for y in dyn_name]
                    names.extend(name)

        return names
        
    def free_to_mapped(self,p):
        """
        Convert free into mapped params over time
        fixed params get copied over time domain
        variable params are indep over time
        dynamic params are mapped using dyn model
        
        Parameters
        ----------
        p : 1D array
        
        Returns
        -------
        2D array (time X params)
        
        """
        # Check input
        if (p.size != self.nfree):
            raise(Exception(f'Input free params does not have expected number of entries. Found {p.size}, expected {self.nfree}'))
        
        # Mapped params is time X nparams (each param is an array of params)
        mapped_params = np.empty((self.ntimes,self.mapped_nparams),dtype=object)

        counter = 0
        for index,name in enumerate(self.mapped_names):
            nmapped   = self.mapped_sizes[index] 
            
            if (self.Parameters[name] == 'fixed'): # repeat param over time
                for t in range(self.ntimes):
                    mapped_params[t,index] = p[counter:counter+nmapped]
                counter += nmapped
                    
            elif (self.Parameters[name] == 'variable'): # copy one param for each time point                
                for t in range(self.ntimes):
                    mapped_params[t,index] = p[counter+t*nmapped:counter+t*nmapped+nmapped]
                    counter += nmapped

            else:
                if 'dynamic' in self.Parameters[name]:
                    # Generate time courses
                    func_name = self.Parameters[name]['dynamic']
                    nfree     = len(self.Parameters[name]['params'])                    
                    
                    mapped = np.zeros((self.ntimes,nmapped))
                    for i in range(nmapped):
                        params      = p[counter:counter+nfree]
                        mapped[:,i] = self.fcns[func_name](params,self.time_variable)
                        counter += nfree

                    for t in range(self.ntimes):
                        mapped_params[t,index] = mapped[t,:]
                
                else:
                    raise(Exception("Unknown Parameter type - should be one of 'fixed', 'variable', {'dynamic'}"))
        
        return mapped_params

    def print_free(self,x):
        """
        Print free params and their names
        """
        print(dict(zip(vm.create_free_names(),x)))
        
    def check_bounds(self,x,tol=1e-10):
        """
        Check that bounds apply and return corrected x
        """
        if self.Bounds is None:
            return x
        
        for i,b in enumerate(self.Bounds):
            LB = b[0] if b[0] is not None else -np.inf
            UB = b[1] if b[1] is not None else  np.inf
            if (x[i] < LB):
                x[i] = LB+tol
            if (x[i] > UB):
                x[i] = UB-tol
        return x
        
    # This function may 'invert' the dynamic mapping
    # if the input params are from a single timepoint it assumes constant
    def mapped_to_free(self,p):
        """
        Convert mapped params to free (e.g. to initialise the free params)
        fixed and variable params are simply copied
        dynamic params are converted by inverting the dyn model with Scipy optimize
        
        Parameters
        ----------
        p : 2D array (time X params)
        
        Returns
        -------
        1D array
        """
        # Check input
        p = self.mapped_from_list(p)
        if (p.shape != (self.ntimes,self.mapped_nparams)):
            raise(Exception(f'Input mapped params does not have expected number of entries. Found {p.shape}, expected {(self.ntimes,self.mapped_nparams)}'))
                
        free_params = np.empty(self.nfree)
        counter = 0
        for index,name in enumerate(self.mapped_names):
            psize = self.mapped_sizes[index]
            if (self.Parameters[name] == 'fixed'):
                free_params[counter:counter+psize] = p[0,index]
                counter += psize
            elif (self.Parameters[name] == 'variable'):                
                for t in range(self.ntimes):
                    free_params[counter:counter+psize] = p[t,index]
                    counter += psize
            else:
                if 'dynamic' in self.Parameters[name]:
                    func_name = self.Parameters[name]['dynamic']
                    time_var  = self.time_variable
                    func      = partial(self.fcns[func_name],t=time_var)
                    nfree     = len(self.Parameters[name]['params'])

                    pp = np.stack(p[:,index][:],axis=0)
                    for ppp in range(pp.shape[1]):
                        def loss(x):
                            pred = func(x)
                            return np.mean((pp[:,ppp]-pred)**2)
                        bounds = self.Bounds[counter:counter+nfree]
                        vals = minimize(loss,
                                        np.zeros(len(self.Parameters[name]['params'])),
                                        method='TNC',bounds=bounds).x
                        free_params[counter:counter+nfree] = vals                                             
                        counter += nfree
                    
                
                else:
                    raise(Exception("Unknown Parameter type - should be one of 'fixed', 'variable', {'dynamic'}"))
        
        return free_params
    
    

