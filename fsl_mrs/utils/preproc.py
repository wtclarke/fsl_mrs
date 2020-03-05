#!/usr/bin/env python

# preproc.py - Preprocessing utilities
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from scipy.signal import tukey
from dataclasses import dataclass

# tmp
# from scipy.sparse.linalg import svds
from scipy.optimize import minimize
import hlsvdpro

from fsl_mrs.utils import misc
from fsl_mrs.core import MRS

# Functions:
# align_FID
# apodize

# TODO
# outlier removal (of individual repeats)

# QC measures?
#    Peak signal / noise variance (between coils / channels)
# 


@dataclass
class datacontainer:
    '''Class for keeping track of data and reference data together.'''
    data: np.array
    dataheader: dict
    datafilename: str
    reference: np.array = None
    refheader: dict = None
    reffilename: str = None


def truncate(FID,k,first_or_last='last'):
    """
    Truncate parts of a FID
    
    Parameters:
    -----------
    FID           : array-like
    k             : int (number of timepoints to remove)
    first_or_last : either 'first' or 'last' (which bit to truncate)

    Returns:
    --------
    array-like
    """
    FID_trunc = FID.copy()
    
    if first_or_last == 'first':
        return FID_trunc[k:]
    elif first_or_last == 'last':
        return FID_trunc[:-k]
    else:
        raise(Exception("Last parameter must either be 'first' or 'last'"))

def pad(FID,k,first_or_last='last'):
    """
    Pad parts of a FID
    
    Parameters:
    -----------
    FID           : array-like
    k             : int (number of timepoints to add)
    first_or_last : either 'first' or 'last' (which bit to pad)

    Returns:
    --------
    array-like
    """
    FID_pad = FID.copy()
    
    if first_or_last == 'first':
        return np.pad(FID_pad,(k,0))
    elif first_or_last == 'last':
        return np.pad(FID_pad,(0,k))
    else:
        raise(Exception("Last parameter must either be 'first' or 'last'"))


def apodize(FID,dwelltime,broadening,filter='exp'):
    """ Apodize FID
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwelltime in seconds
        broadening (tuple,float): shift in Hz
        filter (str,optional):'exp','l2g'

    Returns:
        FID (ndarray): Apodised FID
    """
    taxis = np.linspace(0,dwelltime*(FID.size-1),FID.size)
    if filter=='exp':
        Tl = 1/broadening[0]
        window = np.exp(-taxis/Tl)
    elif filter=='l2g':
        Tl = 1/broadening[0]
        Tg = 1/broadening[1]
        window = np.exp(taxis/Tl)*np.exp(taxis**2/Tg**2)
    return window*FID


# Phase-Freq alignment functions
def align_FID(mrs,src_FID,tgt_FID,ppmlim=None,shift=True):
    """
       Phase and frequency alignment

    Parameters
    ----------
    mrs : MRS Object
    src_FID : array-like
    tgt_FID : array-like
    ppmlim : tuple

    Returns
    -------
    array-like
    """

    # Internal functions so they can see globals
    def shift_phase_freq(FID,phi,eps,extract=True):        
        sFID = np.exp(-1j*phi)*misc.shift_FID(mrs,FID,eps)
        if extract:
            sFID = misc.extract_spectrum(mrs,sFID,ppmlim=ppmlim,shift=shift)
        return sFID
    def cf(p):
        phi    = p[0] #phase shift
        eps    = p[1] #freq shift
        FID    = shift_phase_freq(src_FID,phi,eps)
        target = misc.extract_spectrum(mrs,tgt_FID,ppmlim=ppmlim,shift=shift)
        xx     = np.linalg.norm(FID-target)
        return xx
    x0  = np.array([0,0])
    res = minimize(cf, x0, method='Powell')
    phi = res.x[0]
    eps = res.x[1]
    
    return shift_phase_freq(src_FID,phi,eps,extract=False)

def get_target_FID(FIDlist,target='mean'):
    """
    target can be 'mean' or 'first' or 'nearest_to_mean' or 'median'
    """
    if target == 'mean':
        return sum(FIDlist) / len(FIDlist)
    elif target == 'first':
        return FIDlist[0].copy()
    elif target == 'nearest_to_mean':
        avg = sum(FIDlist) / len(FIDlist)
        d   = [np.linalg.norm(fid-avg) for fid in FIDlist]
        return FIDlist[np.argmin(d)].copy()
    elif target == 'median':
        return np.median(np.real(np.asarray(FIDlist)),axis=0)+1j*np.median(np.imag(np.asarray(FIDlist)),axis=0)
    else:
        raise(Exception('Unknown target type {}'.format(target)))


def phase_freq_align(FIDlist,bandwidth,centralFrequency,ppmlim=None,niter=2,verbose=False,shift=True,target=None):
    """
    Algorithm:
       Average spectra
       Loop over all spectra and find best phase/frequency shifts
       Iterate

    TODO: 
       test the code
       add dedrifting?

    Parameters:
    -----------
    FIDlist          : list
    bandwidth        : float (unit=Hz)
    centralFrequency : float (unit=Hz)
    ppmlim           : tuple
    niter            : int
    verbose          : bool
    shift            : apply H20 shift to ppm limit
    ref              : reference data to align to
    
    Returns:
    --------
    list of FID aligned to each other
    """
    all_FIDs = FIDlist.copy()
    
    for iter in range(niter):
        if verbose:
            print(' ---- iteration {} ----\n'.format(iter))

        if target is None:
            target = get_target_FID(FIDlist,target='nearest_to_mean')

        MRSargs = {'FID':target,'bw':bandwidth,'cf':centralFrequency}
        mrs = MRS(**MRSargs)
        for idx,FID in enumerate(all_FIDs):
            if verbose:
                print('... aligning FID number {}'.format(idx),end='\r')
            all_FIDs[idx] = align_FID(mrs,FID,target,ppmlim=ppmlim,shift=shift)
        if verbose:
            print('\n')
    return all_FIDs


# Methods for FID combination

def dephase(FIDlist):
    """
      Uses first data point of each FID to dephase each FID in list
      Returns a list of dephased FIDs
    """
    return [fid*np.exp(-1j*np.angle(fid[0])) for fid in FIDlist]

def prewhiten(FIDlist,prop=.1,C=None):
    """
       Uses noise covariance to prewhiten data

    Parameters:
    -----------
    FIDlist : list of FIDs
    prop    : proportion of data used to estimate noise covariance
    C       : noise covariance matrix, if provided it is not measured from data.

    Returns : 
    list of FIDs
    pre-whitening matrix
    noise covariance matrix
    """
    FIDs = np.asarray(FIDlist,dtype=np.complex)
    if C is None:
        # Estimate noise covariance
        start = int((1-prop)*FIDs.shape[0])
        # Raise warning if not enough samples
        if (FIDs.shape[0]-start)<1.5*FIDs.shape[1]:
            raise(Warning('You may not have enough samples to robustly estimate the noise covariance'))
        C     = np.cov(FIDs[start:,:],rowvar=False)

    D,V   = np.linalg.eigh(C,UPLO='U') #UPLO = 'U' to match matlab implementation    
    # Pre-whitening matrix
    W     = V@np.diag(1/np.sqrt(D))
    # Pre-whitened data
    FIDs = FIDs@W
    return FIDs,W,C

def svd_reduce(FIDlist,W=None,C=None,return_alpha=False):
    """
    Combine different channels by SVD method
    Based on C.T. Rodgers and M.D. Robson, Magn Reson Med 63:881–891, 2010

    Parameters:
    -----------
    FIDlist      : list of FIDs
    W            : pre-whitening matrix (only used to calculate sensitivities)
    return_alpha : return sensitivities?

    Returns:
    --------
    array-like (FID)
    array-like (sensitivities) - optional
    """
    FIDs  = np.asarray(FIDlist)
    U,S,V = np.linalg.svd(FIDs,full_matrices=False)
    # U,S,V = svds(FIDs,k=1)  #this is much faster but fails for two coil case

    nCoils = FIDs.shape[1]
    svdQuality = ((S[0]/np.linalg.norm(S))*np.sqrt(nCoils)-1)/(np.sqrt(nCoils)-1)

    # get arbitrary amplitude
    iW = np.eye(FIDs.shape[1])
    if W is not None:
        iW = np.linalg.inv(W)
    amp = V[0,:]@iW
    
    # arbitrary scaling here such that the first coil weight is real and positive
    svdRescale = np.linalg.norm(amp)*(amp[0]/np.abs(amp[0]))

    # combined channels
    FID = U[:,0]*S[0]*svdRescale

    if return_alpha:
        # sensitivities per channel        
        # alpha = amp/svdRescale # equivalent to svdCoilAmplitudes in matlab implementation

        # Instead incorporate the effect of the whitening stage as well.
        if C is None:
            C = np.eye(FIDs.shape[1])        
        scaledAmps = (amp/svdRescale).conj().T
        alpha = np.linalg.inv(C)@scaledAmps * svdRescale.conj() * svdRescale    
        return FID,alpha
    else:
        return FID

def weightedCombination(FIDlist,weights):
    """
    Combine different FIDS with different complex weights

    Parameters:
    -----------
    FIDlist      : list of FIDs
    weights      : complex weights

    Returns:
    --------
    array-like (FID)    
    """
    if isinstance(FIDlist,list):
        FIDlist  = np.asarray(FIDlist)
    if isinstance(weights,list):
        weights = np.asarray(weights)
    # combine channels
    FID = np.sum(FIDlist*weights[None,:],axis=1)

    return FID

def combine_FIDs(FIDlist,method,do_prewhiten=False,do_dephase=False,do_phase_correct=False,weights=None):
    """
       Combine FIDs (either from multiple coils or multiple averages)
    
    Parameters:
    -----------
    FIDlist   : list of FIDs or array with time dimension first
    method    : one of 'mean', 'svd', 'svd_weights', 'weighted'
    prewhiten : bool
    dephase   : bool

    Returns:
    --------
    array-like

    """

    if isinstance(FIDlist,list):
        FIDlist = np.asarray(FIDlist).T

    # Pre-whitening
    W = None
    C = None
    if do_prewhiten:
        FIDlist,W,C = prewhiten(FIDlist)

    # Dephasing
    if do_dephase:
        FIDlist   = dephase(FIDlist)

    # Combining channels
    if method == 'mean':
        return np.mean(FIDlist,axis=-1)
    elif method == 'svd':
        return svd_reduce(FIDlist,W)
    elif method == 'svd_weights':
        return svd_reduce(FIDlist,W,C,return_alpha=True)
    elif method == 'weighted':
        return weightedCombination(FIDlist,weights)
    else:
        raise(Exception("Unknown method '{}'. Should be either 'mean' or 'svd'".format(method)))


def eddy_correct(FIDmet,FIDPhsRef):
    """
    Subtract water phase from metab phase
    Typically do this after coil combination

    Parameters:
    -----------
    FIDmet : array-like (FID for the metabolites)
    FIDPhsRef   : array-like (Phase reference FID)

    Returns:
    --------
    array-like (corrected metabolite FID)
    """
    phsRef = np.angle(FIDPhsRef)
    return np.abs(FIDmet) * np.exp(1j*(np.angle(FIDmet)-phsRef))

def hlsvd(FID,dwelltime,frequencylimit,numSingularValues=50):
    """ Run HLSVDPRO on FID
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwell time in seconds
        frequencylimit (tuple): Limit deletion of singular values in this range 
        numSingularValues (int, optional): Max number of singular values

    Returns:
        FID (ndarray): Modified FID
    """
    nsv_found, singular_values, frequencies, damping_factors, amplitudes, phases = hlsvdpro.hlsvd(FID,numSingularValues,dwelltime)

    # convert to np array
    frequencies = np.asarray(frequencies)
    damping_factors = np.asarray(damping_factors)
    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)

    # Limit by frequencies
    limitIndicies = (frequencies > frequencylimit[0]) & (frequencies < frequencylimit[1])

    sumFID = np.zeros(FID.shape,dtype=np.complex128)
    timeAxis = np.linspace(0,dwelltime*(FID.size-1),FID.size)

    for use,f,d,a,p in zip(limitIndicies,frequencies,damping_factors,amplitudes,phases):
        if use:
            sumFID += a * np.exp((timeAxis/d) + 1j*2*np.pi * (f*timeAxis+p/360.0))

    return FID - sumFID

def timeshift(FID,dwelltime,shiftstart,shiftend,samples=None):
    """ Shift data on time axis
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwell time in seconds
        shiftstart (float): Shift start point in seconds 
        shiftend (float): Shift end point in seconds
        samples (int, optional): Resample to this number of points

    Returns:
        FID (ndarray): Shifted FID
    """
    originalAcqTime = dwelltime*(FID.size-1)    
    originalTAxis = np.linspace(0,originalAcqTime,FID.size)
    if samples is None:        
        newDT = dwelltime
    else:
        totalacqTime = originalAcqTime-shiftstart+shiftend
        newDT = totalacqTime/samples
    newTAxis = np.arange(originalTAxis[0]+shiftstart,originalTAxis[-1]+shiftend,newDT)
    FID = np.interp(newTAxis,originalTAxis,FID,left=0.0+1j*0.0, right=0.0+1j*0.0)

    return FID,newDT

def freqshift(FID,dwelltime,shift):
    """ Shift data on frequency axis
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwelltime in seconds
        shift (float): shift in Hz

    Returns:
        FID (ndarray): Shifted FID
    """
    tAxis = np.linspace(0,dwelltime*FID.size,FID.size)
    phaseRamp = 2*np.pi*tAxis*shift
    FID = FID * np.exp(1j*phaseRamp)
    return FID

def identifyUnlikeFIDs(FIDList,bandwidth,centralFrequency,sdlimit = 1.96,iterations=2,ppmlim=None,shift=True):
    """ Identify FIDs in a list that are unlike the others

    Args:
        FIDList (list of ndarray): Time domain data
        bandwidth (float)        : Bandwidth in Hz
        centralFrequency (float) : Central frequency in Hz
        sdlimit (float,optional) : Exclusion limit (number of stnadard deviations). Default = 3.
        iterations (int,optional): Number of iterations to use.
        ppmlim (tuple,optional)  : Limit to this ppm range
        shift (bool,optional)    : Apply H20 shft

    Returns:
        goodFIDS (list of ndarray): FIDs that passed the criteria
        badFIDS (list of ndarray): FIDs that failed the likeness critera
        rmIndicies (list of int): Indicies of those FIDs that have been removed
        metric (list of floats): Likeness metric of each FID
    """

    # Calculate the FID to compare to
    target = get_target_FID(FIDList,target='median')

    if ppmlim is not None:
        MRSargs = {'FID':target,'bw':bandwidth,'cf':centralFrequency}
        mrs = MRS(**MRSargs)
        target = misc.extract_spectrum(mrs,target,ppmlim=ppmlim,shift=shift)
        compareList = [misc.extract_spectrum(mrs,f,ppmlim=ppmlim,shift=shift) for f in FIDList]
    else:
        compareList = [misc.FIDToSpec(f) for f in FIDList]
        target = misc.FIDToSpec(target)
    
    # Do the comparison    
    for idx in range(iterations):
        metric = []
        for data in compareList:
            metric.append(np.linalg.norm(data-target))            
        metric = np.asarray(metric)
        metric_avg = np.mean(metric)
        metric_std = np.std(metric)

        goodFIDs,badFIDs,rmIndicies,keepIndicies = [],[],[],[]
        for iDx,(data,m) in enumerate(zip(FIDList,metric)):
            if m > ((sdlimit*metric_std)+metric_avg) or m < (-(sdlimit*metric_std)+metric_avg):
                badFIDs.append(data)
                rmIndicies.append(iDx)
            else:
                goodFIDs.append(data)
                keepIndicies.append(iDx)

        target = get_target_FID(goodFIDs,target='median')
        if ppmlim is not None:
            target = misc.extract_spectrum(mrs,target,ppmlim=ppmlim,shift=shift)
        else:
            target = misc.FIDToSpec(target)
    

    return goodFIDs,badFIDs,keepIndicies,rmIndicies,metric.tolist()

def phaseCorrect(FID,bw,cf,ppmlim=(2.8,3.2),shift=True):
    #Find maximum of absolute spectrum in ppm limit
    MRSargs = {'FID':FID,'bw':bw,'cf':cf}
    mrs = MRS(**MRSargs)
    spec = misc.extract_spectrum(mrs,FID,ppmlim=ppmlim,shift=shift)

    maxIndex = np.argmax(np.abs(spec))
    phaseAngle = np.angle(spec[maxIndex])
    
    return FID * np.exp(1j*-phaseAngle)