#!/usr/bin/env python

# preproc.py - Preprocessing utilities
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from scipy.signal import tukey

# tmp
from scipy.sparse.linalg import svds

from fsl_mrs.utils import misc

# Functions
#    apodize --> check!
#    phase_freq_align --> check!
#    channel_combine --> check!
#    eddy_correct : subtract water FID phase from metab FID

# TODO
# residual water removal
# outlier removal (of individual repeats)

# QC measures?
#    Peak signal / noise variance (between coils / channels)
# 


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


def apodize(FID):
    """
       FID Apodization

    TODO: optional choice of filter

    Parameters:
    -----------
     FID : array-like
    Returns:
    --------
    Array-like
    """
    window = tukey(FID.size * 2)[FID.size:]
    return window*FID


# Phase-Freq alignment functions
def align_FID(mrs,src_FID,tgt_FID,ppmlim=None):
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
            sFID = misc.extract_spectrum(mrs,sFID,ppmlim=ppmlim)
        return sFID
    def cf(p):
        phi    = p[0] #phase shift
        eps    = p[1] #freq shift
        FID    = shift_phase_freq(src_FID,phi,eps)
        target = misc.extract_spectrum(mrs,tgt_FID,ppmlim=ppmlim)
        xx     = np.linalg.norm(FID-target)
        return xx
    x0  = np.array([0,0])
    res = minimize(cf, x0, method='Powell')
    phi = res.x[0]
    eps = res.x[1]
    
    return shift_phase_freq(src_FID,phi,eps,extract=False)

def get_target_FID(FIDlist,target='mean'):
    """
    target can be 'mean' or 'first' or 'nearest_to_mean'
    """
    if target == 'mean':
        return sum(FIDlist) / len(FIDlist)
    elif target == 'first':
        return FIDlist[0].copy()
    elif target == 'nearest_to_mean':
        avg = sum(FIDlist) / len(FIDlist)
        d   = [np.linalg.norm(fid-avg) for fid in FIDlist]
        return FIDlist[np.argmin(d)].copy()
    else:
        raise(Exception('Unknown target type {}'.format(target)))


def phase_freq_align(FIDlist,bandwidth,centralFrequency,ppmlim=None,niter=2,verbose=False):
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
    
    Returns:
    --------
    list of FID aligned to each other
    """
    all_FIDs = FIDlist.copy()
    for iter in range(niter):
        if verbose:
            print(' ---- iteration {} ----\n'.format(iter))
        target = get_target_FID(FIDlist,target='nearest_to_mean')
        MRSargrgs = {'FID':target,'bw':bandwidth,'cf':centralFrequency}
        mrs = MRS(**MRSargs)
        for idx,FID in enumerate(all_FIDs):
            if verbose:
                print('... aligning FID number {}'.format(idx),end='\r')
            all_FIDs[idx] = align_FID(mrs,FID,target,ppmlim=ppmlim)
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

def prewhiten(FIDlist,prop=.1):
    """
       Uses noise covariance to prewhiten data

    Parameters:
    -----------
    FIDlist : list of FIDs
    prop    : proportion of data used to estimate noise covariance

    Returns : 
    list of FIDs
    pre-whitening matrix
    """
    FIDs = np.asarray(FIDlist,dtype=np.complex)
    # Estimate noise covariance
    start = int((1-prop)*FIDs.shape[1])
    # Raise warning if not enough samples
    if (FIDs.shape[1]-start)<1.5*FIDs.shape[0]:
        raise(Warning('You may not have enough samples to robustly estimate the noise covariance'))
    C     = np.cov(FIDs[:,start:])
    D,V   = np.linalg.eigh(C)
    # Pre-whitening matrix
    W     = np.diag(1/np.sqrt(D))@np.conj(V.T)
    # Pre-whitened data
    FIDs = W@FIDs
    return list(FIDs),W


def svd_reduce(FIDlist,W=None,return_alpha=False):
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
    #U,S,V = np.linalg.svd(FIDs)
    U,S,V = svds(FIDs,k=1)  #this is much faster

    # get arbitrary amplitude
    iW = np.eye(FIDs.shape[0])
    if W is not None:
        iW = np.linalg.inv(W)
    amp  = np.linalg.norm(np.mean(iW@FIDs,axis=0))
    
    # combined channels
    FID = amp*V[0,:].T

    if return_alpha:
        # sensitivities per channel
        if W is not None:
            alpha = iW@U[:,0]
        else:
            alpha = U[:,0]

        return FID,alpha
    else:
        return FID

def combine_FIDs(FIDlist,method,do_prewhiten=False,do_dephase=False,do_phase_correct=False):
    """
       Combine FIDs (either from multiple coils or multiple averages
    
    Parameters:
    -----------
    FIDlist   : list of FIDs
    method    : one of 'mean', 'svd'
    prewhiten : bool
    dephase   : bool

    Returns:
    --------
    array-like

    """

    # Pre-whitening
    W = None
    if do_prewhiten:
        FIDlist,W = prewhiten(FIDlist)

    # Dephasing
    if do_dephase:
        FIDlist   = dephase(FIDlist)

    # Combining channels
    if method == 'mean':
        return sum(FIDlist) / len(FIDlist)
    elif method == 'svd':
        return svd_reduce(FIDlist,W)
    else:
        raise(Exception("Unknown method '{}'. Should be either 'mean' or 'svd'".format(method)))


def eddy_correct(FIDmet,FIDw):
    """
    Subtract water phase from metab phase
    Typically do this after coil combination

    Parameters:
    -----------
    FIDmet : array-like (FID for the metabolites)
    FIDw   : array-like (water FID)

    Returns:
    --------
    array-like (corrected metabolite FID)
    """

    return FIDmet / np.exp(1j*np.angle(FIDw))
