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


def ppm2hz(cf,ppm,shift=True):
    if shift:
        return -(ppm-H2O_PPM_TO_TMS)*cf*1E-6
    else:
        return -(ppm)*cf*1E-6

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
