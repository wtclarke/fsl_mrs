#!/usr/bin/env python

# newcore.py - main MRS classes / functions definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.utils import misc
from fsl_mrs.utils import mrs_io
from scipy.interpolate import interp1d

def resample(basis,fid):
    '''
       Resample basis signal to match fid sampling rate
    '''
    bdt    = basis._dwellTime
    bbw    = 1/bdt
    bn     = basis._numPoints
    
    bt     = np.linspace(bdt,bdt*bn,bn)-bdt
    fidt   = fid._timeAxis.flatten()-fid._dwellTime
    
    f      = interp1d(bt,basis._FID,axis=0)
    newiFB = f(fidt)

    new_basis = basis
    new_basis._FID  = newiFB
    new_basis._Spec = np.fft.fft(new_basis._FID)
    

    return new_basis



class FID(object):
    # Data
    _FID       = None
    _Spec      = None
    # Properties
    _dwellTime        = None
    _numPoints        = None
    _centralFrequency = None
    _bandwidth        = None
    
    def __init__(object,filename):
        # Read data
        _FID,header = mrs_io.readLCModelRaw(filename)
        # Set internal parameters based on the header information
        _numPoints  = _FID.size


        if header['centralFrequency'] is None:
           self._centralFrequency = 123.2E6
            warnings.warn('Cannot determine central Frequency from input. Setting to default of 123.2E6 Hz (3T)')
        if header['bandwidth'] is None:
            self._bandwidth = 4000
            warnings.warn('Cannot determine bandwidth. Setting to default of 4000Hz.')
       if header['echotime'] is None:
            self._echotime = 30e-3
            warnings.warn('Cannot determine echo time. Setting to default of 30ms.')
        
        self._dwellTime        = 1/self._bandwidth; 
        self._timeAxis         = np.linspace(self.dwellTime,
                                            self.dwellTime*self.numPoints,
                                            self.numPoints)  
        self._frequencyAxis    = np.linspace(-self.bandwidth/2,
                                            self.bandwidth/2,
                                            self.numPoints)
        
        self._ppmAxis          = misc.hz2ppm(self.centralFrequency,
                                            self.frequencyAxis,shift=False)
        self._ppmAxisShift     = misc.hz2ppm(self.centralFrequency,
                                            self.frequencyAxis,shift=True)
        self._ppmAxisFlip      = np.flipud(self.ppmAxisShift)
        
        # turn into column vectors
        self._timeAxis         = self._timeAxis[:,None]
        self._frequencyAxis    = self._frequencyAxis[:,None]




        
class Basis(object,FID):
    def __init__(object,filename):
        _FID,header = readLCModelBasis(filename)

class H2O(object,FID):
    def __init__(object,filename):
        _FID,header = readLCModelRaw(filename)
    




