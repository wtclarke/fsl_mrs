#!/usr/bin/env python

# core.py - main MRS class definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT


import warnings
from os.path import isfile

from fsl_mrs.utils import mrs_io as io
from fsl_mrs.utils import misc
from fsl_mrs.utils.constants import *

import numpy as np


#------------------------------------------------
#
#
#------------------------------------------------




class MRS(object):
    """
      MRS Class - container for FID, Basis, and sequence info
    """
    def __init__(self,FID=None,header=None,basis=None,names=None,basis_hdr=None,H2O=None,cf=None,bw=None):

        # If FID and basis are files then read data from file
        #if FID is not None and basis is not None:
        #    if isfile(FID) and isfile(basis):
        #        self.from_files(FID,basis)
        #        return
    
        # Read in class data input
        # (now copying the data - looks ugly but better than referencing.
        # now I can run multiple times with different setups)
        if FID is not None:
            self.set_FID(FID)
        else:
            return
        
        if H2O is not None:
            self.H2O           = H2O.copy()
        else:
            self.H2O           = None

                
        # Set FID class attributes
        if header is not None:
            self.set_acquisition_params(centralFrequency=header['centralFrequency'],bandwidth=header['bandwidth'])
        elif (cf is not None) and (bw is not None):
            self.set_acquisition_params(centralFrequency=cf,bandwidth=bw)
        else:
            raise ValueError('You must pass a header or bandwidth and central frequency.')
        
        # Set Basis info
        if basis is not None:
            self.basis          = basis.copy()
            # Handle single basis spectra
            if self.basis.ndim==1:
                self.basis = self.basis[:,np.newaxis]
            # Assume that there will always be more timepoints than basis spectra.
            if self.basis.shape[0] < self.basis.shape[1]:
                self.basis = self.basis.T
            self.numBasis       = self.basis.shape[1]            
            self.numBasisPoints = self.basis.shape[0]
            
            if (names is not None) and (basis_hdr is not None):
                self.names         = names.copy()
                self.set_acquisition_params_basis(1/basis_hdr['bandwidth'])
            else:
                raise ValueError('Pass basis names and header with basis.')

            # Now interpolate the basis to the same time axis.
            self.resample_basis()

        else:
            self.basis         = None
            self.names         = None
            self.numBasis      = None
            self.basis_dwellTime   = None
            self.basis_bandwidth   = None

        # Other properties
        self.metab_groups      = None
        
            
    def from_files(self,FID_file,Basis_file):
        FID,FIDheader       = io.read_FID(FID_file)
        basis,names,Bheader = io.read_basis(Basis_file)

        cf = FIDheader['centralFrequency']
        bw = FIDheader['bandwidth']

        MRSArgs = {'bw':bw,'cf':cf,
                   'basis':basis,'basis_hdr':Bheader[0],
                   'names':names}

        self.__init__(FID=FID,**MRSArgs)

        return

        
    def __str__(self):
        out  = '------- MRS Object ---------\n'
        out += '     FID.shape             = {}\n'.format(self.FID.shape)        
        out += '     FID.centralFreq (MHz) = {}\n'.format(self.centralFrequency/1e6)
        out += '     FID.centralFreq (T)   = {}\n'.format(self.centralFrequency/H1_gamma/1e6)        
        out += '     FID.bandwidth (Hz)    = {}\n'.format(self.bandwidth)
        out += '     FID.dwelltime (s)     = {}\n'.format(self.dwellTime)
        if self.basis is not None:
            out += '     basis.shape           = {}\n'.format(self.basis.shape)
            out += '     Metabolites           = {}\n'.format(self.names)
            out += '     numBasis              = {}\n'.format(self.numBasis)
        out += '     timeAxis              = {}\n'.format(self.timeAxis.shape)
        out += '     freqAxis              = {}\n'.format(self.frequencyAxis.shape)
        
        return out

    
    # Acquisition parameters
    def set_acquisition_params(self,centralFrequency,bandwidth):
        """
          Set useful params for fitting

          Parameters
          ----------
          centralFrequency : float  (unit=Hz)
          bandwidth : float (unit=Hz)
          echotime : float (unit=sec)

        """
        # Store CF in Hz
        self.centralFrequency = misc.checkCFUnits(centralFrequency)        

        self.bandwidth        = bandwidth 
        
        self.dwellTime        = 1/self.bandwidth

        axes = misc.calculateAxes(self.bandwidth,
                                  self.centralFrequency,
                                  self.numPoints)

        self.timeAxis         = axes['time']  
        self.frequencyAxis    = axes['freq']         
        self.ppmAxis          = axes['ppm']  
        self.ppmAxisShift     = axes['ppmshift']  
        self.ppmAxisFlip      = np.flipud(self.ppmAxisShift)
        # turn into column vectors
        self.timeAxis         = self.timeAxis[:,None]
        self.frequencyAxis    = self.frequencyAxis[:,None]
        self.ppmAxisShift     = self.ppmAxisShift[:,None]


    def set_acquisition_params_basis(self,dwelltime):
        """
           sets basis-specific timing params
        """
        # Basis has different dwelltime
        self.basis_dwellTime     = dwelltime
        self.basis_bandwidth     = 1/dwelltime
        self.basis_frequencyAxis = np.linspace(-self.basis_bandwidth/2,
                                               self.basis_bandwidth/2,
                                               self.numBasisPoints)
        self.basis_timeAxis      = np.linspace(self.basis_dwellTime,
                                               self.basis_dwellTime*self.numBasisPoints,
                                               self.numBasisPoints)

    def getSpectrum(self,ppmlim=None,shift=True):
        spectrum = misc.FIDToSpec(self.FID)
        f,l = self.ppmlim_to_range(ppmlim,shift=shift)
        return spectrum[f:l]
    
    def getAxes(self,axis='ppmshift',ppmlim=None):        
        if axis.lower() == 'ppmshift':
            f,l = self.ppmlim_to_range(ppmlim,shift=True)
            return np.squeeze(self.ppmAxisShift[f:l])
        elif axis.lower() == 'ppm':
            f,l = self.ppmlim_to_range(ppmlim,shift=False)
            return np.squeeze(self.ppmAxis[f:l])
        elif axis.lower() == 'freq':
            f,l = self.ppmlim_to_range(ppmlim,shift=False)
            return np.squeeze(self.frequencyAxis[f:l])
        elif axis.lower() == 'time':
            return np.squeeze(self.timeAxis)
        else:
            raise ValueError('axis must be one of ppmshift, ppm, freq or time.')

    def ppmlim_to_range(self,ppmlim=None,shift=True):
        """
           turns ppmlim into data range

           Parameters:
           -----------

           ppmlim : tuple

           Outputs:
           --------

           int : first position
           int : last position
        """
        if ppmlim is not None:
            if shift:
                ppm2range = lambda x: np.argmin(np.abs(self.ppmAxisShift-x))
            else:
                ppm2range = lambda x: np.argmin(np.abs(self.ppmAxis-x))
            first = ppm2range(ppmlim[0])
            last  = ppm2range(ppmlim[1])
            if first>last:
                first,last = last,first
        else:
            first,last = 0,self.numPoints 

        return int(first),int(last)


    def resample_basis(self):
        """
           Usually the basis is simulated using different timings/number of points
           This interpolates the basis to match the FID
        """
        # RESAMPLE BASIS FUNCTION
        # bdt    = self.basis_dwellTime
        # bbw    = self.basis_bandwidth
        # bn     = self.numBasisPoints
        
        # bt     = np.linspace(bdt,bdt*bn,bn)-bdt
        # fidt   = self.timeAxis.flatten()-self.dwellTime
        
        # f      = interp1d(bt,self.basis,axis=0)
        # newiFB = f(fidt)       
        
        self.basis = misc.ts_to_ts(self.basis,self.basis_dwellTime,self.dwellTime,self.numPoints)
        self.basis_dwellTime = self.dwellTime
        self.basis_bandwidth = 1/self.dwellTime
        self.numBasisPoints = self.numPoints

        
    # Helper functions
    def check_FID(self,ppmlim=(.2,4.2),repair=False):
        """
           Check if FID needs to be conjugated
           by looking at total power within ppmlim range

        Parameters
        ----------
        ppmlim : list
        repare : if True applies conjugation to FID

        Returns
        -------
        0 if check successful and -1 if not (also issues warning)

        """
        first,last = self.ppmlim_to_range(ppmlim)
        Spec1 = np.real(misc.FIDToSpec(self.FID))[first:last]
        Spec2 = np.real(misc.FIDToSpec(np.conj(self.FID)))[first:last]
        
        if np.linalg.norm(misc.detrend(Spec1,deg=4)) < np.linalg.norm(misc.detrend(Spec2,deg=4)):
            if repare is False:
                warnings.warn('YOU MAY NEED TO CONJUGATE YOUR FID!!!')
                return -1
            else:
                self.conj_FID()
                return 1
            
        return 0

    def conj_FID(self):
        """
        Conjugate FID and recalculate spectrum
        """
        self.FID  = np.conj(self.FID)
        self.Spec = misc.FIDToSpec(self.FID)

    def check_Basis(self,ppmlim=(.2,4.2),repair=False):
        """
           Check if Basis needs to be conjugated
           by looking at total power within ppmlim range

        Parameters
        ----------
        ppmlim : list
        repare : if True applies conjugation to basis

        Returns
        -------
        0 if check successful and -1 if not (also issues warning)

        """
        first,last = self.ppmlim_to_range(ppmlim)

        conjOrNot = []
        for b in self.basis.T:
            Spec1 = np.real(misc.FIDToSpec(b))[first:last]
            Spec2 = np.real(misc.FIDToSpec(np.conj(b)))[first:last]            
            if np.linalg.norm(misc.detrend(Spec1,deg=4)) < np.linalg.norm(misc.detrend(Spec2,deg=4)):
                conjOrNot.append(1.0)
            else:
                conjOrNot.append(0.0)

        if (sum(conjOrNot)/len(conjOrNot))>0.5:
            if repare is False:
                warnings.warn('YOU MAY NEED TO CONJUGATE YOUR BASIS!!!')
                return -1
            else:
                self.conj_Basis()
                return 1
            
        return 0

    def conj_Basis(self):
        """
        Conjugate FID and recalculate spectrum
        """
        self.basis  = np.conj(self.basis)

    def ignore(self,metabs):
        """
          Ignore a subset of metabolites by removing them from the basis

          Parameters
          ----------

          metabs: list
        
        """
        if self.basis is None:
            raise Exception('You must first specify a basis before ignoring a subset of it!')

        if metabs is not None:
            for m in metabs:
                idx = self.names.index(m)
                self.names.pop(idx)
                self.basis = np.delete(self.basis,idx,axis=1)
            self.numBasis = len(self.names)

    def keep(self,metabs):
        """
          Keep a subset of metabolites by removing all others from basis

          Parameters
          ----------

          metabs: list
        
        """
        if metabs is not None:
            metabs = [m for m in self.names if m not in metabs]
            self.ignore(metabs)
            

    def add_peak(self,ppm,name,gamma=0,sigma=0):
        """
           Add peak to basis
        """

        peak = misc.create_peak(self,ppm,gamma,sigma)[:,None]
        self.basis = np.append(self.basis,peak,axis=1)
        self.names.append(name)
        self.numBasis += 1

    def add_MM_peaks(self,ppmlist=None,gamma=0,sigma=0):
        """
           Add macromolecule list
           
        Parameters
        ----------
    
        ppmlist : default is [1.7,1.4,1.2,2.0,0.9]

        gamma,sigma : float parameters of Voigt blurring
        """
        if ppmlist is None:
            ppmlist = [1.7,1.4,1.2,2.0,0.9]
        names   = ['MM'+'{:.0f}'.format(i*10).zfill(2) for i in ppmlist]

        for name,ppm in zip(names,ppmlist):
            self.add_peak(ppm,name,gamma,sigma)

        return len(ppmlist)


    def set_FID(self,FID):
        """
          Sets the FID and calculates spectrum
        """
        self.FID         = FID.copy()
        self.numPoints   = self.FID.size
        self.Spec        = misc.FIDToSpec(self.FID)
        
                  
    # I/O functions  [NOW OBSOLETE?]
    # @staticmethod 
    # def read(filename,TYPE='RAW'):
    #     """
    #       Read data file

    #       Parameters
    #       ----------

    #       filename : string
    #       TYPE     : string

    #       Outputs
    #       -------

    #       numpy array : data 
    #       string list : header information 
    #     """
    #     if TYPE == 'RAW':
    #         data,header = io.readLCModelRaw(filename)
    #     else:
    #         raise Exception('Unknow file type {}'.format(TYPE))
    #     return data, header
    
    # def read_data(self,filename,TYPE='RAW'):
    #     """
    #       Read data file and update acq params

    #       Parameters
    #       ----------

    #       filename : string
    #       TYPE     : string

    #     """
    #     self.datafile = filename
    #     FID, header   = self.read(filename,TYPE)
    #     self.set_FID(FID)

    #     if header['centralFrequency'] is None:
    #         header['centralFrequency'] = 123.2E6
    #         warnings.warn('Cannot determine central Frequency from input. Setting to default of 123.2E6 Hz')
    #     if header['bandwidth'] is None:
    #         header['bandwidth'] = 4000
    #         warnings.warn('Cannot determine bandwidth. Setting to default of 4000Hz.')

        
    #     self.set_acquisition_params(centralFrequency=header['centralFrequency'],
    #                                 bandwidth=header['bandwidth'])

    # def read_basis_files(self,basisfiles,TYPE='RAW',ignore=[]):
    #     """
    #        Reads basis file and extracts name of metabolite from file name
    #        Assumes .RAW files are FIDs (not spectra)
    #        Should change this to reading metabolite name from header
    #     """
    #     self.numBasis = 0
    #     self.basis = []
    #     self.names = []
    #     for iDx,file in enumerate(basisfiles):
    #         data,_ = self.read(file,TYPE)
    #         name = os.path.splitext(os.path.split(file)[-1])[-2]
    #         if name not in ignore:
    #             self.names.append(name)
    #             self.basis.append(data)
    #             self.numBasis +=1
    #     self.basis = np.asarray(self.basis).astype(np.complex).T
    #     #self.basis = self.basis - self.basis.mean(axis=0)
    
    # def read_basis_from_folder(self,folder,TYPE='RAW',ignore=[]):
    #     """
    #        Reads all .RAW files from folder assuming they are all metabolite FIDs
    #     """
    #     basisfiles = sorted(glob.glob(os.path.join(folder,'*.'+TYPE)))
    #     self.read_basis_files(basisfiles,ignore=ignore)

    # def read_basis_from_file(self,filename):
    #     """
    #        Reads single basis (.BASIS) file assuming it is spectra (not FIDs)
    #     """
    #     self.basis, self.names, header = io.readLCModelBasis(filename,self.numPoints)

    #     if header['dwelltime'] is not None:
    #         self.set_acquisition_params_basis(header['dwelltime'])
    #         self.resample_basis()
        
    #     self.numBasis = len(self.names)

    # def read_h2o(self,filename,TYPE='RAW'):
    #     """
    #        Reads H2O file 
    #     """
    #     self.H2O, header = self.read(filename, TYPE)
        
        
