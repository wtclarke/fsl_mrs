#!/usr/bin/env python

# core.py - main MRS class definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT


import sys, os, glob, warnings

from fsl_mrs.utils.mh import MH
from fsl_mrs.utils import mrs_io as io
from fsl_mrs.utils import models, misc
from fsl_mrs.utils import plotting


import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

#------------------------------------------------
#
#
#------------------------------------------------



## ASK MICHIEL/PAUL HOW BEST TO SET UP GLOBAL VARIABLES
H2O_MOLECULAR_MASS = 18.01528   # g/mol
H2O_Conc           = 55.51E3    # mmol/kg      
H2O_PPM_TO_TMS     = 4.65       # Shift of water to Tetramethylsilane
H2O_to_Cr          = 0.4        # Proton ratio
H1_gamma           = 42.576     # MHz/tesla

class MRS(object):
    """
      MRS Class - deals with data I/O and model fitting
    """
    def __init__(self):
        # Data
        self.FID               = None
        self.Spec              = None
        self.basis             = None
        self.H2O               = None
        # Constants
        self.centralFrequency  = None   # Hz
        self.bandwidth         = None   # Hz
        self.echotime          = None   # seconds
        self.numPoints         = None
        self.numBasis          = None
        self.dwellTime         = None   # time between sampling points of FID
        self.timeAxis          = None   # seconds
        self.frequencyAxis     = None   # Hz
        self.ppmAxis           = None   # Chemical shift in ppm
        self.ppmAxisShift      = None   # Shift from water
        self.ppmAxisFlip       = None   # flipped axis
        self.names             = None
        self.metab_groups      = None
        self.T2s               = None
        self.basis_dwellTime   = None
        self.basis_bandwidth   = None
        # Modelling variables
        self.pred              = None   # Predicted FID
        self.con               = None   # Concentrations
        self.eps               = None   # peak shift
        self.gamma             = None   # peak spread
        self.phi0              = None   # phase shift
        self.phi1              = None   # phase ramp
        self.baseline          = None   # baseline
        self.samples           = None   # MCMC samples
        self.all_params        = None   # All model parameters including concentration
        self.con_names         = None   # zipped con and names
        self.all_con_names     = None   # zipped con and names for all metabolites and metab groups
        self.all_con_names_h2o = None   # zipped con and names for all metabolites and metab groups norm h2o
        self.all_con_names_std = None   # zipped std and names for all metabolites and metab groups std
        # Quantification
        self.quantif           = MRS_quantif() # Instantiate quantif object

    def __str__(self):
        out  = '------- MRS Object ---------\n'
        out += '     FID.shape             = {}\n'.format(self.FID.shape)
        out += '     H2O.shape             = {}\n'.format(self.H2O.shape)
        out += '     basis.shape           = {}\n'.format(self.basis.shape)
        out += '     FID.centralFreq (MHz) = {}\n'.format(self.centralFrequency/1e6)
        out += '     FID.centralFreq (T)   = {}\n'.format(self.centralFrequency/H1_gamma/1e6)        
        out += '     FID.bandwidth (Hz)    = {}\n'.format(self.bandwidth)
        out += '     FID.dwelltime (s)     = {}\n'.format(self.dwellTime)
        out += '     FID.echotime (s)      = {}\n'.format(self.echotime)

        return out

    
    # Acquisition parameters
    def set_acquisition_params(self,header):
        """
          Set useful params for fitting

          Parameters
          ----------
          header: dict
                contains the keys: 'centralFrequency', 'bandwidth', 'echotime'

        """
        self.centralFrequency = header['centralFrequency']
        self.bandwidth        = header['bandwidth']
        self.echotime         = header['echotime']
        
        self.dwellTime        = 1/self.bandwidth; 
        self.timeAxis         = np.linspace(self.dwellTime,
                                            self.dwellTime*self.numPoints,
                                            self.numPoints)  
        self.frequencyAxis    = np.linspace(-self.bandwidth/2,
                                            self.bandwidth/2,
                                            self.numPoints)        
        self.ppmAxis          = misc.hz2ppm(self.centralFrequency,
                                            self.frequencyAxis,shift=False)
        self.ppmAxisShift     = misc.hz2ppm(self.centralFrequency,
                                            self.frequencyAxis,shift=True)
        self.ppmAxisFlip      = np.flipud(self.ppmAxisShift)
        # turn into column vectors
        self.timeAxis         = self.timeAxis[:,None]
        self.frequencyAxis    = self.frequencyAxis[:,None]

        # by default, basis setup like data
        self.set_acquisition_params_basis(self.dwellTime)


    def set_acquisition_params_basis(self,dwelltime):
        """
           sets basis-specific timing params
        """
        # Basis has different dwelltime
        self.basis_dwellTime     = dwelltime
        self.basis_bandwidth     = 1/dwelltime
        self.basis_frequencyAxis = np.linspace(-self.basis_bandwidth/2,
                                               self.basis_bandwidth/2,
                                               self.numPoints)
        self.basis_timeAxis      = np.linspace(self.basis_dwellTime,
                                               self.basis_dwellTime*self.numPoints,
                                               self.numPoints)

        
    def ppmlim_to_range(self,ppmlim=None):
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
            axis   = np.flipud(np.fft.fftshift(self.ppmAxisShift))
            first  = np.argmin(np.abs(axis[:int(self.numPoints/2)]-ppmlim[0]))
            last   = np.argmin(np.abs(axis[:int(self.numPoints/2)]-ppmlim[1]))
            if first>last:
                first,last = last,first
        else:
            first,last = 0,self.numPoints 

        return int(first),int(last)


    def resample_basis(self):
        """
           Sometimes the basis is simulated using different timings (dwelltime)
           This interpolates the basis to match the FID
        """
        # RESAMPLE BASIS FUNCTION
        bdt    = self.basis_dwellTime
        bbw    = 1/bdt
        bn     = self.basis.shape[0]
        
        bt     = np.linspace(bdt,bdt*bn,bn)-bdt
        fidt   = self.timeAxis.flatten()-self.dwellTime
        
        f      = interp1d(bt,self.basis,axis=0)
        newiFB = f(fidt)
        
        self.basis = newiFB
        

    def remove_reference_from_basis(self):
        """
           Some basis spectra include the reference spectrum
           This function regress out the ref spectrum   

           DO NOT USE THIS FUNCTION - HAS NOT BEEN TESTED
        """

        basis_freq  = np.fft.fft(self.basis,axis=0)
        first, last = self.ppmlim_to_range(ppmlim=(0-.05,0+0.05))
        basis_freq[first:last] = (np.mean(basis_freq[first-10:first])+np.mean(basis_freq[last:last+10]))/2
        self.basis  = np.fft.ifft(basis_freq,axis=0)
        
        #ref = np.exp(1j*self.timeAxis*self.centralFrequency*2.0*np.pi)
        #ref = ref@np.linalg.pinv(ref)@self.basis
        #self.basis = self.basis - ref

        
    # Helper functions
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
        

    def combine(self,metabs):
        """
         Create combined metabolite groups

          Parameters
          ----------

          metabs: list of lists
        """
        if metabs is not None:
            self.metab_groups = []
            for mgroups in metabs:
                group = []
                for m in mgroups:
                    group.append(m)
                self.metab_groups.append(group)
            
        
    # I/O functions
    @staticmethod 
    def read(filename,TYPE='RAW'):
        """
          Read data file

          Parameters
          ----------

          filename : string
          TYPE     : string

          Outputs
          -------

          numpy array : data 
          string list : header information 
        """
        if TYPE == 'RAW':
            data,header = io.readLCModelRaw(filename)
        else:
            raise Exception('Unknow file type {}'.format(TYPE))
        return data, header
    
    def read_data(self,filename,TYPE='RAW'):
        """
          Read data file and update acq params

          Parameters
          ----------

          filename : string
          TYPE     : string

        """
        self.FID, header   = self.read(filename,TYPE)
        self.numPoints     = self.FID.size

        if header['centralFrequency'] is None:
            header['centralFrequency'] = 123.2E6
            warnings.warn('Cannot determine central Frequency from input. Setting to default of 123.2E6 Hz')
        if header['bandwidth'] is None:
            header['bandwidth'] = 4000
            warnings.warn('Cannot determine bandwidth. Setting to default of 4000Hz.')


        # Calculate spectrum straight away
        self.Spec = np.fft.fft(self.FID)
        
        self.set_acquisition_params(header)
    
    def read_basis_files(self,basisfiles,TYPE='RAW',ignore=[]):
        """
           Reads basis file and extracts name of metabolite from file name
           Assumes .RAW files are FIDs (not spectra)
           Should change this to reading metabolite name from header
        """
        self.numBasis = 0
        self.basis = []
        self.names = []
        for iDx,file in enumerate(basisfiles):
            data,_ = self.read(file,TYPE)
            name = os.path.splitext(os.path.split(file)[-1])[-2]
            if name not in ignore:
                self.names.append(name)
                self.basis.append(data)
                self.numBasis +=1
        self.basis = np.asarray(self.basis).astype(np.complex).T
        self.basis = self.basis - self.basis.mean(axis=0)
    
    def read_basis_from_folder(self,folder,TYPE='RAW',ignore=[]):
        """
           Reads all .RAW files from folder assuming they are all metabolite FIDs
        """
        basisfiles = sorted(glob.glob(os.path.join(folder,'*.'+TYPE)))
        self.read_basis_files(basisfiles,ignore=ignore)

    def read_basis_from_file(self,filename):
        """
           Reads single basis (.BASIS) file assuming it is spectra (not FIDs)
        """
        self.basis, self.names, header = io.readLCModelBasis(filename,self.numPoints)

        if header['dwelltime'] is not None:
            self.set_acquisition_params_basis(header['dwelltime'])
            self.resample_basis()
        
        self.numBasis = len(self.names)

    def read_h2o(self,filename,TYPE='RAW'):
        """
           Reads H2O file 
        """
        self.H2O, header = self.read(filename, TYPE)
        
        
    #######################################################
    ################### Fitting functions #################
    #################################### ##################
    
    def fit(self, model='LCModel',method='Newton',ppmlim=None):
        """
           Fit given model to the FID or Spectrum

           Parameters
           ----------
           model  : string 
                options are: 'LCModel', 'GLM'
                'GLM' only fits concentrations
                'LCModel' fits concentrations, phi0, phi1, gamma, epsilon, [TBC]
         
           method : string
                options are: 'Newton', 'MH' (Metropolis Hastings)
                these do not apply for the GLM model           

           ppmlim  : tuple
                frequency bands over which the fitting is done. 
        """
        if model == 'LCModel':
            self.fit_LCModel(method=method,ppmlim=ppmlim)
        elif model == 'GLM':
            self.fit_simple_glm()
        else:
            raise Exception('Unknown model {}'.format(model))
      
    def fit_simple_glm(self,real=False):
        """
           Basic GLM fitting 

           Parameters:
           ----------
           real : bool 
                  Force the concentrations to be real by concatenation of real/imag data and basis

           Outputs:
           array-like : concentrations

        """

        if real:
            data   = np.append(np.real(self.FID),np.imag(self.FID),axis=0)
            desmat = np.append(np.real(self.basis),np.imag(self.basis),axis=0)            
            beta   = np.real(np.linalg.pinv(desmat)@data)
            print(beta)
        else:
            beta      = np.linalg.pinv(self.basis)@self.FID
        
        self.con       = beta
        self.pred      = self.basis@self.con
        self.con_names = dict(zip(self.names,self.con))

        return beta

    
        
     
    def fit_LCModel(self,method='Newton',ppmlim=None):
        """
           A simplified version of LCModel
        """
        
        # Set up
        err_func   = models.LCModel_err_freq        # error function
        jac        = models.LCModel_jac_freq        # gradient
        forward    = models.LCModel_forward_freq    # forward model
        data       = self.Spec.copy()               # data
        first,last = self.ppmlim_to_range(ppmlim)   # data range
                
        # Initialise all params
        #  simple GLM for concentrations
        x0  = self.fit_simple_glm(real=True) # np.abs(np.linalg.pinv(self.basis)@self.FID)

        
        #  Zero for the rest
        x0  = np.append(np.abs(x0),[0,0,0,0])
        constants = (self.frequencyAxis,
                     self.timeAxis,
                     self.basis,data,first,last)
        
        if method == 'Newton':
            # Bounds
            bnds = []
            for i in range(self.numBasis):
                bnds.append((0,None))
            bnds.append((0,None))
            bnds.append((None,None))
            bnds.append((None,None))
            bnds.append((None,None))            
            res = minimize(err_func, x0, args=constants, method='TNC',jac=jac,bounds=bnds)
            x   = res.x
            self.reset_params(x)
            
            # Calc covariance
            #cf = lambda x : models.LCModel_err_freq(x,self.frequencyAxis,
            #                                        self.timeAxis,self.basis,self.Spec,first,last)        
            #hess = misc.hessian(x,cf)
            #noise_std = np.sqrt(np.sum(np.absolute(self.pred[first:last]-self.Spec[first:last])**2))
            #self.all_params_cov = np.linalg.inv(hess)/2/noise_std
            #self.hess = hess
  
        elif method == 'MH':
            forward_mh = lambda p : forward(p,self.frequencyAxis,self.timeAxis,self.basis)
            numPoints_over_2  = (last-first)/2.0
            y      = data[first:last]
            loglik = lambda  p : np.log(np.linalg.norm(y-forward_mh(p)[first:last]))*numPoints_over_2            
            logpr  = lambda  p : 0 # uniform priors for now

            # Setup the fitting
            p0   = self.fit_LCModel(method='Newton',ppmlim=ppmlim) 
            LB   = np.zeros(self.numBasis)
            LB   = np.append(LB,-np.inf*np.ones(4))
            UB   = np.inf*np.ones(p0.size)

            # Do the fitting
            mh           = MH(loglik,logpr,burnin=100,njumps=1000)
            self.samples = mh.fit(p0,LB=LB,UB=UB,verbose=False)

            # some stats on the posterior distribution
            self.all_params_cov = np.cov(self.samples)
            x            = self.samples.mean(axis=0)
            self.reset_params(x)
            self.params_std = self.samples.std(axis=0)
        else:
            raise Exception('Unknown optimisation method.')
                
                   
        return x


    def calc_baseline(self,spec=None,ppmlim=(0,4.6),order=10):
        """
        Estimate baseline
        
        parameters
        ----------
        spec : array-like
               spectrum to use for estimating baseline. default: uses self.Spec
               
        ppmlim : tuple
                 upper and lower limit over which spectrum is calculated
        order : integer
                order of polynomial used to estimate baseline
        """
        # Get axes
        axis   = np.flipud(self.ppmAxisFlip)
        first  = np.argmin(np.abs(axis-ppmlim[0]))
        last   = np.argmin(np.abs(axis-ppmlim[1]))
        if first>last:
            first,last = last,first    
        freq       = axis[first:last]
        # Build design matrix
        desmat = []
        for i in range(order+1):
            regressor  = freq**i                     # power
            if i>0:
                regressor -= np.mean(regressor)      # demean
            regressor /= np.linalg.norm(regressor)   # normalise
            desmat.append(regressor.flatten())
        desmat = np.asarray(desmat).T
        # Append basis to design matrix so it doesn't
        # model out good signal
        # First, do a quick nonlinear fit:
        self.fit_LCModel(method='Newton',ppmlim=ppmlim)
        basis  = np.exp(-1j*(self.phi0+self.phi1*self.frequencyAxis))*np.fft.fft(self.basis*np.exp(-(self.gamma+1j*self.eps)*self.timeAxis),axis=0)
        basis  = np.flipud(np.fft.fftshift(basis))
        basis  = basis[first:last,:]      
        desmat = np.concatenate((desmat,basis),axis=1)
        if spec is None:
            spec     = self.Spec
        spec = np.flipud(np.fft.fftshift(spec))
        beta = np.matmul(np.linalg.pinv(desmat),spec[first:last])
    
        # Model is:
        # data = [nuisance basis]*beta
        # so baseline = nuisance*beta[:order+1]
    
        baseline = np.zeros(self.numPoints,dtype='complex')
        baseline[first:last] = np.matmul(desmat[:,:order+1],beta[:order+1])
        baseline = np.flipud(baseline)
        baseline = np.fft.fftshift(baseline)
    
        return baseline

    def reset_params(self,x):
        """
           Set params and recalculate model prediction
           
           Parameters
           ----------

           x : array-like
             Parameters are in the followin order: [concentrations,gamma,epsilon,phi0,phi1]
        """
    
        # Keep track of all params
        self.all_params =x
        # Split into groups of params
        self.con   = x[:self.numBasis]
        self.gamma = x[self.numBasis]
        self.eps   = x[self.numBasis+1]
        self.phi0  = x[self.numBasis+2]
        self.phi1  = x[self.numBasis+3]
        
        # Readable mean concentrations
        self.con_names = dict(zip(self.names,self.con))
        self.pred      = models.LCModel_forward(x,self.frequencyAxis,self.timeAxis,self.basis)



        
    # Quantification
    def init_quantification(self,T2s=None,volfrac=None):
        """
           Set params useful for absolute quantification
        """

        # Set H2O quantif params
        self.quantif.T2s     = np.asarray(T2s)
        self.quantif.volfrac = np.asarray(volfrac)
        self.quantif.TE      = self.echotime
        self.quantif.H2O     = self.H2O
        self.quantif.Cr      = self.basis[:,self.names.index('Cr')]
        self.quantif.PCr     = self.basis[:,self.names.index('PCr')]
            
        # set up all metabolites
        con_names = dict(zip(self.names,self.con))
        if self.metab_groups is not None:
            for mgroup in self.metab_groups:
                name = '+'.join(mgroup)
                con  = 0
                for m in mgroup:
                    con += con_names[m]
                con_names[name] = con 

        self.quantif.con_names = con_names
    
    def rescale_concentrations(self,metab=None,scale=1.0):
        """
           Rescales all concentrations 
        """
        if self.quantif.con_names == None:
            self.init_quantification(no_h2o=True)

        if isinstance(metab,list):
            self.all_con_names = self.quantif.rescale_to_metab_grp(metab_list=metab,scale=scale)
        else:
            self.all_con_names = self.quantif.rescale_to_metab(metab=metab,scale=scale)
        

    def rescale_concentrations_to_h2o(self):
        """
           Rescales all concentrations to water peak

           Using equations from: Gasparovic et al. MRM 2006 55:1219-1226
        """

        #h2o_signal = 
        self.all_con_names_h2o = self.quantif.rescale_to_h2o(self.ppmAxisShift)
        
        return
    
    def all_metab_toTotalCr(self):
        """
           Rescales all metabolites and groups of metabolites such that Cr+PCr=8.0
        """
        self.rescale_concentrations_to_TotalCr()
        self.concentration_to_TotalCr['Cr+PCr'] = self.concentration_to_TotalCr['Cr']
        self.concentration_to_TotalCr += self.concentration_toTotalCr['PCr']
        if self.metab_groups is not None:
            for mgroup in self.metab_groups:
                name = '+'.join(mgroup)
                con  = 0
                for m in mgroup:
                    con += self.concentration_to_TotalCr[m]
                self.concentration_to_TotalCr[name] = con 

    def post_process(self,metab,T2s=None,volfrac=None,scale=1.0):
        """
           Post processing includes:
           - Rescaling to Cr+PCr=8.0
           - Quantification relative to water
           - etc TBC
        """
        self.init_quantification(T2s=T2s,volfrac=volfrac)
        self.rescale_concentrations(metab=metab,scale=scale)
        self.rescale_concentrations_to_h2o()
        
        return 


    def save_results_to_file(self,filename):
        """
           Write concentrations (abs and relative) to text file
        """
        header = 'metabolite,Conc,/Cr+PCr\n'
        with open(filename,'w') as f:
            f.write(header)
            for met in self.all_con_names:
                x,y,z = met,self.all_con_names_h2o[met],self.all_con_names[met]
                f.write('{},{},{}\n'.format(x,y,z))

    
    def save_fit_to_figure(self,filename,ppmlim=(.4,4.2)):
        """
           Save fit to figure
        """
        if self.pred is None:
            raise Exception('Cannot plot fit before fitting')

        axis   = np.flipud(self.ppmAxisFlip)
        spec   = np.flipud(np.fft.fftshift(self.Spec))
        pred   = np.fft.fft(self.pred)
        pred   = np.flipud(np.fft.fftshift(pred))

        if self.baseline is not None:
            B = np.flipud(np.fft.fftshift(self.baseline))
    
        first  = np.argmin(np.abs(axis-ppmlim[0]))
        last   = np.argmin(np.abs(axis-ppmlim[1]))
        if first>last:
            first,last = last,first    
        freq = axis[first:last] 

        plt.figure(figsize=(9,10))
        plt.plot(axis[first:last],spec[first:last])
        plt.gca().invert_xaxis()
        plt.plot(axis[first:last],pred[first:last],'r')
        if self.baseline is not None:
            plt.plot(axis[first:last],B[first:last],'k')

        # style stuff
        plt.minorticks_on()
        plt.grid(b=True, axis='x', which='major',color='k', linestyle='--', linewidth=.3)
        plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':',linewidth=.3)

        # Save to file
        plt.savefig(filename)
        return plt.gcf()

        

class MRS_quantif(object):
    """
       Deals with quantification
       e.g. normalising by specific metabolites or water
    """
    def __init__(self, names=None, T2s=None, volfrac=None,TE=None):
        
        self.T2s        = T2s        # T2s for GM/WM/CSF
        self.volfrac    = volfrac    # volume fractions of GM/WM/CSF
        self.TE         = TE
        self.con_names  = None       # names and un-normalised concentrations
        self.H2O        = None
        self.Cr         = None
        self.PCr        = None
        
    def rescale_to_metab(self,metab=None,scale=1.0,ref=None):
        """
           Rescales all concentrations to one of the metabolites
           Results in the concentration of the chosen metabolite to be = scale
        """

        if ref is None:
            ref = self.con_names[metab]
        
        if ref > 0 :
            con = {k: v/ref*scale for k, v in self.con_names.items()}
            return con
        else:
            warnings.warn("[{}]=0!! Something went wrong somewhere. Can't rescale concentrations...".format(metab))
            return self.con

        
    def rescale_to_metab_grp(self,metab_list,scale=1.0):
        """
           rescale to total concentration of a metabolite group
        """

        ref = 0
        for metab in metab_list:
            ref += self.con_names[metab]
        con = self.rescale_to_metab(ref=ref,scale=scale)
        
        return con

    def rescale_to_h2o(self,ppmaxisshift=None):
        
        """
           Rescales all concentrations to water peak

           Using equations from: Gasparovic et al. MRM 2006 55:1219-1226

           [M] = S_M/S_H2O*2/HM*[H2O]
        """

        if self.volfrac is None:
            raise Exception('Cannot rescale to water without GM/WM/CSF volume fractions')
        if self.T2s is None:
            raise Exception('Cannot rescale to water without T2s')
        if self.TE is None:
            raise Exception('Cannot rescale to water without echo time')

        # Correct volume fractions for densities
        densities = np.array((0.78,0.65,0.97))
        ftotal    = np.sum(self.volfrac * densities)
        frac      = self.volfrac * densities / ftotal

        # Relaxation
        R  = np.exp(-self.TE/self.T2s[1:])  # GM/WM/CSF
        RM = np.exp(-self.TE/self.T2s[0])   # Metab
        
        
        # rescale
        Cr  = self.con_names['Cr']
        PCr  = self.con_names['PCr']
        toTotCr_ratio = {k: v/(Cr+PCr) for k, v in self.con_names.items()}
        # Use Cr+PCr
        
        interval   = np.ones(self.Cr.size)
        if ppmaxisshift is not None:
            interval[ppmaxisshift<2.5] = 0
            interval[ppmaxisshift>4.5] = 0
            self.Cr = self.Cr*interval
            self.Pcr = self.PCr*interval
            
        Cr_area    = np.sum(np.abs(self.con_names['Cr']*self.Cr))/interval.sum()
        PCr_area   = np.sum(np.abs(self.con_names['PCr']*self.PCr))/interval.sum()
        TotCr_area = PCr_area+Cr_area
        H2O_area = np.sum(np.real(self.H2O))/self.H2O.size
        
        CrH2O_ratio = Cr_area/H2O_area
        frac_ratio  = np.sum(frac*R)/(1-frac[-1])/RM*H2O_Conc

        H2O_protons    = 2
        TotCr_protons  = 10

        absQuantifFactor = CrH2O_ratio*frac_ratio*H2O_protons/TotCr_protons
        
        rescaled = {k: v*absQuantifFactor for k,v in toTotCr_ratio.items()}

        return rescaled

