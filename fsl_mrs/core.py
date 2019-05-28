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


import numpy as np
import time
from scipy.optimize import minimize

#------------------------------------------------
#
#
#------------------------------------------------


# TODOs
#  - add Gaussian blurring
#  - add B-spline or other 'background' fitting 
#  - add fitting mask
#  - water normalisation
#  - fix discrepancies between data and basis


## ASK MICHIEL/PAUL HOW BEST TO SET UP GLOB VARIABLES
H2O_MOLECULAR_MASS = 18.01528   # g/mol
H2O_Conc           = 55.51E3    # mmol/kg      
H2O_PPM_TO_TMS     = 4.65       # Shift of water to Tetramethylsilane
H2O_to_Cr          = 0.4        # Proton ratio

class MRS(object):
    """
      MRS Class - deals with data I/O and model fitting
    """
    def __init__(self):
        # Data
        self.FID               = None
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
        self.names             = None
        self.metab_groups      = None
        self.T2s               = None
        # Modelling variables
        self.pred              = None   # Predicted FID
        self.con               = None   # Concentrations
        self.eps               = None   # peak shift
        self.gamma             = None   # peak spread
        self.phi0              = None   # phase shift
        self.phi1              = None   # phase ramp
        self.samples           = None   # MCMC samples
        self.con_names         = None   # zipped con and names
        self.all_con_names     = None   # zipped con and names for all metabolites and metab groups
        self.all_con_names_h2o = None   # zipped con and names for all metabolites and metab groups
        self.all_con_names_std = None   # zipped std and names for all metabolites and metab groups
        # Quantification
        self.quantif           = MRS_quantif() # Instantiate quantif object
        
    # Acquisition parameters
    def set_acquisition_params(self,header):
        """
          Set useful params for fitting

          Parameters
          ----------
          header: dict

        """
        self.centralFrequency = header['centralFrequency']
        self.bandwidth        = header['bandwidth']
        self.echotime         = header['echotime']
        
        self.dwellTime     = 1/self.bandwidth; 
        self.timeAxis      = np.linspace(self.dwellTime,
                                         self.dwellTime*self.numPoints,
                                         self.numPoints)  
        self.frequencyAxis = np.linspace(-self.bandwidth/2,
                                         self.bandwidth/2,
                                         self.numPoints)  
        self.ppmAxis       = misc.hz2ppm(self.centralFrequency,
                                         self.frequencyAxis,shift=False)
        self.ppmAxisShift  = misc.hz2ppm(self.centralFrequency,
                                         self.frequencyAxis,shift=True) 
        self.timeAxis      = self.timeAxis[:,None]
        self.frequencyAxis = self.frequencyAxis[:,None]


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
        self.basis, self.names, _ = io.readLCModelBasis(filename,self.numPoints)
        self.numBasis = len(self.names)

    def read_h2o(self,filename,TYPE='RAW'):
        """
           Reads H2O file 
        """
        self.H2O, header = self.read(filename, TYPE)
        
        
    
    # Fitting functions
    def fit(self, model='LCModel',method='Newton',domain='time',ppmlim=None):
        """
           Fit given model to the FID or Spectrum

           Parameters
           ----------
           model  : string 
                options are: 'LCModel', 'GLM'
                'GLM' only fits concentrations
                'LCModel' fits concentrations, phi0, phi1, gamma, epsilon, [TBC]
         
           method : string
                options are: 'Newton', 'Powell', 'MH' (Metropolis Hastings)
                these do not apply for the GLM model
           
           domains : string
                either 'time' or 'frequency'

           ppmlim  : tuple
                frequency bands over which the fitting is done. Only applies if domain='frequency'
        """
        if model == 'LCModel':
            self.fit_LCModel(method=method,domain=domain,ppmlim=ppmlim)
        elif model == 'GLM':
            self.fit_simple_glm()
        else:
            raise Exception('Unknown model {}'.format(model))
      
    def fit_simple_glm(self):
        """
           Basic GLM fitting 
        """
        beta      = np.linalg.pinv(self.basis)@self.FID
        self.con  = beta
        self.pred = self.basis@self.con
        self.concentrations = dict(zip(self.names,self.con))
        return beta
    
     
    def fit_LCModel(self,method='Newton',domain='time',ppmlim=None):
        """
           A simplified version of LCModel
        """
        # Different set up depending on fitting domain
        if domain == 'time':
            err_func = models.LCModel_err
            jac      = models.LCModel_jac
            forward  = models.LCModel_forward
            data     = self.FID
            first    = 0
            last     = self.numPoints
        elif domain == 'frequency':
            err_func = models.LCModel_err_freq
            jac      = models.LCModel_jac_freq
            forward  = models.LCModel_forward_freq
            data     = np.fft.fft(self.FID)
            if ppmlim is not None:
                first = np.argmin(np.abs(self.ppmAxisShift-ppmlim[0]))
                last  = np.argmin(np.abs(self.ppmAxisShift-ppmlim[1]))
            else:
                first    = 0
                last     = self.numPoints
        else:
            raise Exception('Unknown domain {}'.format(domain))
                
        # Initialise all params to zero
        x0  = np.abs(np.linalg.pinv(self.basis)@data)
        x0  = np.append(x0,[0,0,0,0])
        constants = (self.numBasis,self.frequencyAxis[first:last],
                     self.timeAxis[first:last],
                     self.basis[first:last,:],data[first:last])
        
        if method == 'Powell':
            res = minimize(err_func, x0, args=constants, method='Powell')
            x   = res.x
        elif method == 'Newton':
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
        elif method == 'MH':
            forward = lambda p : forward(p,self.numBasis,self.frequencyAxis,self.timeAxis,self.basis)
            loglik = lambda p : np.log(np.linalg.norm(data-forward(p)))*self.numPoints/2
            logpr  = lambda p : 0

            p0   = self.fit_LCModel(method='Newton',domain=domain,pplim=pplim) 
            LB   = np.zeros(self.numBasis)
            LB   = np.append(LB,-np.inf*np.ones(4))
            UB   = np.inf*np.ones(p0.size)

            mh           = MH(loglik,logpr,burnin=100,njumps=1000)
            self.samples = mh.fit(p0,LB=LB,UB=UB,verbose=False)
            x            = self.samples.mean(axis=0)
        else:
            raise Exception('Unknown optimisation method.')
        
        
        # Get params
        self.reset_params(x)
        
         
        return x
    
    def reset_params(self,x):
        """
           Set params and recalculate model prediction
           
           Parameters
           ----------

           x : array-like
             Parameters are in the followin order: [concentrations,gamma,epsilon,phi0,phi1]
        """
        # Get params
        self.con   = x[:self.numBasis]
        self.gamma = x[self.numBasis]
        self.eps   = x[self.numBasis+1]
        self.phi0  = x[self.numBasis+2]
        self.phi1  = x[self.numBasis+3]
        
        # Readable mean concentrations
        self.concentrations = dict(zip(self.names,self.con))
        self.pred           = models.LCModel_forward(x,self.numBasis,self.frequencyAxis,self.timeAxis,self.basis)



    # Quantification
    def init_quantification(self,T2s=None,volfrac=None):

        # Set H2O quantif params
        self.quantif.T2s     = np.asarray(T2s)
        self.quantif.volfrac = np.asarray(volfrac)
        self.quantif.TE      = self.echotime
        self.quantif.H2O     = self.H2O
        self.quantif.Cr      = self.basis[:,self.names.index('Cr')] 
            
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
        self.all_con_names_h2o = self.quantif.rescale_to_h2o()
        
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

    def rescale_to_h2o(self):
        
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
        toToCr_ratio = {k: v/Cr for k, v in self.con_names.items()}
        # Below hack for Cr area assuming the two peaks are 2to3
        Cr_area = np.sum(np.abs(self.con_names['Cr']*self.Cr))*3/5
        H2O_area = np.sum(np.real(self.H2O))
        
        CrH2O_ratio = Cr_area/H2O_area
        frac_ratio  = np.sum(frac*R)/(1-frac[-1])/RM*H2O_Conc

        H2O_protons = 2
        Cr_protons  = 5

        absQuantifFactor = CrH2O_ratio*frac_ratio*H2O_protons/Cr_protons
        
        rescaled = {k: v*absQuantifFactor for k,v in toToCr_ratio.items()}

        return rescaled
