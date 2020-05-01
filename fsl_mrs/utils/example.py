#!/usr/bin/env python

# Get example dataset
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <will.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT



import os

from fsl_mrs.utils import mrs_io,misc
from fsl_mrs.core import MRS

def simulated(ID=1):
    """
    ID = 1 up to 28
    """
    fileDir = os.path.dirname(__file__)
    datafolder  = os.path.join(fileDir,'../pkg_data/mrs_fitting_challenge/datasets_JMRUI')
    basisfolder = os.path.join(fileDir,'../pkg_data/mrs_fitting_challenge/basisset_JMRUI')

    # Load data and basis
    FID,FIDheader = mrs_io.read_FID(os.path.join(datafolder,f'dataset{ID}_WS.txt'))
    FIDW,_ = mrs_io.read_FID(os.path.join(datafolder,f'dataset{ID}_nWS.txt'))
    basis,names,Bheader = mrs_io.read_basis(basisfolder)
  
    MRSArgs = {'header':FIDheader,'basis':basis,'names':names,'basis_hdr':Bheader[0],'H2O':FIDW}
    
    mrs = MRS(FID=FID,**MRSArgs)
    # Check orientation and rescale for extra robustness
    mrs.processForFitting()

    return mrs


def dMRS():
    from scipy.io import loadmat
    from fsl_mrs.utils.preproc.phasing import phaseCorrect
    from fsl_mrs.utils.preproc.align import phase_freq_align
    from fsl_mrs.utils.preproc.shifting import shiftToRef
    import numpy as np
    
    dataPath = '/Users/saad/Desktop/Spectroscopy/DMRS/WT_High_b'
    basispath = '/Users/saad/Desktop/Spectroscopy/DMRS/basis_STE_LASER_8_25_50_LacZ.BASIS'
    basis,names,header = mrs_io.read_basis(basispath)

    centralFrequency = 500.30
    bandwidth = 5000


    currentdir = os.path.join(dataPath,'mouse1')

    
    fidList = []
    blist = [20,3000,6000,10000,20000,30000,50000]
    for b in blist:
        file = os.path.join(currentdir,'high_b_'+str(b)+'.mat')
        tmp = loadmat(file)
        fid =  np.squeeze(tmp['soustraction'].conj())
        fid,_,_ = phaseCorrect(fid,bandwidth,centralFrequency,ppmlim=(2.8,3.2),shift=True)        
        fidList.append(fid) 
    
    # Align and shift to Cr reference.
    alignedFids,phiOut,epsOut = phase_freq_align(fidList,bandwidth,centralFrequency,ppmlim=(0.2,4.2),niter=2)

    mrsList = []
    for fid,b in zip(alignedFids,blist):
        fid,_ = shiftToRef(fid,3.027,bandwidth,centralFrequency,ppmlim=(2.9,3.1))
        mrs = MRS(FID=fid,cf=centralFrequency,bw=bandwidth,basis=basis,names=names,basis_hdr=header[0])
        mrs.check_FID(repair=True)
        mrs.check_Basis(repair=True)
        mrs.ignore(['Gly'])
        mrsList.append(mrs)

    return mrsList,blist
        
