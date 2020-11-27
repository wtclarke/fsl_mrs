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


def dMRS(mouse='mouse1'):
    """
    Load Diffusion MRS data from hard-coded location on disk
    Args:
        mouse: 'mouse1' ... 'mouse10'

    Returns:
        mrs Object list
        list (time variable)
    """
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


    currentdir = os.path.join(dataPath,mouse)

    
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
        mrs = MRS(FID=fid,cf=centralFrequency,bw=bandwidth,basis=basis,names=names,basis_hdr=header[0],nucleus='1H')
        mrs.check_FID(repair=True)
        mrs.check_Basis(repair=True)
        mrs.ignore(['Gly'])
        mrsList.append(mrs)

    mrsList[0].rescaleForFitting()
    for i, mrs in enumerate(mrsList):
        if i > 0:
            mrs.FID *= mrsList[0].scaling['FID']
            mrs.basis *= mrsList[0].scaling['basis']

    blist = [b / 1e3 for b in blist]
    return mrsList,blist


def dMRS_SNR(avg=1):
    """
    Load DMRS data with different numbers of averages
    Args:
        avg: int (1,2,4,8,16,32,64,128)

    Returns:
        list of MRS objects
        bvals

    """
    from fsl_mrs.utils import mrs_io

    basispath = '/Users/saad/Desktop/Spectroscopy/DMRS/basis_STE_LASER_8_25_50_LacZ.BASIS'
    basis,names,Bheader = mrs_io.read_basis(basispath)
    # Bheader['ResonantNucleus'] = '1H'
    centralFrequency = 500.30
    bandwidth = 5000

    FIDpath = f'/Users/saad/Desktop/Spectroscopy/DMRS/WT_multi/{avg:03}_avg'
    bvals   = [20,3020,6000,10000,20000,30000,50000]
    MRSlist = []
    for b in bvals:
        FID,FIDheader = mrs_io.read_FID(FIDpath+f'/b_{b:05}.nii.gz')
        MRSArgs = {'header': FIDheader, 'basis': basis, 'names': names, 'basis_hdr': Bheader[0]}
        mrs = MRS(FID=FID, **MRSArgs)
        MRSlist.append(mrs)

    MRSlist[0].rescaleForFitting()
    for i,mrs in enumerate(MRSlist):
        if i>0:
            mrs.FID *= MRSlist[0].scaling['FID']
            mrs.basis *= MRSlist[0].scaling['basis']

    bvals = [b / 1e3 for b in bvals]

    return MRSlist,bvals


def FMRS(smooth=False):
    """
    Load Functional MRS data from hard-coded location on disk
    Args:
        smooth: bool

    Returns:
        mrs Object list
        list (time variable)
    """
    import glob
    from numpy import repeat

    folder = '/Users/saad/Desktop/Spectroscopy/FMRS/Jacob_data'
    filelist = glob.glob(os.path.join(folder, 'RAWFORMAT', 'run_???.RAW'))

    FIDlist = []
    for file in filelist:
        FID, FIDheader = mrs_io.read_FID(file)
        FIDlist.append(FID)

    basisfile = os.path.join(folder, '7T_slaser36ms_2013_oxford_tdcslb1_ivan.BASIS')
    basis, names, basisheader = mrs_io.read_basis(basisfile)

    # # Resample basis
    from fsl_mrs.utils import misc
    basis = misc.ts_to_ts(basis, basisheader[0]['dwelltime'], FIDheader['dwelltime'], FID.shape[0])

    if smooth:
        sFIDlist = misc.smooth_FIDs(FIDlist, window=5)
    else:
        sFIDlist = FIDlist

    MRSargs = {'names': names, 'basis': basis, 'basis_hdr': basisheader[0], 'bw': FIDheader['bandwidth'],
               'cf': FIDheader['centralFrequency']}

    mrsList = []
    for fid in sFIDlist:
        mrs = MRS(FID=fid, **MRSargs)
        mrsList.append(mrs)

    # Stimulus variable
    stim = repeat([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 16, axis=0).flatten()

    return mrsList,stim

def MPRESS(noise=1):
    """
    Load MegaPress edited MRS data from hard-coded location on disk
    Args:
        noise: float (noise std)

    Returns:
        mrs Object list
        list (time variable)
    """
    from fsl_mrs.utils import mrs_io
    from fsl_mrs.utils.synthetic.synthetic_from_basis import syntheticFromBasisFile
    mpress_on = '/Users/saad/Desktop/Spectroscopy/mpress_basis/ON'
    mpress_off = '/Users/saad/Desktop/Spectroscopy/mpress_basis/OFF'

    basis,names,basis_hdr = mrs_io.read_basis(mpress_on)
    FIDs,header,conc = syntheticFromBasisFile(mpress_on,noisecovariance=[[noise]])
    mrs1 = MRS(FID=FIDs,header=header,basis=basis,basis_hdr=basis_hdr[0],names=names)
    mrs1.check_FID(repair=True)
    mrs1.Spec = misc.FIDToSpec(mrs1.FID)
    mrs1.check_Basis(repair=True)

    basis,names,basis_hdr = mrs_io.read_basis(mpress_off)
    FIDs,header,conc = syntheticFromBasisFile(mpress_off,noisecovariance=[[noise]])
    mrs2 = MRS(FID=FIDs,header=header,basis=basis,basis_hdr=basis_hdr[0],names=names)
    mrs2.check_FID(repair=True)
    mrs2.Spec = misc.FIDToSpec(mrs2.FID)
    mrs2.check_Basis(repair=True)

    return [mrs1,mrs2],[0,1]