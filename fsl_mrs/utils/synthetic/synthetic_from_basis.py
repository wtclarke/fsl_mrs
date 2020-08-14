# synthetic.py - Create synthetic data basis sets
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.utils.misc import ts_to_ts,FIDToSpec,SpecToFID,rescale_FID
from fsl_mrs.utils import mrs_io
def standardConcentrations(basisNames):
    """Return standard concentrations for 1H MRS brain metabolites for those which match basis set names."""
    # These defaults are from the average of the MRS fitting challenge
    standardconcs = {'Ala':0.60,
                    'Asc':1.20,
                    'Asp':2.40,
                    'Cr':4.87,
                    'GABA':1.20,
                    'Glc':1.20,
                    'Gln':3.37,
                    'Glu':12.41,
                    'GPC':0.74,
                    'GSH':1.20,
                    'Gly':1.20,
                    'Ins':7.72,
                    'Lac':0.60,
                    'NAA':13.80,
                    'NAAG':1.20,
                    'PCho':0.85,
                    'PCh':0.85,
                    'PCr':4.87,
                    'PE':1.80,
                    'sIns':0.30,
                    'Scyllo':0.30,
                    'Tau':1.80}
    concs = []
    for name in basisNames:
        if name in standardconcs:
            concs.append(standardconcs[name])
        else:
            print(f'{name} not in standard concentrations. Setting to random between 1 and 5.')
            concs.append(np.random.random()*(5.0-1.0)+1.0)

    return concs

def syntheticFromBasisFile(basisFile,                        
                            concentrations=None,
                            broadening = (9.0,0.0),
                            shifting= 0.0,
                            baseline = [0,0],
                            coilamps = [1.0],
                            coilphase = [0.0],
                            noisecovariance =[[0.1]],
                            bandwidth = 4000.0,
                            points = 2048):
    """ Create synthetic data from a set of FSL-MRS basis files.

    Args:
            basisFile (str): path to directory containg basis spectra json files
            concentrations (list or dict or None, optional ): If None, standard concentrations will be used.
                                                        If list of same length as basis spectra, then these will be used.
                                                        Pass dict to overide standard values for specific metabolites. Key should be metabolite name.
            broadening (list of tuples or tuple:floats, optional): Tuple containg a gamma and sigma or a list of tuples for 
                                                                    each basis.
            shifting (list of floats or float, optional): Eps shift value or a list for each basis.
            baseline (list of floats, optional): Baseline parameters. Not yet implemented
            coilamps (list of floats, optional): If multiple coils, specify magnitude scaling.
            coilphase (list of floats, optional): If multiple coils, specify phase.
            noisecovariance (list of floats, optional): N coils x N coils array of noise variance/covariance.
            bandwidth (float,optional): Bandwidth of output spectrum in Hz
            points (int,optional): Number of points in output spectrum. 

    Returns:
        FIDs: Numpy array of synthetic FIDs
        outHeader: Header suitable for loading FIDs into MRS object.
        concentrations: Final concentration scalinfs
    """
    basis,names,header = mrs_io.read_basis(basisFile)

    if concentrations is None:
        concentrations = standardConcentrations(names)
    elif isinstance(concentrations,(list,np.ndarray)):
        if len(concentrations) != len(names):
            raise ValueError(f'Concentrations must have the same number of elements as basis spectra. {len(concentrations)} concentrations, {len(names)} basis spectra.')
    elif isinstance(concentrations,dict):
        newconcs = []
        for name in names:
            if name in concentrations:
                newconcs.append(concentrations[name])
            else:
                newconcs.extend(standardConcentrations([name]))
        concentrations = newconcs
    else:
        raise ValueError('Concentrations must be None, a list or a dict containing overides for particular metabolites.')

    
    FIDs = syntheticFromBasis(basis,
                        header[0]['bandwidth'],
                        concentrations,
                        broadening = broadening,
                        shifting= shifting,
                        baseline = baseline,
                        coilamps = coilamps,
                        coilphase = coilphase,
                        noisecovariance =noisecovariance,
                        bandwidth = bandwidth,
                        points = points)

    outHeader = {'bandwidth':bandwidth,'centralFrequency':header[0]['centralFrequency']}
    return FIDs,outHeader,concentrations

def syntheticFromBasis(basis,
                        basis_bandwidth,
                        concentrations,
                        broadening = (9.0,0.0),
                        shifting= 0.0,
                        baseline = [0,0],
                        coilamps = [1.0],
                        coilphase = [0.0],
                        noisecovariance =[[0.1]],
                        bandwidth = 4000.0,
                        points = 2048):
    """ Create synthetic spectra from basis FIDs. Use syntheticFromBasisFile interface."""
    # sort out inputs
    numMetabs = basis.shape[1]
    if len(concentrations)!=numMetabs:
        raise ValueError('Provide a concentration for each basis spectrum.')

    if isinstance(broadening,list):
        if len(broadening)!=numMetabs:
            raise ValueError('Broadening values must be either a single tuple or list of tuples with the same number of elements as basis sets.')
        gammas = [b[0] for b in broadening]
        sigmas = [b[1] for b in broadening]
    elif isinstance(broadening,tuple):
        gammas = [broadening[0]]*numMetabs
        sigmas = [broadening[1]]*numMetabs
    else:
        raise ValueError('Broadening values must be either a single tuple or list of tuples with the same number of elements as basis sets.')

    if isinstance(shifting,list):
        if len(shifting)!=numMetabs:
            raise ValueError('shifting values must be either a float or list with the same number of elements as basis sets.')
        eps = shifting
    elif isinstance(shifting,float):
        eps = [shifting]*numMetabs
    else:
        raise ValueError('shifting values must be either a float or list with the same number of elements as basis sets.')

    # Form noise vectors
    ncoils = len(coilamps)
    noisecovariance = np.asarray(noisecovariance)
    if len(coilphase) != ncoils:
        raise ValueError('Length of coilamps and coilphase must match.')
    if noisecovariance.shape != (ncoils,ncoils):
        raise ValueError('noisecovariance must be ncoils x ncoils.')

    noise = np.random.multivariate_normal(np.zeros((ncoils)), noisecovariance, points) + 1j*np.random.multivariate_normal(np.zeros((ncoils)), noisecovariance, points)

    # Interpolate basis
    dwelltime = 1/bandwidth
    basis_dwelltime= 1/basis_bandwidth
    basis = ts_to_ts(basis,basis_dwelltime,dwelltime,points)
    # basis = rescale_FID(basis,scale=100)

    # Create the spectrum
    baseFID    = np.zeros((points),np.complex128)
    dwellTime   = 1/bandwidth
    timeAxis    = np.linspace(dwellTime,
                              dwellTime*points,
                              points)
    for b,c,e,g,s in zip(basis.T,concentrations,eps,gammas,sigmas):
        tmp   = b*np.exp(-(1j*e+g+timeAxis*s**2)*timeAxis)
        M     = FIDToSpec(tmp)
        baseFID += SpecToFID(M*c)
        

    # Add baseline
    # TO DO

    # Add noise and write tot output list
    FIDs = []
    for cDx,(camp,cphs) in enumerate(zip(coilamps,coilphase)):
        FIDs.append((camp*np.exp(1j*cphs)*baseFID)+noise[:,cDx])
    FIDs = np.asarray(FIDs).T
    FIDs = np.squeeze(FIDs)

    return FIDs