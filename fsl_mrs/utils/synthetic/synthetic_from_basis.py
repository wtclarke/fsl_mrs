# synthetic.py - Create synthetic data basis sets
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.utils.misc import SpecToFID, parse_metab_groups
from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.baseline import prepare_polynomial_regressor
from fsl_mrs.core import MRS
from fsl_mrs.core.basis import Basis
from fsl_mrs import models
from pathlib import Path


def standardConcentrations(basisNames):
    """Return standard concentrations for 1H MRS brain metabolites for those which match basis set names."""
    # These defaults are from the average of the MRS fitting challenge
    standardconcs = {'Ala': 0.60,
                     'Asc': 1.20,
                     'Asp': 2.40,
                     'Cr': 4.87,
                     'GABA': 1.20,
                     'Glc': 1.20,
                     'Gln': 3.37,
                     'Glu': 12.41,
                     'GPC': 0.74,
                     'GSH': 1.20,
                     'Gly': 1.20,
                     'Ins': 7.72,
                     'Lac': 0.60,
                     'NAA': 13.80,
                     'NAAG': 1.20,
                     'PCho': 0.85,
                     'PCh': 0.85,
                     'PCr': 4.87,
                     'PE': 1.80,
                     'sIns': 0.30,
                     'Scyllo': 0.30,
                     'Tau': 1.80}
    concs = []
    for name in basisNames:
        if name in standardconcs:
            concs.append(standardconcs[name])
        else:
            print(f'{name} not in standard concentrations. Setting to random between 1 and 5.')
            concs.append(np.random.random() * (5.0 - 1.0) + 1.0)

    return concs


def prep_mrs_for_synthetic(basisFile, points, bandwidth, ignore, ind_scaling, concentrations, metab_groups):
    """Prepare an mrs object for use in creating a synthetic spectrum,
       and return selected concentrations.

       metabolites in the basis file can be ignored or independently scaled.
    """

    if isinstance(basisFile, (str, Path)):
        basis = mrs_io.read_basis(basisFile)
    elif isinstance(basisFile, Basis):
        basis = basisFile
    else:
        basis = Basis(*basisFile)

    empty_mrs = MRS(FID=np.ones((points,)),
                    cf=basis.cf,
                    bw=bandwidth,
                    nucleus='1H',
                    basis=basis)

    empty_mrs.ignore = ignore
    empty_mrs.processForFitting(ind_scaling=ind_scaling)

    mg = parse_metab_groups(empty_mrs, metab_groups)

    if concentrations is None:
        concentrations = standardConcentrations(empty_mrs.names)
    elif isinstance(concentrations, (list, np.ndarray)):
        if len(concentrations) != len(empty_mrs.names):
            raise ValueError(f'Concentrations must have the same number of elements as basis spectra.'
                             f'{len(concentrations)} concentrations, {len(basis.names)} basis spectra.')
    elif isinstance(concentrations, dict):
        newconcs = []
        for name in empty_mrs.names:
            if name in concentrations:
                newconcs.append(concentrations[name])
            else:
                newconcs.extend(standardConcentrations([name]))
        concentrations = newconcs
    else:
        raise ValueError('Concentrations must be None, a list,'
                         'or a dict containing overides for particular metabolites.')
    return empty_mrs, concentrations, mg


def syntheticFromBasisFile(basisFile,
                           ignore=None,
                           metab_groups=None,
                           ind_scaling=None,
                           concentrations=None,
                           baseline=None,
                           baseline_ppm=None,
                           broadening=(9.0, 0.0),
                           shifting=0.0,
                           phi0=0.0,
                           phi1=0.0,
                           coilamps=[1.0],
                           coilphase=[0.0],
                           noisecovariance=[[0.1]],
                           bandwidth=4000.0,
                           points=2048):
    """ Create synthetic data from a set of FSL-MRS basis files.

    Args:
            basisFile (str): path to directory containg basis spectra json files
            ignore (list of str, optional): Ignore metabolites in basis set.
            metab_groups (list of str, optional): Group metabolites to apply different model parameters to groups.
            ind_scaling (list of str, optional): Independently scale basis spectra in the basis set.
            concentrations (list or dict or None, optional ): If None, standard concentrations will be used.
                                                        If list of same length as basis spectra, then these will be
                                                        used.
                                                        Pass dict to overide standard values for specific metabolites.
                                                        Key should be metabolite name.
            baseline (list of floats, optional): Baseline coeeficients, 2 (real, imag) needed per order.
                e.g. [1,1,0.1, 0.1] to specifiy a 1st order baseline.
            baseline_ppm (tuple, optional): Specify ppm range over which baseline is calculated.
            broadening (list of tuples or tuple:floats, optional): Tuple containg a gamma and sigma or a list of tuples
            for each basis.
            shifting (list of floats or float, optional): Eps shift value or a list for each basis.
            coilamps (list of floats, optional): If multiple coils, specify magnitude scaling.
            coilphase (list of floats, optional): If multiple coils, specify phase.
            noisecovariance (list of floats, optional): N coils x N coils array of noise variance/covariance.
            bandwidth (float,optional): Bandwidth of output spectrum in Hz
            points (int,optional): Number of points in output spectrum.

    Returns:
        FIDs: Numpy array of synthetic FIDs
        outHeader: Header suitable for loading FIDs into MRS object.
        concentrations: Final concentration scalings
    """

    empty_mrs, concentrations, mg = prep_mrs_for_synthetic(basisFile,
                                                           points,
                                                           bandwidth,
                                                           ignore,
                                                           ind_scaling,
                                                           concentrations,
                                                           metab_groups)

    # Currently hardcoded to voigt model. Sigma can always be set to 0.
    _, _, fwd_model, _, p2x = models.getModelFunctions('voigt')
    g = max(mg) + 1

    if not isinstance(broadening, list):
        broadening = [broadening, ]
    if not isinstance(shifting, list):
        shifting = [shifting, ]

    if g == 1:
        gamma = broadening[0][0]
        sigma = broadening[0][1]
        eps = shifting
    else:
        if len(broadening) == g:
            gamma = [br[0] for br in broadening]
            sigma = [br[1] for br in broadening]
        elif len(broadening) == 1:
            gamma = [broadening[0][0], ] * g
            sigma = [broadening[0][1], ] * g
        else:
            raise ValueError('broadening must be single value,'
                             'match the length of metab_groups.'
                             f'Currently {len(broadening)}.')

        if len(shifting) == g:
            eps = shifting
        elif len(shifting) == 1:
            eps = shifting * g
        else:
            raise ValueError('shifting must be single value,'
                             'match the length of metab_groups.'
                             f'Currently {len(shifting)}.')

    if baseline is None:
        baseline_order = 0
        b = [0, 0]
    else:
        baseline_order = int(len(baseline) / 2 - 1)
        b = baseline

    model_params = p2x(concentrations,
                       gamma,
                       sigma,
                       eps,
                       phi0,
                       phi1,
                       b)

    FIDs = synthetic_from_fwd_model(fwd_model,
                                    model_params,
                                    empty_mrs,
                                    baseline_order,
                                    mg,
                                    coilamps=coilamps,
                                    coilphase=coilphase,
                                    noisecovariance=noisecovariance,
                                    ppmlim=baseline_ppm)

    return FIDs, \
        empty_mrs, \
        concentrations


def synthetic_from_fwd_model(fwd_model,
                             model_params,
                             mrs,
                             baseline_order,
                             metab_groups,
                             coilamps=[1.0],
                             coilphase=[0.0],
                             noisecovariance=[[0.1]],
                             ppmlim=None):
    """ Create a synthetic spectrum from the forward fitting model"""
    freq, time, basis = mrs.frequencyAxis, mrs.timeAxis, mrs.basis
    g = max(metab_groups) + 1
    b = prepare_polynomial_regressor(
        mrs.numPoints,
        baseline_order,
        mrs.ppmlim_to_range(ppmlim))

    basespec = fwd_model(model_params,
                         freq,
                         time,
                         basis,
                         b,
                         metab_groups,
                         g)

    baseFID = SpecToFID(basespec)

    # Form noise vectors
    ncoils = len(coilamps)
    noisecovariance = np.asarray(noisecovariance)
    if len(coilphase) != ncoils:
        raise ValueError('Length of coilamps and coilphase must match.')
    if noisecovariance.shape != (ncoils, ncoils):
        raise ValueError('noisecovariance must be ncoils x ncoils.')

    noise = np.random.multivariate_normal(np.zeros((ncoils)), noisecovariance, mrs.numPoints) \
        + 1j * np.random.multivariate_normal(np.zeros((ncoils)), noisecovariance, mrs.numPoints)

    # Add noise and write to output list
    FIDs = []
    for cDx, (camp, cphs) in enumerate(zip(coilamps, coilphase)):
        FIDs.append((camp * np.exp(1j * cphs) * baseFID) + noise[:, cDx])
    FIDs = np.asarray(FIDs).T
    FIDs = np.squeeze(FIDs)

    return FIDs
