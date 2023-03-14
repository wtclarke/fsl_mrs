#!/usr/bin/env python

# qc.py - Calculate various QC measures
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

from fsl_mrs import models
from fsl_mrs.utils.constants import NOISE_REGION
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID
from fsl_mrs.core import MRS
from scipy.signal import find_peaks
from scipy.stats import levene
import numpy as np
from collections import namedtuple
import pandas as pd
import numpy.polynomial.polynomial as poly

SNR = namedtuple('SNR', ['spectrum', 'peaks', 'residual'])


class NoiseNotFoundError(ValueError):
    pass


def calcQC(mrs, res, ppmlim=None):
    """ Calculate SNR and FWHM on fitted data

    """
    if ppmlim is None:
        ppmlim = res.ppmlim

    if res.method == 'MH':
        MCMCUsed = True
    else:
        MCMCUsed = False

    try:
        if MCMCUsed:
            # Loop over the individual MH results
            fwhm = []
            snrPeaks = []
            for _, rp in res.fitResults.iterrows():
                qcres = calcQCOnResults(mrs, res, rp, ppmlim)
                snrPeaks.append(qcres[0])
                fwhm.append(qcres[1])
            snrSpec = qcres[2]
            fwhm = np.asarray(fwhm).T
            snrPeaks = np.asarray(snrPeaks).T
        else:
            # Pass the single Newton results
            snrPeaks, fwhm, snrSpec = calcQCOnResults(mrs, res, res.params, ppmlim)
            fwhm = np.asarray(fwhm)
            snrPeaks = np.asarray(snrPeaks)
    except NoiseNotFoundError:
        outShape = (len(res.metabs), res.fitResults.shape[0])
        fwhm = np.full(outShape, np.nan)
        snrSpec = np.nan
        snrPeaks = np.full(outShape, np.nan)

    # Calculate the LCModel style SNR based on peak height over SD of residual
    first, last = mrs.ppmlim_to_range(ppmlim=res.ppmlim)
    baseline = FIDToSpec(res.predictedFID(mrs, mode='baseline'))[first:last]
    spectrumMinusBaseline = mrs.get_spec(ppmlim=res.ppmlim) - baseline
    snrResidual_height = np.max(np.abs(np.real(spectrumMinusBaseline)))
    rmse = 2.0 * np.sqrt(res.mse)
    snrResidual = snrResidual_height / rmse

    # Assemble outputs
    # SNR output
    snrdf = pd.DataFrame()
    for m, snr in zip(res.metabs, snrPeaks):
        snrdf[f'SNR_{m}'] = pd.Series(snr)
    snrdf.fillna(0.0, inplace=True)

    SNRobj = SNR(spectrum=snrSpec, peaks=snrdf, residual=snrResidual)

    fwhmdf = pd.DataFrame()
    for m, width in zip(res.metabs, fwhm):
        fwhmdf[f'fwhm_{m}'] = pd.Series(width)
    fwhmdf.fillna(0.0, inplace=True)

    return fwhmdf, SNRobj


def calcQCOnResults(mrs, res, resparams, ppmlim):
    """ Calculate QC metrics on single instance of fitting results

    """
    # Generate MRS objs for the results
    basisMRS = generateBasisFromRes(mrs, res, resparams)

    # ID noise region
    noisemask = idNoiseRegion(mrs, debug=False)

    fwhm = []
    for basemrs in basisMRS:
        # FWHM
        baseFWHM = res.getLineShapeParams()
        fwhm_curr, _, _ = idPeaksCalcFWHM(basemrs, estimatedFWHM=np.max(baseFWHM[0]), ppmlim=ppmlim)
        fwhm.append(fwhm_curr)

    # Identify min FWHM to use:
    singlet_fwhm = min(fwhm)

    # Calculate single spectrum SNR - based on max value of actual data in region
    # No apodisation applied.
    allSpecHeight = np.max(np.abs(np.real(mrs.get_spec(ppmlim=ppmlim))))
    unApodNoise = noiseSD(mrs.get_spec(), noisemask)
    specSNR = allSpecHeight / unApodNoise

    basisSNR = []
    for basemrs in basisMRS:
        # Basis SNR
        basisSNR.append(matchedFilterSNR(mrs, basemrs, singlet_fwhm, noisemask, ppmlim))

    return basisSNR, fwhm, specSNR


def noiseSD(spectrum, noisemask=None):
    """ Return noise SD. sqrt(2)*real(spectrum)"""
    if noisemask is None:
        return np.sqrt(2) * np.std(np.real(spectrum))
    else:
        return np.sqrt(2) * np.std(np.real(spectrum[noisemask]))


class NucleusQCNotImplemented(Exception):
    pass


def _detrend_noise(spec, axis):
    '''Polynomial fit to remove trend from noise region'''
    coefs = poly.polyfit(axis, spec, 4)
    fit = poly.polyval(axis, coefs)
    return spec - fit


def idNoiseRegion(mrs, debug=False):
    """ Identify noise region in given spectrum"""
    spectrum = mrs.get_spec()
    npoints = spectrum.size
    ppmaxis = mrs.getAxes()

    # Noise regions dependent on nucleus
    try:
        noise_regions = NOISE_REGION[mrs.nucleus]
    except KeyError:
        raise NucleusQCNotImplemented(f'QC for {mrs.nucleus} not yet implemented.')

    noise_mask = np.zeros(npoints, dtype=bool)
    detrended_noise_spec = np.zeros_like(spectrum.real)

    for region in noise_regions:
        # Convert ppm range to index in this spectrum
        if region[0] is None:
            region[0] = ppmaxis.min()
        if region[1] is None:
            region[1] = ppmaxis.max()
        first, last = mrs.ppmlim_to_range(ppmlim=region)
        last += 1

        # Assign to initial noise mask
        noise_mask[first:last] = 1

        # Detrend the data for the os detection
        detrended_noise_spec[first:last] = _detrend_noise(
            spectrum.real[first:last],
            ppmaxis[first:last])

    # Now check for oversampling that hasn't been removed
    # Identify possible OS region - assume max OS of 2
    poss_os_region = np.zeros(spectrum.shape, dtype=bool)
    poss_os_region[:int(spectrum.size / 4)] = 1
    poss_os_region[-int(spectrum.size / 4):] = 1

    just_os = noise_mask & poss_os_region
    just_normal = noise_mask ^ poss_os_region

    os_region = detrended_noise_spec[just_os]
    central_region = detrended_noise_spec[just_normal]

    if noise_mask.sum() < 100:
        # Fallback - take 50 points from each end
        noise_mask[:50] = 1
        noise_mask[-50:] = 1
        p = 1.0
    else:
        # Test for unequal variances between possible os region and the remainder noise region
        _, p = levene(os_region, central_region, center='mean')

    if debug:
        import matplotlib.pyplot as plt
        max_val = np.abs(detrended_noise_spec).max()
        plt.figure(figsize=(8, 8))
        plt.plot(max_val * spectrum.real / spectrum.real.max(), 'k')
        plt.plot(detrended_noise_spec)
        plt.plot(noise_mask * max_val)
        plt.plot(poss_os_region * max_val * 0.75)
        plt.plot(just_os * max_val * 0.5)
        plt.plot(just_normal * max_val * 0.25)
        plt.savefig('noise_id.png')

    if p < 0.05:
        if debug:
            print('Oversampling detected')
        return just_normal
    else:
        return noise_mask


def idPeaksCalcFWHM(mrs, estimatedFWHM=15.0, ppmlim=None):
    """ Identify peaks and calculate FWHM of fitted basis spectra

    """
    spectrum = np.abs(np.real(mrs.get_spec(ppmlim=ppmlim)))
    with np.errstate(divide='ignore', invalid='ignore'):
        spectrum /= np.max(spectrum)

    peaks, props = find_peaks(spectrum, prominence=(0.4, None), width=(None, estimatedFWHM * 2))

    if peaks.size == 0:
        return 0, None, None

    sortind = np.argsort(props['prominences'])

    pkIndex = peaks[sortind[-1]]
    pkPosition = mrs.getAxes(ppmlim=ppmlim)[pkIndex]
    fwhm = props['widths'][sortind[-1]] * (mrs.bandwidth / mrs.numPoints)

    return fwhm, pkIndex, pkPosition


def generateBasisFromRes(mrs, res, resparams):
    """ Return mrs objects for each fitted basis spectrum"""
    mrsFits = []
    for metab in mrs.names:
        pred = models.getFittedModel(res.model,
                                     resparams,
                                     res.base_poly, res.
                                     metab_groups,
                                     mrs,
                                     basisSelect=metab,
                                     noBaseline=True)
        pred = SpecToFID(pred)  # predict FID not Spec
        mrsOut = MRS(pred,
                     cf=mrs.centralFrequency,
                     bw=mrs.bandwidth,
                     nucleus=mrs.nucleus)
        mrsFits.append(mrsOut)
    return mrsFits


def specApodise(mrs, amount):
    """ Apply apodisation to spectrum"""
    FIDApod = mrs.FID * np.exp(-amount * mrs.getAxes('time'))
    return FIDToSpec(FIDApod)


def matchedFilterSNR(mrs, basismrs, lw, noisemask, ppmlim):
    apodbasis = specApodise(basismrs, lw)
    apodSpec = specApodise(mrs, lw)

    apodNoise = _detrend_noise(
        apodSpec[noisemask],
        np.arange(noisemask.sum()))

    currNoise = noiseSD(apodNoise)
    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim)
    peakHeight = np.max(np.abs(np.real(apodbasis[first:last])))

    return peakHeight / currNoise
