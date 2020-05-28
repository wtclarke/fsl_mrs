from fsl_mrs.utils import models
from fsl_mrs.utils.misc import FIDToSpec,SpecToFID
from fsl_mrs.core import MRS
from scipy.signal import find_peaks
import numpy as np
from numpy.lib.stride_tricks import as_strided
from collections import namedtuple
import pandas as pd

SNR = namedtuple('SNR',['spectrum','peaks','residual'])

class NoiseNotFoundError(ValueError):
    pass

def calcQC(mrs,res,ppmlim=(0.2,4.2)):
    """ Calculate SNR and FWHM on fitted data

    """
    if res.method == 'MH':
        MCMCUsed = True
    else:
        MCMCUsed = False

    try:
        if MCMCUsed:
            # Loop over the individual MH results
            fwhm = []
            snrPeaks = []
            for _,rp in res.fitResults.iterrows():
                qcres = calcQCOnResults(mrs,res,rp,ppmlim)
                snrPeaks.append(qcres[0])
                fwhm.append(qcres[1])
            snrSpec = qcres[2]
            fwhm = np.asarray(fwhm).T
            snrPeaks = np.asarray(snrPeaks).T
        else:
            # Pass the single Newton results
            snrPeaks,fwhm,snrSpec = calcQCOnResults(mrs,res,res.params,ppmlim)
            fwhm = np.asarray(fwhm)
            snrPeaks = np.asarray(snrPeaks)
    except NoiseNotFoundError:
        outShape = (len(res.metabs),res.fitResults.shape[0])
        fwhm = np.full(outShape,np.nan)
        snrSpec = np.nan
        snrPeaks = np.full(outShape,np.nan)

    # Calculate the LCModel style SNR based on peak height over SD of residual
    first,last = mrs.ppmlim_to_range(ppmlim=res.ppmlim)
    baseline = FIDToSpec(res.predictedFID(mrs,mode='baseline'))[first:last]
    spectrumMinusBaseline = mrs.getSpectrum(ppmlim=res.ppmlim)-baseline
    snrResidual_height = np.max(np.real(spectrumMinusBaseline))
    rmse = 2.0*np.sqrt(res.mse)
    snrResidual = snrResidual_height/rmse

    # Assemble outputs
    # SNR output    
    snrdf = pd.DataFrame()
    for m,snr in zip(res.metabs,snrPeaks):
            snrdf[f'SNR_{m}'] = pd.Series(snr)            
    SNRobj = SNR(spectrum=snrSpec,peaks=snrdf,residual=snrResidual)

    fwhmdf = pd.DataFrame()
    for m,width in zip(res.metabs,fwhm):
            fwhmdf[f'fwhm_{m}'] = pd.Series(width)
        

    return fwhmdf,SNRobj


def calcQCOnResults(mrs,res,resparams,ppmlim):
    """ Calculate QC metrics on single instance of fitting results

    """    
    # Generate MRS objs for the results
    basisMRS = generateBasisFromRes(mrs,res,resparams)
    
    # Generate masks for noise regions
    combinedSpectrum = np.zeros(mrs.FID.size)
    for basemrs in basisMRS:
        combinedSpectrum += np.real(basemrs.getSpectrum())

    noisemask = idNoiseRegion(mrs,combinedSpectrum)

    # Calculate single spectrum SNR - based on max value of actual data in region
    # No apodisation applied.
    allSpecHeight = np.max(np.real(mrs.getSpectrum(ppmlim=ppmlim)))
    unApodNoise = noiseSD(mrs.getSpectrum(),noisemask)
    specSNR = allSpecHeight/unApodNoise

    fwhm = []
    basisSNR = []
    for basemrs in basisMRS:
        #FWHM
        baseFWHM = res.getLineShapeParams()
        fwhm_curr,_,_ = idPeaksCalcFWHM(basemrs,estimatedFWHM=np.max(baseFWHM[0]),ppmlim=ppmlim)        
        fwhm.append(fwhm_curr)

        #Basis SNR
        basisSNR.append(matchedFilterSNR(mrs,basemrs,fwhm_curr,noisemask,ppmlim))
    
    return basisSNR,fwhm,specSNR

def noiseSD(spectrum,noisemask):
    """ Return noise SD. sqrt(2)*real(spectrum)"""
    return np.sqrt(2)*np.std(np.real(spectrum[noisemask]))

def idNoiseRegion(mrs,spectrum,startingNoiseThreshold = 0.001):
    """ Identify noise region in given spectrum"""
    normspec = np.real(spectrum)/np.max(np.real(spectrum))
    noiseThreshold = startingNoiseThreshold
    noiseRegion = np.abs(normspec)<noiseThreshold
    # print(np.sum(noiseRegion))
    while np.sum(noiseRegion)<100:
        if noiseThreshold>0.1:
            raise NoiseNotFoundError(f'Unable to identify suitable noise area. Only {np.sum(noiseRegion)} points of {normspec.size} found. Minimum of 100 needed.')
        noiseThreshold += 0.001
        noiseRegion = np.abs(normspec)<noiseThreshold
        # print(np.sum(noiseRegion))
    
    # Noise region OS masks
    noiseOSMask = detectOS(mrs,noiseRegion)
    combinedMask = noiseRegion&noiseOSMask

    return combinedMask


def idPeaksCalcFWHM(mrs,estimatedFWHM=15.0,ppmlim=(0.2,4.2)):
    """ Identify peaks and calculate FWHM of fitted basis spectra

    """
    fwhmInPnts = estimatedFWHM/(mrs.bandwidth/mrs.numPoints)
    spectrum = np.real(mrs.getSpectrum(ppmlim=ppmlim))
    with np.errstate(divide='ignore', invalid='ignore'): 
        spectrum /= np.max(spectrum)
    
    peaks,props = find_peaks(spectrum,prominence=(0.4,None),width=(None,estimatedFWHM*2))
    
    if peaks.size ==0:
            return 0,None,None

    sortind = np.argsort(props['prominences'])
    
    pkIndex = peaks[sortind[-1]]
    pkPosition = mrs.getAxes(ppmlim=ppmlim)[pkIndex]
    fwhm = props['widths'][sortind[-1]]*(mrs.bandwidth/mrs.numPoints)
    
    return fwhm,pkIndex,pkPosition

def generateBasisFromRes(mrs,res,resparams):
    """ Return mrs objects for each fitted basis spectrum"""
    mrsFits = []
    for metab in mrs.names:
        pred = models.getFittedModel(res.model,
                              resparams,
                              res.base_poly,res.
                              metab_groups,
                              mrs,
                              basisSelect= metab,
                              noBaseline=True)
        pred = SpecToFID(pred) # predict FID not Spec
        mrsOut = MRS(pred,cf=mrs.centralFrequency,bw=mrs.bandwidth)
        mrsFits.append(mrsOut)
    return mrsFits

def specApodise(mrs,amount):
    """ Apply apodisation to spectrum"""
    FIDApod = mrs.FID * np.exp(-amount*mrs.getAxes('time'))
    return FIDToSpec(FIDApod)

def matchedFilterSNR(mrs,basismrs,lw,noisemask,ppmlim):
    apodbasis = specApodise(basismrs,lw)
    apodSpec = specApodise(mrs,lw)
    currNoise = noiseSD(apodSpec,noisemask)
    f,l = mrs.ppmlim_to_range(ppmlim=ppmlim)
    peakHeight = np.max(np.real(apodbasis[f:l]))
    currSNR= peakHeight/currNoise
    # import matplotlib.pyplot as plt
    # plt.plot(np.real(apodSpec))
    # plt.plot(noisemask*np.max(np.real(apodSpec)))
    # plt.show()
    # print(f'SNR: {currSNR:0.1f} ({peakHeight:0.2e}/{currNoise:0.2e}), LW: {lw:0.1f}')
    return peakHeight/currNoise

def detectOS(mrs,noiseregionmask):
    """ If OS is detected restrict noise region to inner half"""

    sdSpec = getNoiseSDDist(np.real(mrs.getSpectrum()),noiseregionmask)
    sdSpec /= np.nanmean(sdSpec)
    sdSpec = sdSpec[~np.isnan(sdSpec)]
    npoints = sdSpec.size

    outerSD = np.concatenate((sdSpec[:int(npoints/10)],sdSpec[-int(npoints/10):]))
    innerSD = np.concatenate((sdSpec[int(npoints/10):2*int(npoints/10)],sdSpec[-2*int(npoints/10):-int(npoints/10)]))


    inner_mean = np.mean(innerSD)
    inner_SD = np.std(innerSD)
    outer_mean = np.mean(outerSD)

    sdDiff = np.abs(outer_mean-inner_mean)/inner_SD

    if sdDiff>2:
        # print('Oversampling detected')
        noiseRegionMask = np.full(mrs.FID.shape, False)
        npoints = mrs.FID.size
        noiseRegionMask[int(npoints*0.25):int(npoints*0.75)] = True 
    else:
        noiseRegionMask = np.full(mrs.FID.shape, True)
    return noiseRegionMask

def getNoiseSDDist(specIn,noiseregionmask):
    """ Calculate rolling SD of noise region"""
    # Normalise
    specIn /= np.max(specIn)
    
    # select noise regions
    noiseOnlySpec = specIn[noiseregionmask]

    npoints = noiseOnlySpec.size
    regionPoints = min(100,int(npoints/10))
    
    def running_std_strides(seq, window=100):
        stride = seq.strides[0]
        sequence_strides = as_strided(seq, shape=[len(seq) - window + 1, window], strides=[stride, stride])
        return sequence_strides.std(axis=1)
    
    out = running_std_strides(noiseOnlySpec,window=regionPoints)
    padded = np.pad(out,(int(regionPoints/2),int(regionPoints/2)-1),constant_values=np.nan)

    return padded



