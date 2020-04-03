from fsl_mrs.utils import models
from fsl_mrs.utils.misc import FIDToSpec,SpecToFID
from fsl_mrs.core import MRS
from scipy.signal import find_peaks
import numpy as np
from numpy.lib.stride_tricks import as_strided

def calcQC(mrs,res,ppmlim=(0.2,4.2)):
    """ Calculate SNR and FWHM on fitted data

    """
    if res.method == 'MH':
        MCMCUsed = True
    else:
        MCMCUsed = False

    if MCMCUsed:
        # Loop over the individual MH results
        fwhm = []
        snrPeaks = []
        for rp in res.mcmc_samples:
            qcres = calcQCOnResults(mrs,res,rp,ppmlim)
            snrPeaks.append(qcres[0])
            fwhm.append(qcres[1])
        snrSpec = qcres[2]
        fwhm = np.asarray(fwhm)
        snrPeaks = np.asarray(snrPeaks)
    else:
        # Pass the single Newton results
        snrPeaks,fwhm,snrSpec = calcQCOnResults(mrs,res,res.params,ppmlim)
        fwhm = np.asarray(fwhm)
        snrPeaks = np.asarray(snrPeaks)

    return fwhm,snrSpec,snrPeaks

def calcQCOnResults(mrs,res,resparams,ppmlim):
    """ Calculate QC metrics on single instance of fitting results

    """    
    # Generate MRS objs for the results
    basisMRS = generateBasisFromRes(mrs,res,resparams)
    
    # Generate masks for noise regions
    combinedSpectrum = np.zeros(mrs.FID.size)
    for basemrs in basisMRS:
        combinedSpectrum += np.real(basemrs.getSpectrum())
    normCombSpec = combinedSpectrum/np.max(combinedSpectrum)
    noiseRegion = np.abs(normCombSpec)<0.015
    # Noise region OS masks
    noiseOSMask = detectOS(mrs,noiseRegion)
    combinedMask = noiseRegion&noiseOSMask

    # Calculate single spectrum SNR - based on max value of actual data in region
    # No apodisation applied.
    allSpecHeight = np.max(np.real(mrs.getSpectrum(ppmlim=ppmlim)))
    unApodNoise = np.sqrt(2)*np.std(np.real(mrs.getSpectrum()[combinedMask]))
    specSNR = allSpecHeight/unApodNoise

    fwhm = []
    basisSNR = []
    for basemrs in basisMRS:
        #FWHM
        fwhm_curr,_,_ = idPeaksCalcFWHM(basemrs,estimatedFWHM=res.eps[0],ppmlim=ppmlim)        
        fwhm.append(fwhm_curr)

        #Basis SNR
        basisSNR.append(matchedFilterSNR(mrs,basemrs,fwhm_curr,combinedMask,ppmlim))
    
    return basisSNR,fwhm,specSNR


def idPeaksCalcFWHM(mrs,estimatedFWHM=15.0,ppmlim=(0.2,4.2)):
    """ Identify peaks and calculate FWHM of fitted basis spectra

    """
    fwhmInPnts = estimatedFWHM/(mrs.bandwidth/mrs.numPoints)
    spectrum = np.real(mrs.getSpectrum(ppmlim=ppmlim))
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
    currNoise = np.sqrt(2)*np.std(np.real(apodSpec)[noisemask])
    f,l = mrs.ppmlim_to_range(ppmlim=ppmlim)
    peakHeight = np.max(np.real(apodbasis[f:l]))
    currSNR= peakHeight/currNoise
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
    regionPoints = 100
    
    def running_std_strides(seq, window=100):
        stride = seq.strides[0]
        sequence_strides = as_strided(seq, shape=[len(seq) - window + 1, window], strides=[stride, stride])
        return sequence_strides.std(axis=1)
    
    out = running_std_strides(noiseOnlySpec,window=regionPoints)
    padded = np.pad(out,(int(regionPoints/2),int(regionPoints/2)-1),constant_values=np.nan)

    return padded



