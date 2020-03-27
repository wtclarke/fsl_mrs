import numpy as np
import hlsvdpro
from fsl_mrs.utils.misc import checkCFUnits
from fsl_mrs.utils.constants import H2O_PPM_TO_TMS

def hlsvd(FID,dwelltime,centralFrequency,limits,limitUnits = 'ppm',numSingularValues=50):
    """ Run HLSVDPRO on FID
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwell time in seconds
        centralFrequency (float) : Central frequency in Hz
        limits (tuple): Limit deletion of singular values in this range. 
        limitUnits (str,optional): Axis that limits are given in. By Default in ppm, relative to receiver frequency (no shift). Can be 'Hz', 'ppm' or 'ppm+shift'.
        numSingularValues (int, optional): Max number of singular values

    Returns:
        FID (ndarray): Modified FID
    """
    nsv_found, singular_values, frequencies, damping_factors, amplitudes, phases = hlsvdpro.hlsvd(FID,numSingularValues,dwelltime)

    # convert to np array
    frequencies = np.asarray(frequencies)
    damping_factors = np.asarray(damping_factors)
    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)

    # Limit by frequencies
    if limitUnits.lower() == 'ppm':
        centralFrequency = checkCFUnits(centralFrequency,units='MHz')
        frequencylimit = np.array(limits)*centralFrequency
    elif limitUnits.lower() == 'ppm+shift':
        centralFrequency = checkCFUnits(centralFrequency,units='MHz')
        frequencylimit = (np.array(limits)-H2O_PPM_TO_TMS)*centralFrequency
    elif limitUnits.lower() == 'hz':
        frequencylimit = limits
    else:
        raise ValueError('limitUnits should be one of: ppm, ppm+shift or hz.')
    limitIndicies = (frequencies > frequencylimit[0]) & (frequencies < frequencylimit[1])

    sumFID = np.zeros(FID.shape,dtype=np.complex128)
    timeAxis = np.linspace(0,dwelltime*(FID.size-1),FID.size)

    for use,f,d,a,p in zip(limitIndicies,frequencies,damping_factors,amplitudes,phases):
        if use:
            sumFID += a * np.exp((timeAxis/d) + 1j*2*np.pi * (f*timeAxis+p/360.0))

    return FID - sumFID

def hlsvd_report(inFID,outFID,hdr,limits,limitUnits = 'ppm',plotlim = (0.2,6)):
    from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    from fsl_mrs.utils.plotting import styleSpectrumAxes

    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    plotIn = toMRSobj(inFID)
    plotOut = toMRSobj(outFID)

    if limitUnits.lower() == 'ppm':
        limits = np.array(limits)+H2O_PPM_TO_TMS
    elif limitUnits.lower() == 'ppm+shift':
        pass
    elif limitUnits.lower() == 'hz':
        limits = (np.array(limits)/(plotIn.centralFrequency/1E6))+H2O_PPM_TO_TMS
    else:
        raise ValueError('limitUnits should be one of: ppm, ppm+shift or hz.')    
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(plotIn.getAxes(ppmlim=plotlim),np.real(plotIn.getSpectrum(ppmlim=plotlim)),'k',label='Uncorrected', linewidth=2)
    plt.plot(plotIn.getAxes(ppmlim=limits),np.real(plotIn.getSpectrum(ppmlim=limits)),'g',label='Limits', linewidth=2)
    plt.plot(plotOut.getAxes(ppmlim=plotlim),np.real(plotOut.getSpectrum(ppmlim=plotlim)),'r',label='Corrected', linewidth=2)
    diff = plotOut.getSpectrum(ppmlim=plotlim)-plotIn.getSpectrum(ppmlim=plotlim)
    plt.plot(plotOut.getAxes(ppmlim=plotlim),np.real(diff),'b--',label='Difference', linewidth=2)
    styleSpectrumAxes(ax=plt.gca())
    plt.legend()
    plt.rcParams.update({'font.size': 12})
    plt.show()