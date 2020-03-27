import numpy as np
from fsl_mrs.core import MRS
from fsl_mrs.utils.misc import extract_spectrum

def timeshift(FID,dwelltime,shiftstart,shiftend,samples=None):
    """ Shift data on time axis
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwell time in seconds
        shiftstart (float): Shift start point in seconds 
        shiftend (float): Shift end point in seconds
        samples (int, optional): Resample to this number of points

    Returns:
        FID (ndarray): Shifted FID
    """
    originalAcqTime = dwelltime*(FID.size-1)    
    originalTAxis = np.linspace(0,originalAcqTime,FID.size)
    if samples is None:        
        newDT = dwelltime
    else:
        totalacqTime = originalAcqTime-shiftstart+shiftend
        newDT = totalacqTime/samples
    newTAxis = np.arange(originalTAxis[0]+shiftstart,originalTAxis[-1]+shiftend,newDT)
    FID = np.interp(newTAxis,originalTAxis,FID,left=0.0+1j*0.0, right=0.0+1j*0.0)

    return FID,newDT

def freqshift(FID,dwelltime,shift):
    """ Shift data on frequency axis
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwelltime in seconds
        shift (float): shift in Hz

    Returns:
        FID (ndarray): Shifted FID
    """
    tAxis = np.linspace(0,dwelltime*FID.size,FID.size)
    phaseRamp = 2*np.pi*tAxis*shift
    FID = FID * np.exp(1j*phaseRamp)
    return FID

def shiftToRef(FID,target,bw,cf,ppmlim=(2.8,3.2),shift=True):
    #Find maximum of absolute spectrum in ppm limit
    padFID = pad(FID,FID.size*3)
    MRSargs = {'FID':padFID,'bw':bw,'cf':cf}
    mrs = MRS(**MRSargs)
    spec = extract_spectrum(mrs,padFID,ppmlim=ppmlim,shift=shift)
    if shift:
        extractedAxis = mrs.getAxes(ppmlim=ppmlim)
    else: 
        extractedAxis = mrs.getAxes(ppmlim=ppmlim,axis='ppm')

    maxIndex = np.argmax(np.abs(spec))
    shiftAmount = extractedAxis[maxIndex]-target
    shiftAmountHz = shiftAmount * mrs.centralFrequency/1E6    

    return freqshift(FID,1/bw,-shiftAmountHz),shiftAmount

def truncate(FID,k,first_or_last='last'):
    """
    Truncate parts of a FID
    
    Parameters:
    -----------
    FID           : array-like
    k             : int (number of timepoints to remove)
    first_or_last : either 'first' or 'last' (which bit to truncate)

    Returns:
    --------
    array-like
    """
    FID_trunc = FID.copy()
    
    if first_or_last == 'first':
        return FID_trunc[k:]
    elif first_or_last == 'last':
        return FID_trunc[:-k]
    else:
        raise(Exception("Last parameter must either be 'first' or 'last'"))

def pad(FID,k,first_or_last='last'):
    """
    Pad parts of a FID
    
    Parameters:
    -----------
    FID           : array-like
    k             : int (number of timepoints to add)
    first_or_last : either 'first' or 'last' (which bit to pad)

    Returns:
    --------
    array-like
    """
    FID_pad = FID.copy()
    
    if first_or_last == 'first':
        return np.pad(FID_pad,(k,0))
    elif first_or_last == 'last':
        return np.pad(FID_pad,(0,k))
    else:
        raise(Exception("Last parameter must either be 'first' or 'last'"))

def shift_report(inFID,outFID,hdr,ppmlim = (0.2,4.2)):
    from matplotlib import pyplot as plt
    from fsl_mrs.utils.plotting import styleSpectrumAxes

    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    plotIn = toMRSobj(inFID)
    plotOut = toMRSobj(outFID)    
    
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,7))

    ax1.plot(plotIn.getAxes(ppmlim=ppmlim),np.real(plotIn.getSpectrum(ppmlim=ppmlim)),'k',label='Original', linewidth=2)
    ax1.plot(plotOut.getAxes(ppmlim=ppmlim),np.real(plotOut.getSpectrum(ppmlim=ppmlim)),'r',label='Shifted', linewidth=2)
    styleSpectrumAxes(ax=ax1)
    ax1.legend()

    ax2.plot(plotIn.getAxes(axis='time'),np.real(plotIn.FID),'k',label='Original', linewidth=2)
    ax2.plot(plotOut.getAxes(axis='time'),np.real(plotOut.FID),'r--',label='Shifted', linewidth=2)
    # styleSpectrumAxes(ax=ax2)
    ax2.legend()
    ax2.set_yticks([0.0])
    ax2.set_ylabel('Re(signal) (a.u.)')
    ax2.set_xlabel('Time (s)')

    ax2.autoscale(enable=True, axis='x', tight=False)

    plt.rcParams.update({'font.size': 12})
    plt.show()
