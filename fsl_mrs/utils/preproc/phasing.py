import numpy as np
from fsl_mrs.core import MRS
from fsl_mrs.utils.misc import extract_spectrum
from fsl_mrs.utils.preproc.shifting import pad
def applyPhase(FID,phaseAngle):
    return FID * np.exp(1j*phaseAngle)

def phaseCorrect(FID,bw,cf,ppmlim=(2.8,3.2),shift=True):
    """ Phase correction based on the phase of a maximum point.

    Args:
        FID (ndarray): Time domain data
        bw (float): bandwidth
        cf (float): central frequency in Hz
        ppmlim (tuple,optional)  : Limit to this ppm range
        shift (bool,optional)    : Apply H20 shft

    Returns:
        FID (ndarray): Phase corrected FID
    """
    #Find maximum of absolute spectrum in ppm limit
    padFID = pad(FID,FID.size*3)
    MRSargs = {'FID':padFID,'bw':bw,'cf':cf}
    mrs = MRS(**MRSargs)
    spec = extract_spectrum(mrs,padFID,ppmlim=ppmlim,shift=shift)

    maxIndex = np.argmax(np.abs(spec))
    phaseAngle = -np.angle(spec[maxIndex])
    
    return applyPhase(FID,phaseAngle),phaseAngle,int(np.round(maxIndex/4))

def phaseCorrect_report(inFID,outFID,hdr,position,ppmlim=(2.8,3.2)):
    from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    from fsl_mrs.utils.plotting import styleSpectrumAxes

    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    plotIn = toMRSobj(inFID)
    plotOut = toMRSobj(outFID)

    widelimit = (0,6)

    fig = plt.figure(figsize=(10,10))
    plt.plot(plotIn.getAxes(ppmlim=widelimit),np.real(plotIn.getSpectrum(ppmlim=widelimit)),'k',label='Unphased', linewidth=2)
    plt.plot(plotIn.getAxes(ppmlim=ppmlim),np.real(plotIn.getSpectrum(ppmlim=ppmlim)),'r',label='search region', linewidth=2)
    plt.plot(plotIn.getAxes(ppmlim=ppmlim)[position],np.real(plotIn.getSpectrum(ppmlim=ppmlim))[position],'rx',label='max point', linewidth=2)
    plt.plot(plotOut.getAxes(ppmlim=widelimit),np.real(plotOut.getSpectrum(ppmlim=widelimit)),'b--',label='Phased', linewidth=2)
    styleSpectrumAxes(ax=plt.gca())
    plt.legend()
    plt.rcParams.update({'font.size': 12})
    plt.show()
