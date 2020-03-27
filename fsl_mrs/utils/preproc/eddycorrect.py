import numpy as np
def eddy_correct(FIDmet,FIDPhsRef):
    """
    Subtract water phase from metab phase
    Typically do this after coil combination

    Parameters:
    -----------
    FIDmet : array-like (FID for the metabolites)
    FIDPhsRef   : array-like (Phase reference FID)

    Returns:
    --------
    array-like (corrected metabolite FID)
    """
    phsRef = np.angle(FIDPhsRef)
    return np.abs(FIDmet) * np.exp(1j*(np.angle(FIDmet)-phsRef))

def eddy_correct_report(inFID,outFID,hdr,ppmlim = (0.2,4.2)):
    from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    from fsl_mrs.utils.plotting import styleSpectrumAxes

    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    plotIn = toMRSobj(inFID)
    plotOut = toMRSobj(outFID)    
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(plotIn.getAxes(ppmlim=ppmlim),np.real(plotIn.getSpectrum(ppmlim=ppmlim)),'k',label='Uncorrected', linewidth=2)
    plt.plot(plotOut.getAxes(ppmlim=ppmlim),np.real(plotOut.getSpectrum(ppmlim=ppmlim)),'r--',label='Corrected', linewidth=2)
    styleSpectrumAxes(ax=plt.gca())
    plt.legend()
    plt.rcParams.update({'font.size': 12})
    plt.show()