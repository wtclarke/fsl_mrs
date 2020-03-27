""" Module containing alignment functions (align and aligndiff)"""
from fsl_mrs.utils.preproc.general import get_target_FID,add,subtract
from fsl_mrs.core import MRS
from fsl_mrs.utils.misc import extract_spectrum,shift_FID
from scipy.optimize import minimize
import numpy as np

# Phase-Freq alignment functions
def align_FID(mrs,src_FID,tgt_FID,ppmlim=None,shift=True):
    """
       Phase and frequency alignment

    Parameters
    ----------
    mrs : MRS Object
    src_FID : array-like
    tgt_FID : array-like
    ppmlim : tuple

    Returns
    -------
    array-like
    """

    # Internal functions so they can see globals
    def shift_phase_freq(FID,phi,eps,extract=True):        
        sFID = np.exp(-1j*phi)*shift_FID(mrs,FID,eps)
        if extract:
            sFID = extract_spectrum(mrs,sFID,ppmlim=ppmlim,shift=shift)
        return sFID
    def cf(p):
        phi    = p[0] #phase shift
        eps    = p[1] #freq shift
        FID    = shift_phase_freq(src_FID,phi,eps)
        target = extract_spectrum(mrs,tgt_FID,ppmlim=ppmlim,shift=shift)
        xx     = np.linalg.norm(FID-target)
        return xx
    x0  = np.array([0,0])
    res = minimize(cf, x0, method='Powell')
    phi = res.x[0]
    eps = res.x[1]
    
    return shift_phase_freq(src_FID,phi,eps,extract=False),phi,eps

def align_FID_diff(mrs,src_FID0,src_FID1,tgt_FID,diffType = 'add',ppmlim=None,shift=True):
    """
       Phase and frequency alignment

    Parameters
    ----------
    mrs : MRS Object
    src_FID0 : array-like - modified
    src_FID1 : array-like - not modifed
    tgt_FID : array-like
    ppmlim : tuple

    Returns
    -------
    array-like
    """
    # Internal functions so they can see globals
    def shift_phase_freq(FID0,FID1,phi,eps,extract=True):               
        sFID = np.exp(-1j*phi)*shift_FID(mrs,FID0,eps)
        if extract:
            sFID = extract_spectrum(mrs,sFID,ppmlim=ppmlim,shift=shift)
            FID1 = extract_spectrum(mrs,FID1,ppmlim=ppmlim,shift=shift)

        if diffType.lower() == 'add':
            FIDOut = add(FID1,sFID)
        elif diffType.lower() == 'sub':
            FIDOut = subtract(FID1,sFID)
        else:
            raise ValueError('diffType must be add or sub.')
        
        return FIDOut
    def cf(p):
        phi    = p[0] #phase shift
        eps    = p[1] #freq shift
        FID    = shift_phase_freq(src_FID0,src_FID1,phi,eps)
        target = extract_spectrum(mrs,tgt_FID,ppmlim=ppmlim,shift=shift)
        xx     = np.linalg.norm(FID-target)        
        return xx
    x0  = np.array([0,0])
    res = minimize(cf, x0)
    phi = res.x[0]
    eps = res.x[1]

    alignedFID0 = np.exp(-1j*phi)*shift_FID(mrs,src_FID0,eps)

    return alignedFID0,phi,eps

# The functions to call
# 1) For normal FIDs
def phase_freq_align(FIDlist,bandwidth,centralFrequency,ppmlim=None,niter=2,verbose=False,shift=True,target=None):
    """
    Algorithm:
       Average spectra
       Loop over all spectra and find best phase/frequency shifts
       Iterate

    TODO: 
       test the code
       add dedrifting?

    Parameters:
    -----------
    FIDlist          : list
    bandwidth        : float (unit=Hz)
    centralFrequency : float (unit=Hz)
    ppmlim           : tuple
    niter            : int
    verbose          : bool
    shift            : apply H20 shift to ppm limit
    ref              : reference data to align to
    
    Returns:
    --------
    list of FID aligned to each other
    """
    all_FIDs = FIDlist.copy()

    phiOut,epsOut = np.zeros(len(FIDlist)),np.zeros(len(FIDlist))
    for iter in range(niter):
        if verbose:
            print(' ---- iteration {} ----\n'.format(iter))

        if target is None:
            target = get_target_FID(FIDlist,target='nearest_to_mean')

        MRSargs = {'FID':target,'bw':bandwidth,'cf':centralFrequency}
        mrs = MRS(**MRSargs)
        
        for idx,FID in enumerate(all_FIDs):
            if verbose:
                print('... aligning FID number {}'.format(idx),end='\r')
            all_FIDs[idx],phi,eps = align_FID(mrs,FID,target,ppmlim=ppmlim,shift=shift)
            phiOut[idx] += phi
            epsOut[idx] += eps
        if verbose:
            print('\n')
    return all_FIDs,phiOut,epsOut

# 2) To align spectra from different groups with optional processing applied.
def phase_freq_align_diff(FIDlist0,FIDlist1,bandwidth,centralFrequency,diffType = 'add',ppmlim=None,shift=True,target=None):
    """ Align subspectra from difference methods.

    Only spectra in FIDlist0 are shifted.   

    Parameters:
    -----------
    FIDlist0         : list - shifted
    FIDlist1         : list - fixed
    bandwidth        : float (unit=Hz)
    centralFrequency : float (unit=Hz)
    diffType         : string - add or subtract
    ppmlim           : tuple    
    shift            : apply H20 shift to ppm limit
    ref              : reference data to align to
    
    Returns:
    --------
    two lists of FID aligned to each other, phase and shift applied to first list.
    """
    # Process target
    if target is not None:
        tgt_FID = target
    else:
        diffFIDList = []
        for fid0,fid1 in zip(FIDlist0,FIDlist1):
            if diffType.lower() == 'add':
                diffFIDList.append(add(fid1,fid0))
            elif diffType.lower() == 'sub':
                diffFIDList.append(subtract(fid1,fid0))
            else:
                raise ValueError('diffType must be add or sub.') 
        tgt_FID = get_target_FID(diffFIDList,target='nearest_to_mean') 


    # Pass to phase_freq_align
    mrs = MRS(FID=FIDlist0[0],cf=centralFrequency,bw=bandwidth)
    phiOut,epsOut = [],[]
    alignedFIDs0 = []
    for fid0,fid1 in zip(FIDlist0,FIDlist1):
        # breakpoint()
        out = align_FID_diff(mrs,fid0,fid1,tgt_FID,diffType = diffType,ppmlim=ppmlim,shift=shift)
        alignedFIDs0.append(out[0])
        phiOut.append(out[1])
        epsOut.append(out[2])     

    return alignedFIDs0,FIDlist1,phiOut,epsOut

# Reporting functions
def phase_freq_align_report(inFIDs,outFIDs,hdr,phi,eps,ppmlim=None):
    from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    from fsl_mrs.utils.preproc.combine import combine_FIDs
    from fsl_mrs.utils.plotting import styleSpectrumAxes
    fig, axs = plt.subplots(2, 2,figsize=(10,10))
    axs[0,0].plot(np.array(phi)*(180.0/np.pi))
    axs[0,0].set_ylabel(r'$\phi$ (degrees)')
    axs[0,1].plot(eps)
    axs[0,1].set_ylabel('Shift (Hz)')

    meanIn = combine_FIDs(inFIDs,'mean')
    meanOut = combine_FIDs(outFIDs,'mean')
    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    meanIn = toMRSobj(meanIn)
    meanOut = toMRSobj(meanOut)

    toPlotIn,toPlotOut = [],[]
    for fid in inFIDs:
        toPlotIn.append(toMRSobj(fid))
    for fid in outFIDs:
        toPlotOut.append(toMRSobj(fid))
    for fid in toPlotIn:
            axs[1,0].plot(fid.getAxes(ppmlim=ppmlim),np.real(fid.getSpectrum(ppmlim=ppmlim)))
    axs[1,0].plot(meanIn.getAxes(ppmlim=ppmlim),np.real(meanIn.getSpectrum(ppmlim=ppmlim)),'k')        
    styleSpectrumAxes(ax=axs[1,0])
    for fid in toPlotOut:
            axs[1,1].plot(fid.getAxes(ppmlim=ppmlim),np.real(fid.getSpectrum(ppmlim=ppmlim)))
    axs[1,1].plot(meanOut.getAxes(ppmlim=ppmlim),np.real(meanOut.getSpectrum(ppmlim=ppmlim)),'k')        
    styleSpectrumAxes(ax=axs[1,1])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10,10))
    plt.plot(meanIn.getAxes(ppmlim=ppmlim),np.real(meanIn.getSpectrum(ppmlim=ppmlim)),'k',label='Unaligned', linewidth=2)
    plt.plot(meanOut.getAxes(ppmlim=ppmlim),np.real(meanOut.getSpectrum(ppmlim=ppmlim)),'r--',label='Aligned', linewidth=2)
    styleSpectrumAxes(ax=plt.gca())
    plt.legend()
    plt.rcParams.update({'font.size': 12})
    plt.show()

def phase_freq_align_diff_report(inFIDs0,inFIDs1,outFIDs0,outFIDs1,hdr,eps,phi,ppmlim=None,diffType='add',shift=True):
    from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    from fsl_mrs.utils.preproc.combine import combine_FIDs
    from fsl_mrs.utils.plotting import styleSpectrumAxes
    fig, axs = plt.subplots(2, 2,figsize=(10,10))
    axs[0,0].plot(np.array(phi)*(180.0/np.pi))
    axs[0,0].set_ylabel(r'$\phi$ (degrees)')
    axs[0,1].plot(eps)
    axs[0,1].set_ylabel('Shift (Hz)')

    diffFIDListIn = []
    diffFIDListOut = []
    for fid0i,fid1i,fid0o,fid1o in zip(inFIDs0,inFIDs1,outFIDs0,outFIDs1):
        if diffType.lower() == 'add':
            diffFIDListIn.append(add(fid1i,fid0i))
            diffFIDListOut.append(add(fid1o,fid0o))
        elif diffType.lower() == 'sub':
            diffFIDListIn.append(subtract(fid1i,fid0i))
            diffFIDListOut.append(subtract(fid1o,fid0o))
        else:
            raise ValueError('diffType must be add or sub.') 

    meanIn = combine_FIDs(diffFIDListIn,'mean')
    meanOut = combine_FIDs(diffFIDListOut,'mean')
    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    meanIn = toMRSobj(meanIn)
    meanOut = toMRSobj(meanOut)

    if shift:
        axis = 'ppmshift'
    else:
        axis = 'ppm'

    toPlotIn,toPlotOut = [],[]
    for fid in diffFIDListIn:
        toPlotIn.append(toMRSobj(fid))
    for fid in diffFIDListOut:
        toPlotOut.append(toMRSobj(fid))
    for fid in toPlotIn:
            axs[1,0].plot(fid.getAxes(ppmlim=ppmlim, axis=axis),np.real(fid.getSpectrum(ppmlim=ppmlim, shift=shift)))
    axs[1,0].plot(meanIn.getAxes(ppmlim=ppmlim, axis=axis),np.real(meanIn.getSpectrum(ppmlim=ppmlim, shift=shift)),'k')        
    styleSpectrumAxes(ax=axs[1,0])
    for fid in toPlotOut:
            axs[1,1].plot(fid.getAxes(ppmlim=ppmlim, axis=axis),np.real(fid.getSpectrum(ppmlim=ppmlim, shift=shift)))
    axs[1,1].plot(meanOut.getAxes(ppmlim=ppmlim, axis=axis),np.real(meanOut.getSpectrum(ppmlim=ppmlim, shift=shift)),'k')        
    styleSpectrumAxes(ax=axs[1,1])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10,10))
    plt.plot(meanIn.getAxes(ppmlim=ppmlim, axis=axis),np.real(meanIn.getSpectrum(ppmlim=ppmlim, shift=shift)),'k',label='Unaligned', linewidth=2)
    plt.plot(meanOut.getAxes(ppmlim=ppmlim, axis=axis),np.real(meanOut.getSpectrum(ppmlim=ppmlim, shift=shift)),'r--',label='Aligned', linewidth=2)
    styleSpectrumAxes(ax=plt.gca())
    plt.legend()
    plt.rcParams.update({'font.size': 12})
    plt.show()