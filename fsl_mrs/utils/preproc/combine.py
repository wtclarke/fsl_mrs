#!/usr/bin/env python

# combine.py - Module containing functions for combining FIDs, includes coil combination
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT



import numpy as np

def dephase(FIDlist):
    """
      Uses first data point of each FID to dephase each FID in list
      Returns a list of dephased FIDs
    """
    return [fid*np.exp(-1j*np.angle(fid[0])) for fid in FIDlist]

def prewhiten(FIDlist,prop=.1,C=None):
    """
       Uses noise covariance to prewhiten data

    Parameters:
    -----------
    FIDlist : list of FIDs
    prop    : proportion of data used to estimate noise covariance
    C       : noise covariance matrix, if provided it is not measured from data.

    Returns : 
    list of FIDs
    pre-whitening matrix
    noise covariance matrix
    """
    FIDs = np.asarray(FIDlist,dtype=np.complex)
    if C is None:
        # Estimate noise covariance
        start = int((1-prop)*FIDs.shape[0])
        # Raise warning if not enough samples
        if (FIDs.shape[0]-start)<1.5*FIDs.shape[1]:
            raise(Warning('You may not have enough samples to robustly estimate the noise covariance'))
        C     = np.cov(FIDs[start:,:],rowvar=False)

    D,V   = np.linalg.eigh(C,UPLO='U') #UPLO = 'U' to match matlab implementation    
    # Pre-whitening matrix
    W     = V@np.diag(1/np.sqrt(D))
    # Pre-whitened data
    FIDs = FIDs@W
    return FIDs,W,C

def svd_reduce(FIDlist,W=None,C=None,return_alpha=False):
    """
    Combine different channels by SVD method
    Based on C.T. Rodgers and M.D. Robson, Magn Reson Med 63:881–891, 2010

    Parameters:
    -----------
    FIDlist      : list of FIDs
    W            : pre-whitening matrix (only used to calculate sensitivities)
    return_alpha : return sensitivities?

    Returns:
    --------
    array-like (FID)
    array-like (sensitivities) - optional
    """
    FIDs  = np.asarray(FIDlist)
    U,S,V = np.linalg.svd(FIDs,full_matrices=False)
    # U,S,V = svds(FIDs,k=1)  #this is much faster but fails for two coil case

    nCoils = FIDs.shape[1]
    svdQuality = ((S[0]/np.linalg.norm(S))*np.sqrt(nCoils)-1)/(np.sqrt(nCoils)-1)

    # get arbitrary amplitude
    iW = np.eye(FIDs.shape[1])
    if W is not None:
        iW = np.linalg.inv(W)
    amp = V[0,:]@iW
    
    # arbitrary scaling here such that the first coil weight is real and positive
    svdRescale = np.linalg.norm(amp)*(amp[0]/np.abs(amp[0]))

    # combined channels
    FID = U[:,0]*S[0]*svdRescale

    if return_alpha:
        # sensitivities per channel        
        # alpha = amp/svdRescale # equivalent to svdCoilAmplitudes in matlab implementation

        # Instead incorporate the effect of the whitening stage as well.
        if C is None:
            C = np.eye(FIDs.shape[1])        
        scaledAmps = (amp/svdRescale).conj().T
        alpha = np.linalg.inv(C)@scaledAmps * svdRescale.conj() * svdRescale    
        return FID,alpha
    else:
        return FID

def weightedCombination(FIDlist,weights):
    """
    Combine different FIDS with different complex weights

    Parameters:
    -----------
    FIDlist      : list of FIDs
    weights      : complex weights

    Returns:
    --------
    array-like (FID)    
    """
    if isinstance(FIDlist,list):
        FIDlist  = np.asarray(FIDlist)
    if isinstance(weights,list):
        weights = np.asarray(weights)
    # combine channels
    FID = np.sum(FIDlist*weights[None,:],axis=1)

    return FID

def combine_FIDs(FIDlist,method,do_prewhiten=False,do_dephase=False,do_phase_correct=False,weights=None):
    """
       Combine FIDs (either from multiple coils or multiple averages)
    
    Parameters:
    -----------
    FIDlist   : list of FIDs or array with time dimension first
    method    : one of 'mean', 'svd', 'svd_weights', 'weighted'
    prewhiten : bool
    dephase   : bool

    Returns:
    --------
    array-like

    """

    if isinstance(FIDlist,list):
        FIDlist = np.asarray(FIDlist).T

    # Pre-whitening
    W = None
    C = None
    if do_prewhiten:
        FIDlist,W,C = prewhiten(FIDlist)

    # Dephasing
    if do_dephase:
        FIDlist   = dephase(FIDlist)

    # Combining channels
    if method == 'mean':
        return np.mean(FIDlist,axis=-1).T
    elif method == 'svd':
        return svd_reduce(FIDlist,W)
    elif method == 'svd_weights':
        return svd_reduce(FIDlist,W,C,return_alpha=True)
    elif method == 'weighted':
        return weightedCombination(FIDlist,weights)
    else:
        raise(Exception("Unknown method '{}'. Should be either 'mean' or 'svd'".format(method)))


def combine_FIDs_report(inFIDs,outFID,hdr,ncha=2,ppmlim = (0.0,6.0),method='not specified',html=None):
    """ Take list of FIDs that are passed to combine and output

    If uncombined data it will display ncha channels (default 2).
    """
    from fsl_mrs.core import MRS
    import plotly.graph_objects as go    
    from fsl_mrs.utils.preproc.reporting import plotStyles,plotAxesStyle
    from matplotlib.pyplot import cm
    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)

    # Assemble data to plot
    toPlotIn = []
    colourVecIn = []
    legendIn = []
    if isinstance(inFIDs,list):    
        for idx,fid in enumerate(inFIDs):
            if inFIDs[0].ndim>1:
                toPlotIn.extend([toMRSobj(f) for f in fid[:,:ncha].T])
                colourVecIn.extend([idx/len(inFIDs)]*ncha)
                legendIn.extend([f'FID #{idx}: CHA #{jdx}' for jdx in range(ncha)])
            else:
                toPlotIn.append(toMRSobj(fid))
                colourVecIn.append(idx/len(inFIDs))
                legendIn.append(f'FID #{idx}')
            
    elif inFIDs.ndim>1:
        toPlotIn.extend([toMRSobj(f) for f in inFIDs[:,:ncha].T])
        colourVecIn.extend([float(jdx)/ncha for jdx in range(ncha)])
        legendIn.extend([f'FID #0: CHA #{jdx}' for jdx in range(ncha)])

    toPlotOut = []
    legendOut = []
    if outFID.ndim>1:
        toPlotOut.extend([toMRSobj(f) for f in outFID[:,:ncha].T])
        legendOut.extend([f'Combined: CHA #{jdx}' for jdx in range(ncha)])
    else:
        toPlotOut.append(toMRSobj(outFID))
        legendOut.append('Combined')

    def addline(fig,mrs,lim,name,linestyle):
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim),
                        y=np.real(mrs.getSpectrum(ppmlim=lim)),
                        mode='lines',
                        name=name,
                        line=linestyle)
        return fig.add_trace(trace)

    lines,colors,_ = plotStyles()
    colors = cm.Spectral(np.array(colourVecIn).ravel())

    fig = go.Figure()
    for idx,fid in enumerate(toPlotIn):
        cval = np.round(255*colors[idx,:])
        linetmp = {'color':f'rgb({cval[0]},{cval[1]},{cval[2]})','width':1}
        fig = addline(fig,fid,ppmlim,legendIn[idx],linetmp)
    
    for idx,fid in enumerate(toPlotOut):
        fig = addline(fig,fid,ppmlim,legendOut[idx],lines['blk'])
    plotAxesStyle(fig,ppmlim,'Combined')

    # Generate report 
    if html is not None:
        from plotly.offline import plot
        from fsl_mrs.utils.preproc.reporting import figgroup, singleReport
        from datetime import datetime
        import os.path as op

        if op.isdir(html):
            filename = 'report_' + datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]+'.html'
            htmlfile=op.join(html,filename)
        elif op.isdir(op.dirname(html)) and op.splitext(html)[1]=='.html':
            htmlfile = html
        else:
            raise ValueError('html must be file ')
        
        opName = 'Combination'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.preproc.combine.combine_FIDs.\n'\
                    + f'Generated at {timestr} on {datestr}.'        
        # Figures
        div = plot(fig, output_type='div',include_plotlyjs='cdn')
        figurelist = [figgroup(fig = div,
                            name= '',
                            foretext= f'Combination of spectra. Method = {method}',
                            afttext= f'')]

        singleReport(htmlfile,opName,headerinfo,figurelist)
        return fig
    else:
        return fig

#  Matplotlib
# def combine_FIDs_report(inFIDs,outFID,hdr,fileout=None,ncha=2):
#     """ Take list of FIDs that are passed to combine and output

#     If uncombined data it will display ncha channels (default 2).
#     """
#     from matplotlib import pyplot as plt
#     from fsl_mrs.core import MRS
#     from fsl_mrs.utils.plotting import styleSpectrumAxes

#     toMRSobj = lambda fid : MRS(FID=fid,header=hdr)

#     # Assemble data to plot
#     toPlotIn = []
#     if isinstance(inFIDs,list):    
#         for fid in inFIDs:
#             if inFIDs[0].ndim>1:
#                 toPlotIn.extend([toMRSobj(f) for f in fid[:,:ncha].T])
#             else:
#                 toPlotIn.append(toMRSobj(fid))
            
#     elif inFIDs.ndim>1:
#         toPlotIn.extend([toMRSobj(f) for f in inFIDs[:,:ncha].T])

#     toPlotOut = []
#     if outFID.ndim>1:
#         toPlotOut.extend([toMRSobj(f) for f in outFID[:,:ncha].T])
#     else:
#         toPlotOut.append(toMRSobj(outFID))
    
#     ppmlim = (0.0,6.0)
#     ax = plt.gca()
#     colourvec = [[i/len(toPlotIn)]*ncha for i in range(len(inFIDs))]
#     colors = plt.cm.Spectral(np.array(colourvec).ravel())
#     style = ['--']*len(colors)
#     ax.set_prop_cycle(color =colors,linestyle=style)
#     for fid in toPlotIn:
#         ax.plot(fid.getAxes(ppmlim=ppmlim),np.real(fid.getSpectrum(ppmlim=ppmlim)))
#     for fid in toPlotOut:
#         ax.plot(fid.getAxes(ppmlim=ppmlim),np.real(fid.getSpectrum(ppmlim=ppmlim)),'k-')    
#     styleSpectrumAxes(ax)
#     plt.show()
