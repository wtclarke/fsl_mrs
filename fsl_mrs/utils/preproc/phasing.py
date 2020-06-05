#!/usr/bin/env python

# phasing.py - Phase correction routines
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.core import MRS
from fsl_mrs.utils.misc import extract_spectrum
from fsl_mrs.utils.preproc.shifting import pad
from fsl_mrs.utils.preproc.remove import hlsvd
def applyPhase(FID,phaseAngle):
    """
    Multiply FID by constant phase
    """
    return FID * np.exp(1j*phaseAngle)

def phaseCorrect(FID,bw,cf,ppmlim=(2.8,3.2),shift=True):
    """ Phase correction based on the phase of a maximum point.

    HLSVD is used to remove peaks outside the limits to flatten baseline first.

    Args:
        FID (ndarray): Time domain data
        bw (float): bandwidth
        cf (float): central frequency in Hz
        ppmlim (tuple,optional)  : Limit to this ppm range
        shift (bool,optional)    : Apply H20 shft

    Returns:
        FID (ndarray): Phase corrected FID
    """
    # Run HLSVD to remove peaks outside limits
    try:
        fid_hlsvd = hlsvd(FID,1/bw,cf,(ppmlim[1]+0.5,ppmlim[1]+3.0),limitUnits='ppm+shift')
        fid_hlsvd = hlsvd(fid_hlsvd,1/bw,cf,(ppmlim[0]-3.0,ppmlim[0]-0.5),limitUnits='ppm+shift')
    except:
        fid_hlsvd = FID
        print('Phasing HLSVD failed, proceeding to phaseing.')
    #Find maximum of absolute spectrum in ppm limit
    padFID = pad(fid_hlsvd,FID.size*3)
    MRSargs = {'FID':padFID,'bw':bw,'cf':cf}
    mrs = MRS(**MRSargs)
    spec = extract_spectrum(mrs,padFID,ppmlim=ppmlim,shift=shift)

    maxIndex = np.argmax(np.abs(spec))
    phaseAngle = -np.angle(spec[maxIndex])
    
    return applyPhase(FID,phaseAngle),phaseAngle,int(np.round(maxIndex/4))

def phaseCorrect_report(inFID,outFID,hdr,position,ppmlim=(2.8,3.2),html=None):
    """
    Generate report
    """
    # from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    import plotly.graph_objects as go    
    from fsl_mrs.utils.preproc.reporting import plotStyles,plotAxesStyle

    # Turn input FIDs into mrs objects
    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    plotIn = toMRSobj(inFID)
    plotOut = toMRSobj(outFID)

    widelimit = (0,6)

    # Fetch line styles
    lines,colors,_ = plotStyles()

    # Make a new figure
    fig = go.Figure()

    # Add lines to figure
    def addline(fig,mrs,lim,name,linestyle):
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim),
                        y=np.real(mrs.getSpectrum(ppmlim=lim)),
                        mode='lines',
                        name=name,
                        line=linestyle)
        return fig.add_trace(trace)    
    
    fig = addline(fig,plotIn,widelimit,'Unphased',lines['in'])
    fig = addline(fig,plotIn,ppmlim,'Search region',lines['emph'])    

    if position is None:
        # re-estimate here.
        position = np.argmax(np.abs(plotIn.getSpectrum(ppmlim=ppmlim)))

    axis    = [plotIn.getAxes(ppmlim=ppmlim)[position]]
    y_data  = [np.real(plotIn.getSpectrum(ppmlim=ppmlim))[position]]
    trace = go.Scatter(x=axis, y=y_data,
                        mode='markers',
                        name='max point',
                        marker=dict(color=colors['emph'],symbol='x',size=8))
    fig.add_trace(trace)

    fig = addline(fig,plotOut,widelimit,'Phased',lines['out'])

    # Axes layout
    plotAxesStyle(fig,widelimit,title = 'Phase correction summary')
   
    # Axea 
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
        
        opName = 'Phase correction'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.preproc.phasing.phaseCorrect.\n'\
                    + f'Generated at {timestr} on {datestr}.'        
        # Figures
        div = plot(fig, output_type='div',include_plotlyjs='cdn')
        figurelist = [figgroup(fig = div,
                            name= '',
                            foretext= f'Phase correction of spectra based on maximum in the range {ppmlim[0]} to {ppmlim[1]} ppm.',
                            afttext= f'')]

        singleReport(htmlfile,opName,headerinfo,figurelist)
        return fig
    else:
        return fig

# matplotlib version of report
# def phaseCorrect_report(inFID,outFID,hdr,position,ppmlim=(2.8,3.2)):
#     from matplotlib import pyplot as plt
#     from fsl_mrs.core import MRS
#     from fsl_mrs.utils.plotting import styleSpectrumAxes

#     toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
#     plotIn = toMRSobj(inFID)
#     plotOut = toMRSobj(outFID)

#     widelimit = (0,6)

#     fig = plt.figure(figsize=(10,10))
#     plt.plot(plotIn.getAxes(ppmlim=widelimit),np.real(plotIn.getSpectrum(ppmlim=widelimit)),'k',label='Unphased', linewidth=2)
#     plt.plot(plotIn.getAxes(ppmlim=ppmlim),np.real(plotIn.getSpectrum(ppmlim=ppmlim)),'r',label='search region', linewidth=2)
#     plt.plot(plotIn.getAxes(ppmlim=ppmlim)[position],np.real(plotIn.getSpectrum(ppmlim=ppmlim))[position],'rx',label='max point', linewidth=2)
#     plt.plot(plotOut.getAxes(ppmlim=widelimit),np.real(plotOut.getSpectrum(ppmlim=widelimit)),'b--',label='Phased', linewidth=2)
#     styleSpectrumAxes(ax=plt.gca())
#     plt.legend()
#     plt.rcParams.update({'font.size': 12})
#     plt.show()
