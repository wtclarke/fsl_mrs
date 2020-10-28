#!/usr/bin/env python

# general.py - General preprocessing functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT


import numpy as np
from dataclasses import dataclass

@dataclass
class datacontainer:
    '''Class for keeping track of data and reference data together.'''
    data: np.array
    dataheader: dict
    datafilename: str
    reference: np.array = None
    refheader: dict = None
    reffilename: str = None

def get_target_FID(FIDlist,target='mean'):
    """
    target can be 'mean' or 'first' or 'nearest_to_mean' or 'median'
    """
    if target == 'mean':
        return sum(FIDlist) / len(FIDlist)
    elif target == 'first':
        return FIDlist[0].copy()
    elif target == 'nearest_to_mean':
        avg = sum(FIDlist) / len(FIDlist)
        d   = [np.linalg.norm(fid-avg) for fid in FIDlist]
        return FIDlist[np.argmin(d)].copy()
    elif target == 'median':
        return np.median(np.real(np.asarray(FIDlist)),axis=0)+1j*np.median(np.imag(np.asarray(FIDlist)),axis=0)
    else:
        raise(Exception('Unknown target type {}'.format(target)))

def subtract(FID1,FID2):
    """ Subtract FID2 from FID1."""
    return (FID1-FID2)/2.0

def add(FID1,FID2):
    """ Add FID2 to FID1."""
    return (FID1+FID2)/2.0

def add_subtract_report(inFID,inFID2,outFID,hdr,ppmlim=(0.2,4.2),function='Not specified',html=None):
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
    plotIn2 = toMRSobj(inFID2)
    plotOut = toMRSobj(outFID)

    # Fetch line styles
    lines,colors,_ = plotStyles()

    # Make a new figure
    fig = go.Figure()

    # Add lines to figure
    def addline(fig,mrs,lim,name,linestyle):
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim),
                        y=np.real(mrs.get_spec(ppmlim=lim)),
                        mode='lines',
                        name=name,
                        line=linestyle)
        return fig.add_trace(trace)    
    
    fig = addline(fig,plotIn,ppmlim,'FID1',lines['in'])
    fig = addline(fig,plotIn2,ppmlim,'FID2',lines['out'])
    fig = addline(fig,plotOut,ppmlim,'Result',lines['diff'])    

    # Axes layout
    plotAxesStyle(fig,ppmlim,title = f'{function} summary')
   
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
            raise ValueError('Report html path must be file or directory. ')
        
        opName = function
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = f'Report for fsl_mrs.utils.preproc.general.{function}.\n'\
                    + f'Generated at {timestr} on {datestr}.'        
        # Figures
        div = plot(fig, output_type='div',include_plotlyjs='cdn')
        figurelist = [figgroup(fig = div,
                            name= '',
                            foretext= f'Report for {function}.',
                            afttext= f'')]

        singleReport(htmlfile,opName,headerinfo,figurelist)
        return fig
    else:
        return fig


def generic_report(inFID,outFID,inHdr,outHdr,ppmlim = (0.2,4.2),html=None,function=''):
    """
    Generate generic report
    """
    from fsl_mrs.core import MRS
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle

    plotIn = MRS(FID=inFID, header=inHdr)
    plotOut = MRS(FID=outFID, header=outHdr)

    # Fetch line styles
    lines, colors, _ = plotStyles()

    # Make a new figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Spectra', 'FID'])

    # Add lines to figure
    trace1 = go.Scatter(x=plotIn.getAxes(ppmlim=ppmlim),
                        y=np.real(plotIn.get_spec(ppmlim=ppmlim)),
                        mode='lines',
                        name='Original',
                        line=lines['in'])
    trace2 = go.Scatter(x=plotOut.getAxes(ppmlim=ppmlim),
                        y=np.real(plotOut.get_spec(ppmlim=ppmlim)),
                        mode='lines',
                        name='Shifted',
                        line=lines['out'])
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)

    # Add lines to figure
    trace3 = go.Scatter(x=plotIn.getAxes(axis='time'),
                        y=np.real(plotIn.FID),
                        mode='lines',
                        name='Original',
                        line=lines['emph'])
    trace4 = go.Scatter(x=plotOut.getAxes(axis='time'),
                        y=np.real(plotOut.FID),
                        mode='lines',
                        name='Shifted',
                        line=lines['diff'])
    fig.add_trace(trace3, row=1, col=2)
    fig.add_trace(trace4, row=1, col=2)

    # Axes layout
    plotAxesStyle(fig, ppmlim, title=f'{function} summary')
    fig.layout.xaxis2.update(title_text='Time (s)')
    fig.layout.yaxis2.update(zeroline=True,
                             zerolinewidth=1,
                             zerolinecolor='Gray',
                             showgrid=False,
                             showticklabels=False)

    if html is not None:
        from plotly.offline import plot
        from fsl_mrs.utils.preproc.reporting import figgroup, singleReport
        from datetime import datetime
        import os.path as op

        if op.isdir(html):
            filename = 'report_' + datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]+'.html'
            htmlfile = op.join(html, filename)
        elif op.isdir(op.dirname(html)) and op.splitext(html)[1] == '.html':
            htmlfile = html
        else:
            raise ValueError('Report html path must be file or directory. ')

        opName = function
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = f'Report for {function}.\n' + \
                     f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext='',
                               afttext='')]

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig
    else:
        return fig
