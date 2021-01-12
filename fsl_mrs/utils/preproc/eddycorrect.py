#!/usr/bin/env python

# eddycorrect.py - Eddy current correction routines
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

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

def eddy_correct_report(inFID,outFID,phsRef,hdr,ppmlim = (0.2,4.2),html=None):
    """
    Generate Eddy correction report
    """
    # from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    import plotly.graph_objects as go    
    from fsl_mrs.utils.preproc.reporting import plotStyles,plotAxesStyle

    # Turn input FIDs into mrs objects
    toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
    plotIn = toMRSobj(inFID)
    plotOut = toMRSobj(outFID)
    plotRef = toMRSobj(phsRef)

    # Fetch line styles
    lines,colors,_ = plotStyles()

    # Make a new figure
    fig = go.Figure()

    # Add lines to figure
    def addline(fig,mrs,lim,name,linestyle):        
        y = np.real(mrs.get_spec(ppmlim=lim))
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim),
                        y=y,
                        mode='lines',
                        name=name,
                        line=linestyle)
        return fig.add_trace(trace)    
    
    fig = addline(fig,plotIn,ppmlim,'Uncorrected',lines['in'])
    fig = addline(fig,plotOut,ppmlim,'Corrected',lines['out'])

    # Axes layout
    plotAxesStyle(fig,ppmlim,title = 'ECC summary')
   
    # Second figure
    def addlinephs(fig,mrs,name,linestyle):                
        trace = go.Scatter(x=mrs.getAxes(axis='time'),
                        y=np.unwrap(np.angle(mrs.FID)),
                        mode='lines',
                        name=name,
                        line=linestyle)
        return fig.add_trace(trace)    
    # Make a new figure
    fig2 = go.Figure() 
    fig2 = addlinephs(fig2,plotIn,'Uncorrected',lines['in'])
    fig2 = addlinephs(fig2,plotOut,'Corrected',lines['out'])
    fig2 = addlinephs(fig2,plotRef,'Reference',lines['diff'])
    fig2.layout.yaxis.update(title_text='Angle (radians)')
    fig2.layout.xaxis.update(title_text='Time (s)')
    fig2.layout.update({'title': 'FID Phase'})
    fig2.update_layout(template = 'plotly_white')

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
        
        opName = 'ECC'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.eddycorrect.eddy_correct.\n'\
                    + f'Generated at {timestr} on {datestr}.'        
        # Figures
        div = plot(fig, output_type='div',include_plotlyjs='cdn')
        figurelist = [figgroup(fig = div,
                            name= '',
                            foretext= f'Eddy correction summary.',
                            afttext= f'')]
        div2 = plot(fig2, output_type='div',include_plotlyjs='cdn')
        figurelist.append(figgroup(fig = div2,
                            name= '',
                            foretext= f'Signal + reference phases.',
                            afttext= f''))

        singleReport(htmlfile,opName,headerinfo,figurelist)
        return fig,fig2
    else:
        return fig,fig2
