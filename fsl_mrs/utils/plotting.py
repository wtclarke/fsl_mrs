#!/usr/bin/env python

# plotting.py - MRS plotting helper functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly import tools



def FID2Spec(x):
    """
       Turn FID to spectrum for plotting
    """
    def scaleFID(x):
        x[0] *= 0.5
        return x
    x = np.fft.fftshift(np.fft.fft(scaleFID(np.conj(x))))
    #x = np.fft.fft(np.conj(x))
    return x


def plot_fit(mrs,pred=None,ppmlim=(0.40,4.2),out=None,proj='real'):
    """
       Main function for plotting a model fit
       Parameters
       ----------
       mrs    : MRS Object
       pred   : array-like
              predicted FID. If not provided, tries to get it from mrs object
       ppmlim : tuple
              (MIN,MAX)
       out    : string
              output figure filename
       proj   : string
              one of 'real', 'imag', 'abs', or 'angle'

    """

    def axes_style(plt,ppmlim,label=None,xticks=None):
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.xlabel(label)
        plt.gca().set_xticks(xticks)
        plt.minorticks_on()
        plt.grid(b=True, axis='x', which='major',color='k', linestyle='--', linewidth=.3)
        plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':',linewidth=.3)
    
    def doPlot(data,c='b',linewidth=1,linestyle='-',xticks=None):
        plt.plot(mrs.ppmAxisShift,data,color=c,linewidth=linewidth,linestyle=linestyle)
        axes_style(plt,ppmlim,label='Chemical shift (ppm)',xticks=xticks)


    # Prepare data for plotting
    data = FID2Spec(mrs.FID)
    if pred is None:
        pred = mrs.pred
    pred = FID2Spec(pred)

    
    axis   = mrs.ppmAxisShift
    first  = np.argmin(np.abs(axis[0:int(mrs.numPoints/2)]-ppmlim[0]))
    last   = np.argmin(np.abs(axis[0:int(mrs.numPoints/2)]-ppmlim[1]))



    # turn to real numbers
    if proj == "real":
        data,pred = np.real(data),np.real(pred)
    elif proj == "imag":
        data,pred = np.imag(data),np.imag(pred)
    elif proj == "abs":
        data,pred = np.abs(data),np.abs(pred)
    elif proj == "angle":
        data,pred = np.angle(data),np.angle(pred)            


    
    if first>last:
        first,last = last,first
    ylim   = (data[first:last].min()-np.abs(data[first:last]).min()/30,
              data[first:last].max()+np.abs(data[first:last]).max()/50)
    
    
    # Create the figure
    plt.figure(figsize=(9,10))

    # Subplots        
    gs = gridspec.GridSpec(2, 1,                               
                           height_ratios=[1, 20])

    ax1 = plt.subplot(gs[0])
    # Start by plotting error
    xticks = np.arange(ppmlim[0],ppmlim[1]+.2,.2)
    exec("plt.plot(mrs.ppmAxisShift,np.{}(data-pred),c='k',linewidth=1,linestyle='-')".format(proj))
    axes_style(plt,ppmlim,xticks=xticks)
    plt.gca().set_xticklabels([])

    ax2 = plt.subplot(gs[1])
      
    doPlot(data,c='k'      ,linewidth=.5,xticks=xticks)
    doPlot(pred,c='#cc0000',linewidth=3,xticks=xticks)
    plt.legend(['data','model fit'])

    if mrs.baseline is not None:
        doPlot(np.fft.fftshift(mrs.baseline),xticks=xticks)
    
    plt.tight_layout()
    plt.ylim(ylim)
    
    
    if out is not None:
        plt.savefig(out)

    return plt.gcf()

def plot_fit_new(mrs,ppmlim=(0.40,4.2)):
    """
        plot model fitting plus baseline
        
        mrs : MRS object
        ppmlim : tuple
    """
    axis   = np.flipud(mrs.ppmAxisFlip)
    spec   = np.flipud(np.fft.fftshift(mrs.Spec))
    pred   = np.fft.fft(mrs.pred)
    pred   = np.flipud(np.fft.fftshift(pred))

    if mrs.baseline is not None:
        B = np.flipud(np.fft.fftshift(mrs.baseline))
    
    first  = np.argmin(np.abs(axis-ppmlim[0]))
    last   = np.argmin(np.abs(axis-ppmlim[1]))
    if first>last:
        first,last = last,first    
    freq = axis[first:last] 

    plt.figure(figsize=(9,10))
    plt.plot(axis[first:last],spec[first:last])
    plt.gca().invert_xaxis()
    plt.plot(axis[first:last],pred[first:last],'r')
    if mrs.baseline is not None:
        plt.plot(axis[first:last],B[first:last],'k')

    # style stuff
    plt.minorticks_on()
    plt.grid(b=True, axis='x', which='major',color='k', linestyle='--', linewidth=.3)
    plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':',linewidth=.3)
 
    return plt.gcf()



def plot_waterfall(mrs,ppmlim=(0.4,4.2),proj='real',mod=True):
    """
       Plot individual metabolit spectra
       
       Parameters
       ----------

       ppmlim : tuple
       proj   : either 'real' or 'imag' or 'abs' or 'angle'
       mod    : True or False
                whether to multiply by estimated concentrations or not
    """
    gs = gridspec.GridSpec(mrs.numBasis, 1)
    fig = plt.figure(figsize=(5,10))

        
    for i in range(mrs.numBasis):
        ax = plt.subplot(gs[i])
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().set_ylabel(mrs.names[i],rotation='horizontal')
        plt.box(False)

        if mod and mrs.con is not None:
            data = FID2Spec(mrs.con[i]*mrs.basis[:,i])
        else:
            data = FID2Spec(mrs.basis[:,i])
        exec("plt.plot(mrs.ppmAxisShift,np.{}(data),c='r',linewidth=1,linestyle='-')".format(proj))
    
    return fig


def plot_spectrum(mrs,FID=None,ppmlim=(0.0,4.5),proj='real'):
    """
       Plotting the spectrum 
       ----------
       mrs    : MRS Object
       FID    : array-like
               If not provided, plots FID in mrs
       ppmlim : tuple
              (MIN,MAX)
       proj   : string
              one of 'real', 'imag', 'abs', or 'angle'

    """

    def axes_style(plt,ppmlim,label=None,xticks=None):
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.xlabel(label)
        plt.gca().set_xticks(xticks)
        plt.minorticks_on()
        plt.grid(b=True, axis='x', which='major',color='k', linestyle='--', linewidth=.3)
        plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':',linewidth=.3)
    
    def doPlot(data,c='b',linewidth=1,linestyle='-',xticks=None):
        plt.plot(mrs.ppmAxisShift,data,color=c,linewidth=linewidth,linestyle=linestyle)
        axes_style(plt,ppmlim,label='Chemical shift (ppm)',xticks=xticks)


    # Prepare data for plotting
    if FID is  None:
        data = FID2Spec(mrs.FID)
    else:
        data = FID2Spec(FID)

    # Create the figure
    #plt.figure(figsize=(7,7))
    xticks = np.linspace(ppmlim[0],ppmlim[1],10)
    exec("doPlot(np.{}(data),c='k'      ,linewidth=2,xticks=xticks)".format(proj))
    
    plt.tight_layout()
    
    return plt.gcf()
    



def plot_fit_pretty(mrs,pred=None,ppmlim=(0.40,4.2),proj='real'):
    """
       Main function for plotting a model fit
       Parameters
       ----------
       mrs    : MRS Object
       pred   : array-like
              predicted FID. If not provided, tries to get it from mrs object
       ppmlim : tuple
              (MIN,MAX)
       out    : string
              output figure filename
       proj   : string
              one of 'real', 'imag', 'abs', or 'angle'

    """

    if pred is None:
        pred = mrs.pred
    
    #exec("data = np.{}(FID2Spec(mrs.FID))".format(proj))
    #exec("pred = np.{}(FID2Spec(pred))".format(proj))
    data = np.real(FID2Spec(mrs.FID))
    pred = np.real(FID2Spec(pred))
    err  = data-pred
    x    = mrs.ppmAxisShift


    fig = tools.make_subplots(rows=2,
                              row_width=[10, 1],
                              shared_xaxes=True,
                              print_grid=False,
                              vertical_spacing=0)

    trace_data = go.Scatter(x=x, y=data, name='data', hoverinfo="none")
    trace_pred = go.Scatter(x=x, y=pred, name='pred', hoverinfo="none")
    trace_err  = go.Scatter(x=x, y=err, name='error', hoverinfo="none")

    fig.append_trace(trace_err, 1, 1)
    fig.append_trace(trace_data, 2, 1)
    fig.append_trace(trace_pred, 2, 1)


    fig['layout'].update(autosize=True,
                         title=None,
                         showlegend=True,
                         margin={'t': 0.01, 'r': 0, 'l':20})

    fig['layout']['xaxis'].update(zeroline=False,
                                  title='Chemical shift (ppm)',
                                  automargin=True,
                                  range=[ppmlim[1],ppmlim[0]])
    fig['layout']['yaxis'].update(zeroline=False, automargin=True)
    fig['layout']['yaxis2'].update(zeroline=False, automargin=True)

    return fig


