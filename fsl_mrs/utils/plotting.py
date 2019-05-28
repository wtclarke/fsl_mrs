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



def FID2Spec(x):
    """
       Turn FID to spectrum for plotting
    """
    def scaleFID(x):
        x[0] *= 0.5
        return x
    x = np.fft.fftshift(np.fft.fft(scaleFID(np.conj(x))))/x.size
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

    # Create the figure
    plt.figure(figsize=(10,10))

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
      
    exec("doPlot(np.{}(data),c='k'      ,linewidth=.5,xticks=xticks)".format(proj))
    exec("doPlot(np.{}(pred),c='#cc0000',linewidth=3,xticks=xticks)".format(proj))
    plt.legend(['data','model fit'])
    
    plt.tight_layout()
    
    if out is not None:
        plt.savefig(out)

    return plt.gcf()
    
def plot_waterfall(mrs,ppmlim=(0.4,4.2),proj='real',mod=True):
    gs = gridspec.GridSpec(mrs.numBasis, 1)
    plt.figure(figsize=(5,10))

        
    for i in range(mrs.numBasis):
        ax = plt.subplot(gs[i])
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().set_ylabel(mrs.names[i],rotation='horizontal')
        plt.box(False)

        if mod:
            data = FID2Spec(mrs.con[i]*self.basis[:,i])
        else:
            data = FID2Spec(mrs.basis[:,i])
        exec("plt.plot(mrs.ppmAxisShift,np.{}(data),c='r',linewidth=1,linestyle='-')".format(proj))



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
    plt.figure(figsize=(7,7))
    xticks = np.linspace(ppmlim[0],ppmlim[1],10)
    exec("doPlot(np.{}(data),c='k'      ,linewidth=2,xticks=xticks)".format(proj))
    
    plt.tight_layout()
    
    #return plt.gcf()
    
