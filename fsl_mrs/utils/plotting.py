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
import nibabel as nib
import scipy.ndimage as ndimage
import itertools as it

from fsl_mrs.utils.misc import hz2ppm,FIDToSpec

def FID2Spec(x):
    """
       Turn FID to spectrum for plotting
    """
    x = FIDToSpec(x)
    return x


def plot_fit(mrs,pred=None,ppmlim=(0.40,4.2),out=None,baseline=None,proj='real'):
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
    if baseline is not None:
        baseline = FID2Spec(baseline)

    
    axis   = mrs.ppmAxisShift
    first  = np.argmin(np.abs(axis[0:int(mrs.numPoints/2)]-ppmlim[0]))
    last   = np.argmin(np.abs(axis[0:int(mrs.numPoints/2)]-ppmlim[1]))


    # turn to real numbers
    if proj == "real":
        data,pred = np.real(data),np.real(pred)
        if baseline is not None:
            baseline = np.real(baseline)
    elif proj == "imag":
        data,pred = np.imag(data),np.imag(pred)
        if baseline is not None:
            baseline = np.imag(baseline)
    elif proj == "abs":
        data,pred = np.abs(data),np.abs(pred)
        if baseline is not None:
            baseline = np.abs(baseline)
    elif proj == "angle":
        data,pred = np.angle(data),np.angle(pred)            
        if baseline is not None:
            baseline = np.angle(baseline)

    if first>last:
        first,last = last,first

    m = min(data[first:last].min(),pred[first:last].min())
    M = max(data[first:last].max(),pred[first:last].max())
    ylim   = (m-np.abs(M)/10,M+np.abs(M)/10)
        
    
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
    doPlot(pred,c='#cc0000',linewidth=1,xticks=xticks)
    if baseline is not None:
        doPlot(baseline,c='k',linewidth=.5,xticks=xticks)

    # plot y=0
    doPlot(data*0,c='k',linestyle=':',linewidth=1,xticks=xticks)

    plt.legend(['data','model fit'])

    
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


def plot_spectrum(FID,bandwidth,centralFrequency,ppmlim=(0.0,4.5),proj='real',c='k'):
    """
       Plotting the spectrum 
       ----------
       FID    : array-like
       bandwidth : float (unit = Hz)
       centralFrequency : float (unit = Hz)
       ppmlim : tuple
              (MIN,MAX)
       proj   : string
              one of 'real', 'imag', 'abs', or 'angle'

    """
    numPoints        = FID.size
    frequencyAxis    = np.linspace(-bandwidth/2,
                                   bandwidth/2,
                                   numPoints)    
    ppmAxisShift     = hz2ppm(centralFrequency,
                                   frequencyAxis,shift=True)

    def axes_style(plt,ppmlim,label=None,xticks=None):
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.xlabel(label)
        plt.gca().set_xticks(xticks)
        plt.minorticks_on()
        plt.grid(b=True, axis='x', which='major',color='k', linestyle='--', linewidth=.3)
        plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':',linewidth=.3)
    
    def doPlot(data,c='b',linewidth=1,linestyle='-',xticks=None):
        plt.plot(ppmAxisShift,data,color=c,linewidth=linewidth,linestyle=linestyle)
        axes_style(plt,ppmlim,label='Chemical shift (ppm)',xticks=xticks)


    # Prepare data for plotting
    data = FID2Spec(FID)


    #m = min(np.real(data))
    #M = max(np.real(data))
    #ylim   = (m-np.abs(M)/10,M+np.abs(M)/10)
    #plt.ylim(ylim)
    
    # Create the figure
    #plt.figure(figsize=(7,7))
    xticks = np.linspace(ppmlim[0],ppmlim[1],10)
    exec("doPlot(np.{}(data),c='{}'      ,linewidth=2,xticks=xticks)".format(proj,c))
    
    plt.tight_layout()
    
    return plt.gcf()
    

def plot_spectra(FIDlist,bandwidth,centralFrequency,ppmlim=(0,4.5),single_FID=None,plot_avg=True):
    numPoints        = FIDlist[0].size
    frequencyAxis    = np.linspace(-bandwidth/2,
                                   bandwidth/2,
                                   numPoints)    
    ppmAxisShift     = hz2ppm(centralFrequency,
                              frequencyAxis,shift=True)

    plt.figure(figsize=(10,10))
    plt.xlim(ppmlim)
    plt.gca().invert_xaxis()
    plt.minorticks_on()
    plt.grid(b=True, axis='x', which='major',color='k', linestyle='--', linewidth=.3)
    plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':',linewidth=.3)

    
    avg=0
    for FID in FIDlist:
        data = np.real(FID2Spec(FID))
        avg += data
        plt.plot(ppmAxisShift,data,color='k',linewidth=.5,linestyle='-')
    if single_FID is not None:
        data = np.real(FID2Spec(single_FID))
        plt.plot(ppmAxisShift,data,color='r',linewidth=2,linestyle='-')
    if plot_avg:
        avg /= len(FIDlist)
        plt.plot(ppmAxisShift,avg,color='g',linewidth=2,linestyle='-')
        
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





def plotly_fit(mrs,res,ppmlim=(.2,4.2),proj='real'):
    """
         plot model fitting plus baseline
        
    Parameters:
         mrs    : MRS object
         res    : ResFit Object
         ppmlim : tuple

    Returns
         fig
     """
    def project(x,proj):
        if proj == 'real':
            return np.real(x)
        elif proj == 'imag':
            return np.imag(x)
        elif proj == 'angle':
            return np.angle(x)
        else:
            return np.abs(x)

    # Prepare the data
    axis   = np.flipud(mrs.ppmAxisFlip)
    data   = project(FID2Spec(mrs.FID),proj)
    pred   = project(FID2Spec(res.pred),proj)
    base   = project(FID2Spec(res.baseline),proj)
    resid  = project(FID2Spec(res.residuals),proj)

    # y-axis range
    ymin = np.min(data)-np.min(data)/10
    ymax = np.max(data)-np.max(data)/30

    # Build the plot
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import pandas as pd

    # Table

    df = pd.DataFrame()
    df['Metab']          = mrs.names
    df['mMol/kg']        = np.round(res.conc_h2o,decimals=2)
    df['%CRLB']          = np.round(res.perc_SD[:mrs.numBasis],decimals=1)
    df['/Cr']            = np.round(res.conc_cr,decimals=2)


    fig = ff.create_table(df, height_constant=50)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 16
        
    colors = dict(data='rgb(67,67,67)', 
                  pred='rgb(253,59,59)',
                  base='rgb(170,170,170)',
                  resid='rgb(170,170,170)')
    line_size = dict(data=1, 
                     pred=2,
                     base=1,resid=1)

    trace1 = go.Scatter(x=axis, y=data,
                        mode='lines',
                        name='data',
                        line=dict(color=colors['data'],width=line_size['data']),
                        xaxis='x2', yaxis='y2')
    trace2 = go.Scatter(x=axis, y=pred,
                        mode='lines',
                        name='model',
                        line=dict(color=colors['pred'],width=line_size['pred']),
                        xaxis='x2', yaxis='y2')
    trace3 = go.Scatter(x=axis, y=base,
                        mode='lines',
                        name='baseline',
                        line=dict(color=colors['base'],width=line_size['base']),
                        xaxis='x2', yaxis='y2')
    trace4 = go.Scatter(x=axis, y=resid,
                        mode='lines',
                        name='residuals',
                        line=dict(color=colors['resid'],width=line_size['resid']),
                        xaxis='x2', yaxis='y2')


    fig.add_traces([trace1,trace2,trace3,trace4])
    fig['layout']['xaxis2'] = {}
    fig['layout']['yaxis2'] = {}

    fig.layout.xaxis.update({'domain': [0, .35]})
    fig.layout.xaxis2.update({'domain': [0.4, 1.]})
    fig.layout.xaxis2.update(title_text='Chemical shift (ppm)',
                             tick0=2, dtick=.5,
                             range=[ppmlim[1],ppmlim[0]])
    # The graph's yaxis MUST BE anchored to the graph's xaxis
    fig.layout.yaxis2.update({'anchor': 'x2'})
    fig.layout.yaxis2.update(zeroline=True, 
                             zerolinewidth=1, 
                             zerolinecolor='Gray',
                             showgrid=False,showticklabels=False,
                             range=[ymin,ymax])
    
    # Update the margins to add a title and see graph x-labels.
    fig.layout.margin.update({'t':50, 'b':100})
    fig.layout.update({'title': 'Fitting summary'})
    fig.update_layout(template = 'plotly_white')
    fig.layout.update({'height':800})
    
    return fig


def plot_dist_approx(mrs,res,refname='Cr'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = int(np.ceil(np.sqrt(mrs.numBasis)))
    fig = make_subplots(rows=n, cols=n,subplot_titles=mrs.names)
    traces = []
    ref = res.params[mrs.names.index(refname)]
    for i,metab in enumerate(mrs.names):
        (r, c) = divmod(i, n)
        mu  = res.params[i]/ref
        sig = np.sqrt(res.crlb[i])/ref
        x   = np.linspace(mu-mu,mu+mu,100)
        N = np.exp(-(x-mu)**2/sig**2)
        N = N/N.sum()/(x[1]-x[0])
        t = go.Scatter(x=x,y=N,mode='lines',
                       name=metab,line=dict(color='black'))
        fig.add_trace(t,row=r+1,col=c+1)
    
    fig.update_layout(template = 'plotly_white',
                      showlegend=False,
                      font=dict(size=10),
                      title='Approximate marginal distributions (ref={})'.format(refname),
                      height=800,width=800)
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=10,color='#ff0000')
    
    return fig


def plot_mcmc_corr(mrs,res):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    #Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,
    #Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,
    #Electric,Viridis,Cividis.
    n = mrs.numBasis
    fig = go.Figure()
    corr = np.ma.corrcoef(res.mcmc_samples.T)
    np.fill_diagonal(corr,np.nan)
    fig.add_trace(go.Heatmap(z=corr,
                     x=mrs.names,y=mrs.names,colorscale='Picnic'))
    
    fig.update_layout(template = 'plotly_white',
                      font=dict(size=10),
                      title='MCMC correlation',
                      width = 700,
                      height = 700,
                      yaxis = dict(
                          scaleanchor = "x",
                          scaleratio = 1,
                        ))
    
    return fig

def plot_dist_mcmc(mrs,res,refname='Cr'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = int(np.ceil(np.sqrt(mrs.numBasis)))
    fig = make_subplots(rows=n, cols=n,subplot_titles=mrs.names)
    traces = []
    ref = res.mcmc_samples[:,mrs.names.index(refname)]
    for i,metab in enumerate(mrs.names):
        (r, c) = divmod(i, n)
        x  = res.mcmc_samples[:,i] / np.mean(ref)
        t = go.Histogram(x=x,
                         name=metab,
                        histnorm='percent',marker_color='#330C73',opacity=0.75)
        
    
        fig.add_trace(t,row=r+1,col=c+1)
    
    fig.update_layout(template = 'plotly_white',
                      showlegend=False,
                      width = 700,
                      height = 700,
                      font=dict(size=10),
                      title='MCMC marginal distributions (ref={})'.format(refname))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=10,color='#ff0000')

        
    
    return fig


def plot_real_imag(mrs,res,ppmlim=(.2,4.2)):
    """
         plot model fitting plus baseline
        
    Parameters:
         mrs    : MRS object
         res    : ResFit Object
         ppmlim : tuple

    Returns
         fig
     """
    def project(x,proj):
        if proj == 'real':
            return np.real(x)
        elif proj == 'imag':
            return np.imag(x)
        elif proj == 'angle':
            return np.angle(x)
        else:
            return np.abs(x)
    
    # Prepare the data
    axis        = np.flipud(mrs.ppmAxisFlip)
    data_real   = project(FID2Spec(mrs.FID),'real')
    pred_real   = project(FID2Spec(res.pred),'real')
    data_imag   = project(FID2Spec(mrs.FID),'imag')
    pred_imag   = project(FID2Spec(res.pred),'imag')



    # Build the plot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2,subplot_titles=['Real','Imag'])
    

    colors = dict(data='rgb(67,67,67)', 
                  pred='rgb(253,59,59)',
                  base='rgb(170,170,170)')
    line_size = dict(data=1, 
                     pred=2,
                     base=1)

    trace1 = go.Scatter(x=axis, y=data_real,
                        mode='lines',
                        name='data : real',
                        line=dict(color=colors['data'],width=line_size['data']))
    trace2 = go.Scatter(x=axis, y=pred_real,
                        mode='lines',
                        name='model : real',
                        line=dict(color=colors['pred'],width=line_size['pred']))
    fig.add_trace(trace1,row=1,col=1)
    fig.add_trace(trace2,row=1,col=1)

    trace1 = go.Scatter(x=axis, y=data_imag,
                        mode='lines',
                        name='data : imag',
                        line=dict(color=colors['data'],width=line_size['data']))
    trace2 = go.Scatter(x=axis, y=pred_imag,
                        mode='lines',
                        name='model : imag',
                        line=dict(color=colors['pred'],width=line_size['pred']))
    fig.add_trace(trace1,row=1,col=2)
    fig.add_trace(trace2,row=1,col=2)

#     fig.layout.xaxis.update({'domain': [0, .35]})
#     fig.layout.xaxis2.update({'domain': [0.4, 1.]})
    fig.layout.xaxis.update(title_text='Chemical shift (ppm)',
                             tick0=2, dtick=.5,
                             range=[ppmlim[1],ppmlim[0]])
    fig.layout.xaxis2.update(title_text='Chemical shift (ppm)',
                             tick0=2, dtick=.5,
                             range=[ppmlim[1],ppmlim[0]])

    fig.layout.yaxis2.update(zeroline=True, 
                             zerolinewidth=1, 
                             zerolinecolor='Gray',
                             showgrid=False,showticklabels=False)
    fig.layout.yaxis.update(zeroline=True, 
                             zerolinewidth=1, 
                             zerolinecolor='Gray',
                             showgrid=False,showticklabels=False)
    
    # Update the margins to add a title and see graph x-labels.
#     fig.layout.margin.update({'t':50, 'b':100})
    fig.layout.update({'title': 'Fitting summary Real/Imag'})
    fig.update_layout(template = 'plotly_white')
    fig.layout.update({'height':800,'width':1000})
    
    return fig


def pred(mrs,res,metab):
    from fsl_mrs.utils import models

    if res.model == 'lorentzian':
        forward    = models.FSLModel_forward      # forward model        

        con,gamma,eps,phi0,phi1,b = models.FSLModel_x2param(res.params,mrs.numBasis,res.g)
        c = con[mrs.names.index(metab)].copy()
        con = 0*con
        con[mrs.names.index(metab)] = c
        x = models.FSLModel_param2x(con,gamma,eps,phi0,phi1,b)    

    elif res.model == 'voigt':
        forward    = models.FSLModel_forward_Voigt # forward model

        con,gamma,sigma,eps,phi0,phi1,b = models.FSLModel_x2param_Voigt(res.params,mrs.numBasis,res.g)
        c = con[mrs.names.index(metab)].copy()
        con = 0*con
        con[mrs.names.index(metab)] = c
        x = models.FSLModel_param2x_Voigt(con,gamma,sigma,eps,phi0,phi1,b)
    else:
        raise Exception('Unknown model.')
  
    pred = forward(x,mrs.frequencyAxis,
                            mrs.timeAxis,
                            mrs.basis,res.base_poly,res.metab_groups,res.g)
    pred = np.fft.ifft(pred) # predict FID not Spec
    return pred

def plot_indiv(mrs,res,ppmlim=(.2,4.2)):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    colors = dict(data='rgb(67,67,67)', 
                  pred='rgb(253,59,59)')
    line_size = dict(data=.5, 
                     pred=2)

    ncols = 3
    nrows = int(np.ceil(mrs.numBasis/ncols))

    fig = make_subplots(rows=nrows, cols=ncols,subplot_titles=mrs.names)
    traces = []
    axis        = np.flipud(mrs.ppmAxisFlip)
    for i,metab in enumerate(mrs.names):
        c,r = i%ncols,i//ncols
        #r = i//ncols
        con     = res.params[i]
        y_data  = np.real(FID2Spec(mrs.FID))
        y_fit   = np.real(FID2Spec(pred(mrs,res,metab)))
        
        trace1 = go.Scatter(x=axis, y=y_data,
                        mode='lines',
                        line=dict(color=colors['data'],width=line_size['data']))
        trace2 = go.Scatter(x=axis, y=y_fit,
                        mode='lines',
                        line=dict(color=colors['pred'],width=line_size['pred']))
        fig.add_trace(trace1,row=r+1,col=c+1)
        fig.add_trace(trace2,row=r+1,col=c+1)
    
        fig.update_layout(template = 'plotly_white',
                      showlegend=False,
                      width = 1500,
                      height = 3000,
                      font=dict(size=10),
                      title='Individual fits')
        for j in fig['layout']['annotations']:
            j['font'] = dict(size=10,color='#ff0000')
        
        if i == 0:
            xax = eval("fig.layout.xaxis")
            yax = eval("fig.layout.yaxis")
        else:
            xax = eval("fig.layout.xaxis{}".format(i+1))
            yax = eval("fig.layout.yaxis{}".format(i+1))
        xax.update(tick0=2,dtick=.5,range=[ppmlim[1],ppmlim[0]],showticklabels=False)
        yax.update(zeroline=True, zerolinewidth=1, zerolinecolor='Gray',
                    showgrid=False,showticklabels=False)
    return fig

# def plot_table_extra(mrs,res):
import plotly.graph_objects as go
from plotly.subplots import make_subplots    
import pandas as pd

def plot_table_qc(mrs,res):
    # QC measures
    header=["S/N","Static phase (deg)", "Linear phase (deg/ppm)"]
    values=[np.round(res.snr,decimals=2),
            np.round(res.phi0_deg,decimals=5),
            np.round(res.phi1_deg_per_ppm,decimals=5)]
    table1 = dict(type='table',
                  header=dict(values=header,font=dict(size=10),align="center"),
                  cells=dict(values=values,align = "right"),                  
                  columnorder = [1,2,3],
                  columnwidth = [80,80,80],
                 )
    fig = go.Figure(table1)
    fig.update_layout(template = 'plotly_white')
    fig.layout.update({'width':800})

    return fig
def plot_table_extras(mrs,res):
    # Gamma/Eps
    header = ['group','linewidth (sec)','shift (ppm)','metab groups']
    values = [[],[],[],[]]
    for g in range(res.g):    
        values[0].append(g)
        values[1].append(np.round(res.inv_gamma_sec[g],decimals=3))
        values[2].append(np.round(res.eps_ppm[g],decimals=5))
        metabs = []
        for i,m in enumerate(mrs.names):
            if res.metab_groups[i] == g:
                metabs.append(m)                        
        values[3].append(metabs)
    table2 = go.Table(header=dict(values=header,font=dict(size=10),align="center"),
                      cells=dict(values=values,align = "right"),             
                      columnorder = [1,2,3,4],
                      columnwidth = [12,12,12,64]
                 )
    fig = go.Figure(data=table2)
    fig.update_layout(template = 'plotly_white')
    fig.layout.update({'width':1000,'height':1000})

    return fig



# ----------- Imaging

#!/usr/bin/env python

# Display MRS voxel




# helper functions
def ijk2xyz(ijk,affine):
    """ Return X, Y, Z coordinates for i, j, k """
    ijk = np.asarray(ijk)
    return affine[:3, :3].dot(ijk.T).T + affine[:3, 3]

def xyz2ijk(xyz,affine):
    """ Return i, j, k coordinates for X, Y, Z """
    xyz = np.asarray(xyz)
    inv_affine = np.linalg.inv(affine)
    return inv_affine[:3, :3].dot(xyz.T).T + inv_affine[:3, 3]


def do_plot_slice(slice,rect):
    vmin = np.quantile(slice,.01)
    vmax = np.quantile(slice,.99)
    plt.imshow(slice, cmap="gray", origin="lower",vmin=vmin,vmax=vmax)
    plt.plot(rect[:, 0], rect[:, 1],c='#FF4646',linewidth=2)
    plt.xticks([])
    plt.yticks([])



def plot_voxel_orient(t1file,voxfile):
    """
    Plot T1 centered on voxel
    Overlay voxel in red
    Plots in voxel coordinates 
    """
    t1      = nib.load(t1file)
    vox     = nib.load(voxfile)
    t1_data = t1.get_fdata()

    # centre of MRS voxel in T1 voxel space (or is it the corner?)
    #
    # PM: Nope, it's the voxel centre - this is mandated by the NIFTI spec
    #
    ijk = xyz2ijk(ijk2xyz([0,0,0],vox.affine),t1.affine)
    i,j,k = ijk

    # half size of MRS voxel (careful this assumes 1mm resolution T1)
    si,sj,sk = np.array(vox.header.get_zooms())[:3]/2
    # Do the plotting
    plt.figure(figsize=(15,10))
    plt.subplot(1,3,1)
    slice = np.squeeze(t1_data[int(i),:,:]).T
    rect = np.asarray([[j-sj,k-sk],
                       [j+sj,k-sk],
                       [j+sj,k+sk],
                       [j-sj,k+sk],
                       [j-sj,k-sk]])
    do_plot_slice(slice,rect)
    plt.subplot(1,3,2)
    slice = np.squeeze(t1_data[:,int(j),:]).T
    rect = np.asarray([[i-si,k-sk],
                       [i+si,k-sk],
                       [i+si,k+sk],
                       [i-si,k+sk],
                       [i-si,k-sk]])
    do_plot_slice(slice,rect)
    plt.subplot(1,3,3)
    slice = np.squeeze(t1_data[:,:,int(k)]).T
    rect = np.asarray([[i-si,j-sj],
                       [i+si,j-sj],
                       [i+si,j+sj],
                       [i-si,j+sj],
                       [i-si,j-sj]])
    do_plot_slice(slice,rect)

    return plt.gcf()


def plot_world_orient(t1file,voxfile):
    """
    Plot sagittal/coronal/axial T1 centered on voxel
    Overlay voxel in red
    Plots in world coordinates with the 'MNI' convention
    """
    t1      = nib.load(t1file)
    vox     = nib.load(voxfile)
    t1_data = t1.get_fdata()

    # transform t1 data into world coordinate system,
    # resampling to 1mm^3.
    #
    # This is a touch fiddly because the affine_transform
    # function uses the destination coordinate system as
    # data indices. So we need to remove any translation
    # in the original affine, to ensure that the origin
    # of the destination coordinate system is forced to
    # be (0, 0, 0).
    #
    # We figure all of this out by transforming the image
    # bounding box coordinates into world coordinates,
    # and then figuring out a suitable offset and resampling
    # data shape from them.
    extents       = zip((0, 0, 0), t1_data.shape)
    extents       = np.asarray(list(it.product(*extents)))
    extents       = ijk2xyz(extents, t1.affine)
    offset        = extents.min(axis=0)
    offaff        = np.eye(4)
    offaff[:3, 3] = -offset
    shape         = (extents.max(axis=0) - offset).astype(np.int)[:3]

    t1_data = ndimage.affine_transform(t1_data,
                                       np.dot(offaff, t1.affine),
                                       output_shape=shape,
                                       order=3,
                                       mode='constant',
                                       cval=0)

    # centre of MRS voxel in (transformed) T1 voxel space
    ijk = xyz2ijk(ijk2xyz([0,0,0],vox.affine),np.linalg.inv(offaff))
    i,j,k = ijk

    
    si,sj,sk = np.array(vox.header.get_zooms())[:3]/2
    # Do the plotting
    plt.figure(figsize=(15,10))
    plt.subplot(1,3,1)
    slice = np.squeeze(t1_data[int(i),:,:]).T
    rect = np.asarray([[j-sj,k-sk],
                       [j+sj,k-sk],
                       [j+sj,k+sk],
                       [j-sj,k+sk],
                       [j-sj,k-sk]])
    do_plot_slice(slice,rect)
    plt.subplot(1,3,2)
    slice = np.squeeze(t1_data[:,int(j),:]).T
    rect = np.asarray([[i-si,k-sk],
                       [i+si,k-sk],
                       [i+si,k+sk],
                       [i-si,k+sk],
                       [i-si,k-sk]])
    do_plot_slice(slice,rect)
    plt.gca().invert_xaxis()
    plt.subplot(1,3,3)
    slice = np.squeeze(t1_data[:,:,int(k)]).T
    rect = np.asarray([[i-si,j-sj],
                       [i+si,j-sj],
                       [i+si,j+sj],
                       [i-si,j+sj],
                       [i-si,j-sj]])
    do_plot_slice(slice,rect)
    plt.gca().invert_xaxis()

    return plt.gcf()


