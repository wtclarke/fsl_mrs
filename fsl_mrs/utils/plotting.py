#!/usr/bin/env python

# plotting.py - MRS plotting helper functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from fsl_mrs.utils import mrs_io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from plotly import tools
import nibabel as nib
import scipy.ndimage as ndimage
import itertools as it

from fsl_mrs.utils.misc import FIDToSpec, SpecToFID, limit_to_range


def FID2Spec(x):
    """
       Turn FID to spectrum for plotting
    """
    x = FIDToSpec(x)
    return x


def plot_fit(mrs, pred=None, ppmlim=(0.40, 4.2),
             out=None, baseline=None, proj='real'):
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

    def axes_style(plt, ppmlim, label=None, xticks=None):
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.xlabel(label)
        plt.gca().set_xticks(xticks)
        plt.minorticks_on()
        plt.grid(b=True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
        plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

    def doPlot(data, c='b', linewidth=1, linestyle='-', xticks=None):
        plt.plot(mrs.getAxes(), data, color=c, linewidth=linewidth, linestyle=linestyle)
        axes_style(plt, ppmlim, label='Chemical shift (ppm)', xticks=xticks)

    # Prepare data for plotting
    data = FID2Spec(mrs.FID)
    if pred is None:
        pred = mrs.pred
    pred = FID2Spec(pred)
    if baseline is not None:
        baseline = FID2Spec(baseline)

    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim, shift=True)

    # turn to real numbers
    if proj == "real":
        data, pred = np.real(data), np.real(pred)
        if baseline is not None:
            baseline = np.real(baseline)
    elif proj == "imag":
        data, pred = np.imag(data), np.imag(pred)
        if baseline is not None:
            baseline = np.imag(baseline)
    elif proj == "abs":
        data, pred = np.abs(data), np.abs(pred)
        if baseline is not None:
            baseline = np.abs(baseline)
    elif proj == "angle":
        data, pred = np.angle(data), np.angle(pred)
        if baseline is not None:
            baseline = np.angle(baseline)

    if first > last:
        first, last = last, first

    m = min(data[first:last].min(), pred[first:last].min())
    M = max(data[first:last].max(), pred[first:last].max())
    ylim = (m - np.abs(M) / 10, M + np.abs(M) / 10)

    # Create the figure
    plt.figure(figsize=(9, 10))

    # Subplots
    gs = gridspec.GridSpec(2, 1,
                           height_ratios=[1, 20])

    plt.subplot(gs[0])
    # Start by plotting error
    xticks = np.arange(ppmlim[0], ppmlim[1] + .2, .2)
    plt.plot(mrs.getAxes(), data_proj(data - pred, proj), c='k', linewidth=1, linestyle='-')
    axes_style(plt, ppmlim, xticks=xticks)
    plt.gca().set_xticklabels([])

    plt.subplot(gs[1])

    doPlot(data, c='k', linewidth=.5, xticks=xticks)
    doPlot(pred, c='#cc0000', linewidth=1, xticks=xticks)
    if baseline is not None:
        doPlot(baseline, c='k', linewidth=.5, xticks=xticks)

    # plot y=0
    doPlot(data * 0, c='k', linestyle=':', linewidth=1, xticks=xticks)

    plt.legend(['data', 'model fit'])

    plt.tight_layout()
    plt.ylim(ylim)

    if out is not None:
        plt.savefig(out)

    return plt.gcf()


def plot_fit_new(mrs, ppmlim=(0.40, 4.2)):
    """
        plot model fitting plus baseline

        mrs : MRS object
        ppmlim : tuple
    """
    axis = mrs.getAxes()
    spec = np.flipud(np.fft.fftshift(mrs.get_spec()))
    pred = FIDToSpec(mrs.pred)
    pred = np.flipud(np.fft.fftshift(pred))

    if mrs.baseline is not None:
        B = np.flipud(np.fft.fftshift(mrs.baseline))

    first = np.argmin(np.abs(axis - ppmlim[0]))
    last = np.argmin(np.abs(axis - ppmlim[1]))
    if first > last:
        first, last = last, first

    plt.figure(figsize=(9, 10))
    plt.plot(axis[first:last], spec[first:last])
    plt.gca().invert_xaxis()
    plt.plot(axis[first:last], pred[first:last], 'r')
    if mrs.baseline is not None:
        plt.plot(axis[first:last], B[first:last], 'k')

    # style stuff
    plt.minorticks_on()
    plt.grid(b=True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
    plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

    return plt.gcf()


def plot_waterfall(mrs, ppmlim=(0.4, 4.2), proj='real', mod=True):
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
    fig = plt.figure(figsize=(5, 10))

    for i in range(mrs.numBasis):
        plt.subplot(gs[i])
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().set_ylabel(mrs.names[i], rotation='horizontal')
        plt.box(False)

        if mod and mrs.con is not None:
            data = FID2Spec(mrs.con[i] * mrs.basis[:, i])
        else:
            data = FID2Spec(mrs.basis[:, i])
        plt.plot(mrs.getAxes(), data_proj(data, proj), c='r', linewidth=1, linestyle='-')

    return fig


def plot_spectrum(mrs, ppmlim=(0.0, 4.5), FID=None, proj='real', c='k'):
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

    ppmAxisShift = mrs.getAxes(ppmlim=ppmlim)

    def axes_style(plt, ppmlim, label=None, xticks=None):
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.xlabel(label)
        plt.gca().set_xticks(xticks)
        plt.minorticks_on()
        plt.grid(b=True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
        plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

    def doPlot(data, c='b', linewidth=1, linestyle='-', xticks=None):
        plt.plot(ppmAxisShift, data, color=c, linewidth=linewidth, linestyle=linestyle)
        axes_style(plt, ppmlim, label='Chemical shift (ppm)', xticks=xticks)

    # Prepare data for plotting
    if FID is not None:
        first, last = mrs.ppmlim_to_range(ppmlim)
        data = FIDToSpec(FID)[first:last]
    else:
        data = mrs.get_spec(ppmlim=ppmlim)

    # m = min(np.real(data))
    # M = max(np.real(data))
    # ylim   = (m-np.abs(M)/10,M+np.abs(M)/10)
    # plt.ylim(ylim)

    # Create the figure
    # plt.figure(figsize=(7,7))
    # Some nicer x ticks on the plots
    if np.abs(ppmlim[1] - ppmlim[0]) > 2:
        xticks = np.arange(np.ceil(ppmlim[0]), np.floor(ppmlim[1]) + 0.1, 1.0)
    else:
        xticks = np.arange(np.around(ppmlim[0], 1), np.around(ppmlim[1], 1) + 0.01, 0.1)

    doPlot(data_proj(data, proj), c=c, linewidth=2, xticks=xticks)

    plt.tight_layout()
    return plt.gcf()


def data_proj(x, proj):
    """Proj is one of 'real', 'imag', 'abs', or 'angle'"""
    if proj == 'real':
        return np.real(x)
    if proj == 'imag':
        return np.imag(x)
    if proj == 'abs':
        return np.abs(x)
    if proj == 'real':
        return np.angle(x)
    else:
        raise ValueError("proj should be one of 'real', 'imag', 'abs', or 'angle'.")


def plot_fid(mrs, tlim=None, FID=None, proj='real', c='k'):
    ''' Plot time domain FID'''

    time_axis = mrs.getAxes(axis='time')

    if FID is not None:
        data = FID
    else:
        data = mrs.FID

    data = getattr(np, proj)(data)

    plt.plot(time_axis, data, color=c, linewidth=2)

    if tlim is not None:
        plt.xlim(tlim)
    plt.xlabel('Time (s)')
    plt.minorticks_on()
    plt.grid(b=True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
    plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

    plt.tight_layout()
    return plt.gcf()


def plot_mrs_basis(mrs, plot_spec=False, ppmlim=(0.0, 4.5)):
    """Plot the formatted basis and optionally the FID from an mrs object

    :param mrs: MRS object
    :type mrs: fsl_mrs.core.mrs.MRS
    :param plot_spec: If True plot the spectrum on same axes, defaults to False
    :type plot_spec: bool, optional
    :param ppmlim: Chemical shift plotting range, defaults to (0.0, 4.5)
    :type ppmlim: tuple, optional
    :return: Figure object
    """
    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim)

    for idx, n in enumerate(mrs.names):
        plt.plot(mrs.getAxes(ppmlim=ppmlim),
                 np.real(FID2Spec(mrs.basis[:, idx]))[first:last],
                 label=n)

    if plot_spec:
        plt.plot(mrs.getAxes(ppmlim=ppmlim),
                 np.real(mrs.get_spec(ppmlim=ppmlim)),
                 'k', label='Data')

    plt.gca().invert_xaxis()
    plt.xlabel('Chemical shift (ppm)')
    plt.legend()

    return plt.gcf()


def plot_basis(basis, ppmlim=(0.0, 4.5), shift=True, conjugate=False):
    """Plot the basis contained in a Basis object

    :param basis: Basis object
    :type basis: fsl_mrs.core.basis.Basis
    :param ppmlim: Chemical shift plotting limits on x axis, defaults to (0.0, 4.5)
    :type ppmlim: tuple, optional
    :param shift: Apply chemical shift referencing shift, defaults to True.
    :type shift: Bool, optional
    :param conjugate: Apply conjugation (flips frequency direction), defaults to False.
    :type conjugate: Bool, optional
    :return: Figure object
    """
    if shift:
        axis = basis.original_ppm_shift_axis
    else:
        axis = basis.original_ppm_axis
    first, last = limit_to_range(axis, ppmlim)

    n_met = basis.n_metabs
    if n_met <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_met))
    elif n_met <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_met))
    elif n_met > 20:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_met))

    ax = plt.gca()
    ax.set_prop_cycle('color', colors)

    for idx, n in enumerate(basis.names):
        FID = basis.original_basis_array[:, idx]
        if conjugate:
            FID = FID.conj()
        ax.plot(axis[first:last],
                np.real(FID2Spec(FID))[first:last],
                label=n)

    plt.gca().invert_xaxis()
    plt.xlabel('Chemical shift (ppm)')
    plt.legend()

    return plt.gcf()


def plot_spectra(MRSList, ppmlim=(0, 4.5), single_FID=None, plot_avg=True):

    plt.figure(figsize=(10, 10))
    plt.xlim(ppmlim)
    plt.gca().invert_xaxis()
    plt.minorticks_on()
    plt.grid(b=True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
    plt.grid(b=True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

    plt.autoscale(enable=True, axis='y', tight=True)

    avg = 0
    for mrs in MRSList:
        data = np.real(mrs.get_spec(ppmlim=ppmlim))
        ppmAxisShift = mrs.getAxes(ppmlim=ppmlim)
        avg += data
        plt.plot(ppmAxisShift, data, color='k', linewidth=.5, linestyle='-')
    if single_FID is not None:
        data = np.real(single_FID.get_spec(ppmlim=ppmlim))
        plt.plot(ppmAxisShift, data, color='r', linewidth=2, linestyle='-')
    if plot_avg:
        avg /= len(MRSList)
        plt.plot(ppmAxisShift, avg, color='g', linewidth=2, linestyle='-')

    autoscale_y(plt.gca(), margin=0.05)

    return plt.gcf()


def autoscale_y(ax, margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        hi, lo = ax.get_xlim()  # Reversed
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot:
            bot = new_bot
        if new_top > top:
            top = new_top

    ax.set_ylim(bot, top)


def plot_fit_pretty(mrs, pred=None, ppmlim=(0.40, 4.2), proj='real'):
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

    data = np.real(FID2Spec(mrs.FID))
    pred = np.real(FID2Spec(pred))
    err = data - pred
    x = mrs.getAxes()

    fig = tools.make_subplots(rows=2,
                              row_width=[10, 1],
                              shared_xaxes=True,
                              print_grid=False,
                              vertical_spacing=0)

    trace_data = go.Scatter(x=x, y=data, name='data', hoverinfo="none")
    trace_pred = go.Scatter(x=x, y=pred, name='pred', hoverinfo="none")
    trace_err = go.Scatter(x=x, y=err, name='error', hoverinfo="none")

    fig.append_trace(trace_err, 1, 1)
    fig.append_trace(trace_data, 2, 1)
    fig.append_trace(trace_pred, 2, 1)

    fig['layout'].update(autosize=True,
                         title=None,
                         showlegend=True,
                         margin={'t': 0.01, 'r': 0, 'l': 20})

    fig['layout']['xaxis'].update(zeroline=False,
                                  title='Chemical shift (ppm)',
                                  automargin=True,
                                  range=[ppmlim[1], ppmlim[0]])
    fig['layout']['yaxis'].update(zeroline=False, automargin=True)
    fig['layout']['yaxis2'].update(zeroline=False, automargin=True)

    return fig


# plotly imports

def plotly_fit(mrs, res, ppmlim=(.2, 4.2), proj='real', metabs=None, phs=(0, 0)):
    """
         plot model fitting plus baseline

    Parameters:
         mrs    : MRS object
         res    : ResFit Object
         ppmlim : tuple
         metabs : list of metabolite to include in pred
         phs    : display phasing in degrees and seconds

    Returns
         fig
     """
    def project(x, proj):
        if proj == 'real':
            return np.real(x)
        elif proj == 'imag':
            return np.imag(x)
        elif proj == 'angle':
            return np.angle(x)
        else:
            return np.abs(x)

    # Prepare the data
    base = FID2Spec(res.baseline)
    axis = mrs.getAxes()
    data = FID2Spec(mrs.FID)

    if ppmlim is None:
        ppmlim = res.ppmlim

    if metabs is not None:
        preds = []
        for m in metabs:
            preds.append(FID2Spec(pred(mrs, res, m, add_baseline=False)))
        preds = sum(preds)
        preds += FID2Spec(res.baseline)
        resid = data - preds
    else:
        preds = FID2Spec(res.pred)
        resid = FID2Spec(res.residuals)

    # phasing
    faxis = mrs.getAxes(axis='freq')
    phaseTerm = np.exp(1j * (phs[0] * np.pi / 180)) * np.exp(1j * 2 * np.pi * phs[1] * faxis)

    base *= phaseTerm
    data *= phaseTerm
    preds *= phaseTerm
    resid *= phaseTerm

    base = project(base, proj)
    data = project(data, proj)
    preds = project(preds, proj)
    resid = project(resid, proj)

    # y-axis range
    minval = min(np.min(base), np.min(data), np.min(preds), np.min(resid))
    maxval = max(np.max(base), np.max(data), np.max(preds), np.max(resid))
    ymin = minval - minval / 2
    ymax = maxval + maxval / 30

    # Build the plot

    # Table

    df = pd.DataFrame()
    df['Metab'] = res.metabs
    if res.concScalings['molality'] is not None:
        df['mMol/kg'] = np.round(res.getConc(scaling='molality'), decimals=2)
        df['CRLB'] = np.round(res.getUncertainties(type='molality'), decimals=2)
    else:
        df['unscaled'] = np.round(res.getConc(), decimals=2)
        df['CRLB'] = np.round(res.getUncertainties(type='raw'), decimals=3)
    df['%CRLB'] = np.round(res.getUncertainties(), decimals=1)
    if res.concScalings['internal'] is not None:
        concstr = f'/{res.concScalings["internalRef"]}'
        df[concstr] = np.round(res.getConc(scaling='internal'), decimals=2)

    tab = create_table(df)

    colors = dict(data='rgb(67,67,67)',
                  pred='rgb(253,59,59)',
                  base='rgb(0,150,242)',
                  resid='rgb(170,170,170)')
    line_size = dict(data=1,
                     pred=2,
                     base=1, resid=1)

    trace1 = go.Scatter(x=axis, y=data,
                        mode='lines',
                        name='data',
                        line=dict(color=colors['data'], width=line_size['data']),
                        )
    trace2 = go.Scatter(x=axis, y=preds,
                        mode='lines',
                        name='model',
                        line=dict(color=colors['pred'], width=line_size['pred']),
                        )
    trace3 = go.Scatter(x=axis, y=base,
                        mode='lines',
                        name='baseline',
                        line=dict(color=colors['base'], width=line_size['base']),
                        )
    trace4 = go.Scatter(x=axis, y=resid,
                        mode='lines',
                        name='residuals',
                        line=dict(color=colors['resid'], width=line_size['resid']),
                        )

    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.4, 0.6],
                        horizontal_spacing=0.03,
                        specs=[[{'type': 'table'}, {'type': 'scatter'}]])

    fig.add_trace(tab, row=1, col=1)
    fig.add_trace(trace1, row=1, col=2)
    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace3, row=1, col=2)
    fig.add_trace(trace4, row=1, col=2)

    fig.update_layout(template='plotly_white')

    fig.update_xaxes({'domain': [0.4, 1.]}, row=1, col=2)
    fig.update_xaxes(title_text='Chemical shift (ppm)',
                     tick0=2, dtick=.5,
                     range=[ppmlim[1], ppmlim[0]])

    fig.update_yaxes(zeroline=True,
                     zerolinewidth=1,
                     zerolinecolor='Gray',
                     showgrid=False, showticklabels=False,
                     range=[ymin, ymax])

    fig.layout.update({'height': 800})

    return fig


def plot_dist_approx(res, refname='Cr'):

    numOrigMetabs = len(res.original_metabs)
    n = int(np.ceil(np.sqrt(numOrigMetabs)))
    fig = make_subplots(rows=n, cols=n, subplot_titles=res.original_metabs)
    if refname is not None:
        ref = res.getConc()[res.metabs.index(refname)]
    else:
        ref = 1.0

    for i, metab in enumerate(res.original_metabs):
        (r, c) = divmod(i, n)
        mu = res.params[i] / ref
        sig = np.sqrt(res.crlb[i]) / ref
        x = np.linspace(mu - mu, mu + mu, 100)
        N = np.exp(-(x - mu)**2 / sig**2)
        N = N / N.sum() / (x[1] - x[0])
        t = go.Scatter(x=x, y=N, mode='lines',
                       name=metab, line=dict(color='black'))
        fig.add_trace(t, row=r + 1, col=c + 1)

    fig.update_layout(template='plotly_white',
                      showlegend=False,
                      font=dict(size=10),
                      title='Approximate marginal distributions (ref={})'.format(refname),
                      height=700, width=700)
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=10, color='#ff0000')
    fig.update_layout(autosize=True)

    return fig


def plot_corr(res, corr=None, title='Correlation'):

    # Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,
    # Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,
    # Electric,Viridis,Cividis.
    # n = mrs.numBasis
    fig = go.Figure()
    if corr is None:
        corr = res.mcmc_cor
    np.fill_diagonal(corr, np.nan)
    corrabs = np.abs(corr)

    fig.add_trace(go.Heatmap(z=corr,
                             x=res.original_metabs, y=res.original_metabs, colorscale='Picnic', zmid=0))

    fig.update_layout(template='plotly_white',
                      font=dict(size=10),
                      title=title,
                      width=600,
                      height=600,
                      yaxis=dict(
                          scaleanchor="x",
                          scaleratio=1,
                      ),
                      updatemenus=[
                          dict(
                              type="buttons",
                              direction="left",
                              buttons=list([
                                  dict(
                                      args=[{"z": [corr], "colorscale":'Picnic'}],
                                      label="Real",
                                      method="restyle"
                                  ),
                                  dict(
                                      args=[{"z": [corrabs], "colorscale":'Picnic'}],
                                      label="Abs",
                                      method="restyle"
                                  )
                              ]),
                              pad={"r": 10, "t": 10},
                              showactive=True,
                              x=0.11,
                              xanchor="left",
                              y=1.1,
                              yanchor="top"
                          ),
                      ])
    fig.update_layout(autosize=True)

    return fig


def plot_dist_mcmc(res, refname='Cr'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = int(np.ceil(np.sqrt(res.numMetabs)))
    fig = make_subplots(rows=n, cols=n, subplot_titles=res.metabs)
    if refname is not None:
        ref = res.fitResults[refname].mean()
    else:
        ref = 1.0

    for i, metab in enumerate(res.metabs):
        (r, c) = divmod(i, n)
        x = res.fitResults[metab].to_numpy() / ref
        t = go.Histogram(x=x,
                         name=metab,
                         histnorm='percent', marker_color='#330C73', opacity=0.75)

        fig.add_trace(t, row=r + 1, col=c + 1)

    fig.update_layout(template='plotly_white',
                      showlegend=False,
                      width=700,
                      height=700,
                      font=dict(size=10),
                      title='MCMC marginal distributions (ref={})'.format(refname))
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=10, color='#ff0000')

    fig.update_layout(autosize=True)

    return fig


def plot_real_imag(mrs, res, ppmlim=(.2, 4.2)):
    """
         plot model fitting plus baseline

    Parameters:
         mrs    : MRS object
         res    : ResFit Object
         ppmlim : tuple

    Returns
         fig
     """
    def project(x, proj):
        if proj == 'real':
            return np.real(x)
        elif proj == 'imag':
            return np.imag(x)
        elif proj == 'angle':
            return np.angle(x)
        else:
            return np.abs(x)

    # Prepare the data
    axis = mrs.getAxes()
    data_real = project(FID2Spec(mrs.FID), 'real')
    pred_real = project(FID2Spec(res.pred), 'real')
    data_imag = project(FID2Spec(mrs.FID), 'imag')
    pred_imag = project(FID2Spec(res.pred), 'imag')

    # Build the plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Real', 'Imag'])

    colors = dict(data='rgb(67,67,67)',
                  pred='rgb(253,59,59)',
                  base='rgb(170,170,170)')
    line_size = dict(data=1,
                     pred=2,
                     base=1)

    trace1 = go.Scatter(x=axis, y=data_real,
                        mode='lines',
                        name='data : real',
                        line=dict(color=colors['data'], width=line_size['data']))
    trace2 = go.Scatter(x=axis, y=pred_real,
                        mode='lines',
                        name='model : real',
                        line=dict(color=colors['pred'], width=line_size['pred']))
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)

    trace1 = go.Scatter(x=axis, y=data_imag,
                        mode='lines',
                        name='data : imag',
                        line=dict(color=colors['data'], width=line_size['data']))
    trace2 = go.Scatter(x=axis, y=pred_imag,
                        mode='lines',
                        name='model : imag',
                        line=dict(color=colors['pred'], width=line_size['pred']))
    fig.add_trace(trace1, row=1, col=2)
    fig.add_trace(trace2, row=1, col=2)

#     fig.layout.xaxis.update({'domain': [0, .35]})
#     fig.layout.xaxis2.update({'domain': [0.4, 1.]})
    fig.layout.xaxis.update(title_text='Chemical shift (ppm)',
                            tick0=2, dtick=.5,
                            range=[ppmlim[1], ppmlim[0]])
    fig.layout.xaxis2.update(title_text='Chemical shift (ppm)',
                             tick0=2, dtick=.5,
                             range=[ppmlim[1], ppmlim[0]])

    fig.layout.yaxis2.update(zeroline=True,
                             zerolinewidth=1,
                             zerolinecolor='Gray',
                             showgrid=False, showticklabels=False)
    fig.layout.yaxis.update(zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='Gray',
                            showgrid=False, showticklabels=False)

    # Update the margins to add a title and see graph x-labels.
    # fig.layout.margin.update({'t':50, 'b':100})
    # fig.layout.update({'title': 'Fitting summary Real/Imag'})
    fig.update_layout(template='plotly_white')
    # fig.layout.update({'height':800,'width':1000})

    return fig


def pred(mrs, res, metab, add_baseline=True):
    from fsl_mrs.utils import models

    if res.model == 'lorentzian':
        forward = models.FSLModel_forward      # forward model

        con, gamma, eps, phi0, phi1, b = models.FSLModel_x2param(res.params, mrs.numBasis, res.g)
        c = con[mrs.names.index(metab)].copy()
        con = 0 * con
        con[mrs.names.index(metab)] = c
        x = models.FSLModel_param2x(con, gamma, eps, phi0, phi1, b)

    elif res.model == 'voigt':
        forward = models.FSLModel_forward_Voigt  # forward model

        con, gamma, sigma, eps, phi0, phi1, b = models.FSLModel_x2param_Voigt(res.params, mrs.numBasis, res.g)
        c = con[mrs.names.index(metab)].copy()
        con = 0 * con
        con[mrs.names.index(metab)] = c
        x = models.FSLModel_param2x_Voigt(con, gamma, sigma, eps, phi0, phi1, b)
    else:
        raise Exception('Unknown model.')

    if add_baseline:
        pred = forward(x, mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis, res.base_poly, res.metab_groups, res.g)
    else:
        pred = forward(x, mrs.frequencyAxis,
                       mrs.timeAxis,
                       mrs.basis, np.zeros(res.base_poly.shape), res.metab_groups, res.g)
    pred = SpecToFID(pred)  # predict FID not Spec
    return pred


def plot_indiv_stacked(mrs, res, ppmlim=(.2, 4.2)):

    colors = dict(data='rgb(67,67,67)',
                  indiv='rgb(253,59,59)')
    line_size = dict(data=.5,
                     indiv=2)
    fig = go.Figure()
    axis = mrs.getAxes()
    y_data = np.real(FID2Spec(mrs.FID))
    trace1 = go.Scatter(x=axis, y=y_data,
                        mode='lines',
                        name='data',
                        line=dict(color=colors['data'], width=line_size['data']))
    fig.add_trace(trace1)

    for i, metab in enumerate(mrs.names):
        y_fit = np.real(FID2Spec(pred(mrs, res, metab)))
        trace2 = go.Scatter(x=axis, y=y_fit,
                            mode='lines',
                            name=metab,
                            line=dict(color=colors['indiv'], width=line_size['indiv']))
        fig.add_trace(trace2)

    fig.layout.xaxis.update(title_text='Chemical shift (ppm)',
                            tick0=2, dtick=.5,
                            range=[ppmlim[1], ppmlim[0]])
    fig.layout.yaxis.update(zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='Gray',
                            showgrid=False, showticklabels=False)

    # Update the margins to add a title and see graph x-labels.
    # fig.layout.margin.update({'t':50, 'b':100})
    # fig.layout.update({'title': 'Individual Fitting summary'})
    fig.update_layout(template='plotly_white')
    # fig.layout.update({'height':800,'width':1000})

    return fig


def plot_indiv(mrs, res, ppmlim=(.2, 4.2)):

    colors = dict(data='rgb(67,67,67)',
                  pred='rgb(253,59,59)')
    line_size = dict(data=.5,
                     pred=2)

    ncols = 3
    nrows = int(np.ceil(mrs.numBasis / ncols))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=mrs.names)
    axis = mrs.getAxes()
    for i, metab in enumerate(mrs.names):
        c, r = i % ncols, i // ncols
        y_data = np.real(FID2Spec(mrs.FID))
        y_fit = np.real(FID2Spec(res.predictedFID(mrs, mode=metab, noBaseline=True)))
        # y_fit   = np.real(FID2Spec(pred(mrs,res,metab)))

        trace1 = go.Scatter(x=axis, y=y_data,
                            mode='lines',
                            line=dict(color=colors['data'], width=line_size['data']))
        trace2 = go.Scatter(x=axis, y=y_fit,
                            mode='lines',
                            line=dict(color=colors['pred'], width=line_size['pred']))
        fig.add_trace(trace1, row=r + 1, col=c + 1)
        fig.add_trace(trace2, row=r + 1, col=c + 1)

        fig.update_layout(template='plotly_white',
                          showlegend=False,
                          #   width = 1500,
                          height=1000,
                          font=dict(size=10),
                          title='Individual fits')
        for j in fig['layout']['annotations']:
            j['font'] = dict(size=10, color='#ff0000')

        if i == 0:
            xax = eval("fig.layout.xaxis")
            yax = eval("fig.layout.yaxis")
        else:
            xax = eval("fig.layout.xaxis{}".format(i + 1))
            yax = eval("fig.layout.yaxis{}".format(i + 1))
        xax.update(tick0=2, dtick=.5, range=[ppmlim[1], ppmlim[0]], showticklabels=False)
        yax.update(zeroline=True, zerolinewidth=1, zerolinecolor='Gray',
                   showgrid=False, showticklabels=False)
    return fig


def plot_references(mrs, res):
    """Generate figure showing the areas and metabolites used to estimate the water scaling.

    :param mrs: MRS object with unsuppressed water reference data.
    :type mrs: fsl_mrs.core.mrs.MRS
    :param res: FSL-MRS results object.
    :type res: fsl_mrs.utils.results.FitRes
    :return: Figure
    :rtype: go.Figure
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Unsuppressed', 'Suppressed'])

    quant_info = res.concScalings['quant_info']
    min_val = np.min([res.ppmlim[0], quant_info.ref_limits[0], quant_info.h2o_limits[0]])
    max_val = np.max([res.ppmlim[1], quant_info.ref_limits[1], quant_info.h2o_limits[1]])
    data_range = (min_val, max_val)
    axis = mrs.getAxes(ppmlim=data_range)
    first, last = mrs.ppmlim_to_range(ppmlim=data_range)
    water_first, water_last = res.concScalings['ref_info']['water_ref'].limits
    water_axis = mrs.getAxes()[water_first:water_last]
    metab_first, metab_last = res.concScalings['ref_info']['metab_ref'].limits
    metab_axis = mrs.getAxes()[metab_first:metab_last]

    y_data = np.real(FIDToSpec(mrs.H2O))[first:last]
    trace1 = go.Scatter(x=axis, y=y_data,
                        mode='lines',
                        name='data',
                        line=dict(color='rgb(0,0,0)', width=1))
    y_data = np.real(FIDToSpec(res.concScalings['ref_info']['water_ref'].fid))[water_first:water_last]
    trace2 = go.Scatter(x=water_axis, y=y_data,
                        mode='lines',
                        name='Fitted, integrated water',
                        fill='tozeroy',
                        line=dict(color='rgb(255,0,0)', width=1))
    fig.add_trace(trace1, 1, 1)
    fig.add_trace(trace2, 1, 1)

    y_data = np.real(FIDToSpec(mrs.FID))[first:last]
    trace3 = go.Scatter(x=axis, y=y_data,
                        mode='lines',
                        name='data',
                        line=dict(color='rgb(0,0,0)', width=1))
    y_data = np.real(FIDToSpec(res.concScalings['ref_info']['metab_ref'].original_fid))[first:last]
    trace4 = go.Scatter(x=axis, y=y_data,
                        mode='lines',
                        name='Fitted Reference',
                        line=dict(color='rgb(0,0,255)', width=1))
    y_data = np.real(FIDToSpec(res.concScalings['ref_info']['metab_ref'].fid))[metab_first:metab_last]
    trace5 = go.Scatter(x=metab_axis, y=y_data,
                        mode='lines',
                        name='Metabolite integrated',
                        fill='tozeroy',
                        line=dict(color='rgb(255,0,0)', width=1))
    fig.add_trace(trace3, 1, 2)
    fig.add_trace(trace4, 1, 2)
    fig.add_trace(trace5, 1, 2)

    fig.layout.xaxis.update(title_text='Chemical shift (ppm)',
                            tick0=2, dtick=.5,
                            range=[max_val, min_val])
    fig.layout.xaxis2.update(title_text='Chemical shift (ppm)',
                             tick0=2, dtick=.5,
                             range=[max_val, min_val])

    fig.layout.yaxis2.update(zeroline=True,
                             zerolinewidth=1,
                             zerolinecolor='Gray',
                             showgrid=False)
    fig.layout.yaxis.update(zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='Gray',
                            showgrid=False)
    return fig


def create_table(df):
    """
    Generate plotly graphical Table from pandas dataframe
    """
    n = df.shape[0]
    colors = ['#F2F2F2', 'white'] * n  # alternating row colors

    header = dict(values=['<b>' + x + '</b>' for x in list(df.columns)],
                  fill_color='#41476C',
                  align='left',
                  font={'color': 'white'})

    cells = dict(values=[df[x] for x in list(df.columns)],
                 fill_color=[colors * 2],
                 align='left')

    tab = go.Table(header=header,
                   cells=cells, visible=True)

    return tab


def plot_table_lineshape(res):
    """
    Creates a table summarising fitted lineshape parameters for each metab group
    """

    shift = res.getShiftParams(units='ppm')
    lw = res.getLineShapeParams(units='Hz')[0]  # Only take combined values

    df = pd.DataFrame()
    header = ['Metab group', 'linewidth (Hz)', 'shift (ppm)']
    values = [[], [], []]
    for g in range(res.g):
        values[1].append(np.round(lw[g], decimals=3))
        values[2].append(np.round(shift[g], decimals=5))
        metabs = []
        for i, m in enumerate(res.original_metabs):
            if res.metab_groups[i] == g:
                metabs.append(m)
        values[0].append(', '.join(metabs))

    for h, v in zip(header, values):
        df[h] = v

    tab = create_table(df)
    fig = go.Figure(data=[tab])
    fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))

    return fig


def plot_table_phase(res):
    """
    Creates a table summarising the fitted phase parameters
    """
    p0, p1 = res.getPhaseParams(phi0='degrees', phi1='deg_per_ppm')

    df = pd.DataFrame()
    df['Static phase (deg)'] = [np.round(p0, decimals=5)]
    df['Linear phase (deg/ppm)'] = [np.round(p1, decimals=5)]

    tab = create_table(df)
    fig = go.Figure(data=[tab])
    fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))

    return fig


def plot_table_lineshape_phase(res):
    shift = res.getShiftParams(units='ppm')
    lw = res.getLineShapeParams(units='Hz')[0]  # Only take combined values

    # Get the lineshape params
    header = ['Metab group', 'linewidth (Hz)', 'shift (ppm)']
    values = [[], [], []]
    for g in range(res.g):
        values[1].append(np.round(lw[g], decimals=3))
        values[2].append(np.round(shift[g], decimals=5))
        metabs = []
        for i, m in enumerate(res.original_metabs):
            if res.metab_groups[i] == g:
                metabs.append(m)
        values[0].append(', '.join(metabs))
    # Fill dataframe
    df1 = pd.DataFrame()
    for h, v in zip(header, values):
        df1[h] = v
    # create table
    tab1 = create_table(df1)

    # Get phase params
    p0, p1 = res.getPhaseParams(phi0='degrees', phi1='deg_per_ppm')
    # Fill dataframe
    df2 = pd.DataFrame()
    df2['Static phase (deg)'] = [np.round(p0, decimals=5)]
    df2['Linear phase (deg/ppm)'] = [np.round(p1, decimals=5)]
    # create table
    tab2 = create_table(df2)

    column_widths = [df1.shape[1], df2.shape[1]]
    fig = make_subplots(rows=1, cols=2,
                        column_widths=column_widths,
                        horizontal_spacing=0.03,
                        specs=[[{'type': 'table'}, {'type': 'table'}]],
                        subplot_titles=['Lineshape params', 'Phase params'])

    fig.add_trace(tab1, row=1, col=1)
    fig.add_trace(tab2, row=1, col=2)

    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))

    return fig


def plot_table_qc(res):
    # Peak by peak snr and fwhm

    snr, fwhm = res.getQCParams()
    df = pd.DataFrame()

    df['Metab'] = res.original_metabs
    df['SNR'] = snr.to_numpy()
    df['FWHM (Hz)'] = fwhm.to_numpy()
    df['SNR'] = df['SNR'].map('{:.1f}'.format)
    df['FWHM (Hz)'] = df['FWHM (Hz)'].map('{:.1f}'.format)

    tab = create_table(df)
    fig = go.Figure(data=[tab])
    fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))

    return fig


# ----------- Dyn MRS
# Visualisation
def plotly_dynMRS(mrs_list, time_var, ppmlim=(.2, 4.2)):
    """
    Plot dynamic MRS data with a slider though time
    Args:
        mrs_list: list of MRS objects
        time_var: list of time variable (or bvals for dMRS)
        ppmlim: list

    Returns:
        plotly Figure
    """
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for i, t in enumerate(time_var):
        x = mrs_list[i].getAxes()
        y = np.real(FIDToSpec(mrs_list[i].FID))
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="black", width=3),
                name=f"{t}",
                x=x,
                y=y))
    fig.update_layout(template='plotly_white')
    fig.update_xaxes(title_text='Chemical shift (ppm)',
                     tick0=2, dtick=.5,
                     range=[ppmlim[1], ppmlim[0]])

    # y-axis range
    data = [np.real(FIDToSpec(mrs.FID)) for mrs in mrs_list]
    data = np.asarray(data).flatten()
    minval = np.min(data)
    maxval = np.max(data)
    ymin = minval - minval / 2
    ymax = maxval + maxval / 30

    fig.update_yaxes(zeroline=True,
                     zerolinewidth=1,
                     zerolinecolor='Gray',
                     showgrid=False, showticklabels=False,
                     range=[ymin, ymax])

    # Make 0th trace visible
    fig.data[0].visible = True
    # Create and add slider
    steps = []
    for i in range(len(time_var)):
        step = dict(
            method="restyle",
            label=f"t={time_var[i]}",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"time_variable : {time_var[i]}"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "time variable: "},
        pad={"t": 50},
        steps=steps)]

    fig.update_layout(sliders=sliders)
    fig.layout.update({'height': 800})

    return fig


# ----------- Imaging
# helper functions
def ijk2xyz(ijk, affine):
    """ Return X, Y, Z coordinates for i, j, k """
    ijk = np.asarray(ijk)
    return affine[:3, :3].dot(ijk.T).T + affine[:3, 3]


def xyz2ijk(xyz, affine):
    """ Return i, j, k coordinates for X, Y, Z """
    xyz = np.asarray(xyz)
    inv_affine = np.linalg.inv(affine)
    return inv_affine[:3, :3].dot(xyz.T).T + inv_affine[:3, 3]


def do_plot_slice(slice, rect):
    vmin = np.quantile(slice, .01)
    vmax = np.quantile(slice, .99)
    plt.imshow(slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.plot(rect[:, 0], rect[:, 1], c='#FF4646', linewidth=2)
    plt.xticks([])
    plt.yticks([])


def plot_voxel_orient(t1file, voxfile):
    """
    Plot T1 centered on voxel
    Overlay voxel in red
    Plots in voxel coordinates
    """
    t1 = nib.load(t1file)
    vox = nib.load(voxfile)
    t1_data = t1.get_fdata()

    # centre of MRS voxel in T1 voxel space (or is it the corner?)
    #
    # PM: Nope, it's the voxel centre - this is mandated by the NIFTI spec
    #
    ijk = xyz2ijk(ijk2xyz([0, 0, 0], vox.affine), t1.affine)
    i, j, k = ijk

    # half size of MRS voxel (careful this assumes 1mm resolution T1)
    si, sj, sk = np.array(vox.header.get_zooms())[:3] / 2
    # Do the plotting
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    slice = np.squeeze(t1_data[int(i), :, :]).T
    rect = np.asarray([[j - sj, k - sk],
                       [j + sj, k - sk],
                       [j + sj, k + sk],
                       [j - sj, k + sk],
                       [j - sj, k - sk]])
    do_plot_slice(slice, rect)
    plt.subplot(1, 3, 2)
    slice = np.squeeze(t1_data[:, int(j), :]).T
    rect = np.asarray([[i - si, k - sk],
                       [i + si, k - sk],
                       [i + si, k + sk],
                       [i - si, k + sk],
                       [i - si, k - sk]])
    do_plot_slice(slice, rect)
    plt.subplot(1, 3, 3)
    slice = np.squeeze(t1_data[:, :, int(k)]).T
    rect = np.asarray([[i - si, j - sj],
                       [i + si, j - sj],
                       [i + si, j + sj],
                       [i - si, j + sj],
                       [i - si, j - sj]])
    do_plot_slice(slice, rect)

    return plt.gcf()


def plot_world_orient(t1file, voxfile):
    """
    Plot sagittal/coronal/axial T1 centered on voxel
    Overlay voxel in red
    Plots in world coordinates with the 'MNI' convention
    """
    t1 = nib.load(t1file)
    vox = mrs_io.read_FID(voxfile)
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
    extents = zip((0, 0, 0), t1_data.shape)
    extents = np.asarray(list(it.product(*extents)))
    extents = ijk2xyz(extents, t1.affine)
    offset = extents.min(axis=0)
    offaff = np.eye(4)
    offaff[:3, 3] = -offset
    shape = (extents.max(axis=0) - offset).astype(np.int)[:3]

    t1_data = ndimage.affine_transform(t1_data,
                                       np.dot(offaff, t1.affine),
                                       output_shape=shape,
                                       order=3,
                                       mode='constant',
                                       cval=0)

    # centre of MRS voxel in (transformed) T1 voxel space
    ijk = xyz2ijk(ijk2xyz([0, 0, 0], vox.voxToWorldMat), np.linalg.inv(offaff))
    i, j, k = ijk

    si, sj, sk = np.array(vox.header.get_zooms())[:3] / 2
    # Do the plotting
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    slice = np.squeeze(t1_data[int(i), :, :]).T
    rect = np.asarray([[j - sj, k - sk],
                       [j + sj, k - sk],
                       [j + sj, k + sk],
                       [j - sj, k + sk],
                       [j - sj, k - sk]])
    do_plot_slice(slice, rect)
    plt.subplot(1, 3, 2)
    slice = np.squeeze(t1_data[:, int(j), :]).T
    rect = np.asarray([[i - si, k - sk],
                       [i + si, k - sk],
                       [i + si, k + sk],
                       [i - si, k + sk],
                       [i - si, k - sk]])
    do_plot_slice(slice, rect)
    plt.gca().invert_xaxis()
    plt.subplot(1, 3, 3)
    slice = np.squeeze(t1_data[:, :, int(k)]).T
    rect = np.asarray([[i - si, j - sj],
                       [i + si, j - sj],
                       [i + si, j + sj],
                       [i - si, j + sj],
                       [i - si, j - sj]])
    do_plot_slice(slice, rect)
    plt.gca().invert_xaxis()

    return plt.gcf()


def plot_world_orient_ax(t1file, voxfile, ax, orientation='axial'):
    """
    Plot sagittal/coronal/axial T1 centered on voxel in axes passed
    Overlay voxel in red
    Plots in world coordinates with the 'MNI' convention
    """
    t1 = nib.load(t1file)
    vox = nib.load(voxfile)
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
    extents = zip((0, 0, 0), t1_data.shape)
    extents = np.asarray(list(it.product(*extents)))
    extents = ijk2xyz(extents, t1.affine)
    offset = extents.min(axis=0)
    offaff = np.eye(4)
    offaff[:3, 3] = -offset
    shape = (extents.max(axis=0) - offset).astype(np.int)[:3]

    t1_data = ndimage.affine_transform(t1_data,
                                       np.dot(offaff, t1.affine),
                                       output_shape=shape,
                                       order=3,
                                       mode='constant',
                                       cval=0)

    # centre of MRS voxel in (transformed) T1 voxel space
    ijk = xyz2ijk(ijk2xyz([0, 0, 0], vox.affine), np.linalg.inv(offaff))
    i, j, k = ijk

    si, sj, sk = np.array(vox.header.get_zooms())[:3] / 2
    # Do the plotting
    plt.sca(ax)
    if orientation == 'sagital':
        slice = np.squeeze(t1_data[int(i), :, :]).T
        rect = np.asarray([[j - sj, k - sk],
                           [j + sj, k - sk],
                           [j + sj, k + sk],
                           [j - sj, k + sk],
                           [j - sj, k - sk]])
        do_plot_slice(slice, rect)
    elif orientation == 'coronal':

        slice = np.squeeze(t1_data[:, int(j), :]).T
        rect = np.asarray([[i - si, k - sk],
                           [i + si, k - sk],
                           [i + si, k + sk],
                           [i - si, k + sk],
                           [i - si, k - sk]])
        do_plot_slice(slice, rect)
        ax.invert_xaxis()
    elif orientation == 'axial':
        slice = np.squeeze(t1_data[:, :, int(k)]).T
        rect = np.asarray([[i - si, j - sj],
                           [i + si, j - sj],
                           [i + si, j + sj],
                           [i - si, j + sj],
                           [i - si, j - sj]])
        do_plot_slice(slice, rect)
        ax.invert_xaxis()

    return plt.gca()
