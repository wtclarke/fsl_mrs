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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from plotly import tools

from fsl.data.image import Image
from fsl_mrs.utils import mrs_io
from fsl.transform.affine import transform
from fsl_mrs.utils.misc import FIDToSpec, limit_to_range


def FID2Spec(x):
    """
       Turn FID to spectrum for plotting
    """
    x = FIDToSpec(x)
    return x


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


def plot_fit(mrs, res, out=None, baseline=True, proj='real'):
    """Primary plotting function for FSL-MRS fits

    :param mrs: mrs object which has been fitted
    :type mrs: fsl_mrs.core.mrs.MRS
    :param res: Fitting results object
    :type res: fsl_mrs.utils.results.FitRes
    :param out: output figure filename, defaults to None
    :type out: str, optional
    :param baseline: optionally plot baseline, defaults to True
    :type baseline: bool, optional
    :param proj: 'real', 'imag', 'abs', or 'angle', defaults to 'real'
    :type proj: str, optional
    """

    def axes_style(plt, ppmlim, label=None, xticks=None):
        plt.xlim(ppmlim)
        plt.gca().invert_xaxis()
        plt.xlabel(label)
        plt.gca().set_xticks(xticks)
        plt.minorticks_on()
        plt.grid(True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
        plt.grid(True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

    def doPlot(data, c='b', linewidth=1, linestyle='-', xticks=None):
        plt.plot(mrs.getAxes(), data, color=c, linewidth=linewidth, linestyle=linestyle)
        axes_style(plt, res.ppmlim, label='Chemical shift (ppm)', xticks=xticks)

    # Prepare data for plotting
    data = FID2Spec(mrs.FID)
    pred = FID2Spec(res.pred)
    if baseline is not None:
        baseline = FID2Spec(res.baseline)

    first, last = mrs.ppmlim_to_range(ppmlim=res.ppmlim, shift=True)

    # turn to real numbers
    data = data_proj(data, proj)
    pred = data_proj(pred, proj)
    if baseline is not None:
        baseline = data_proj(baseline, proj)

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
    if mrs.nucleus == '1H':
        xticks = np.arange(res.ppmlim[0], res.ppmlim[1], .2)
    else:
        xticks = np.round(np.linspace(res.ppmlim[0], res.ppmlim[1], 10), decimals=1)
    plt.plot(mrs.getAxes(), data_proj(data - pred, proj), c='k', linewidth=1, linestyle='-')
    axes_style(plt, res.ppmlim, xticks=xticks)
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
        plt.grid(True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
        plt.grid(True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

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
    plt.grid(True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
    plt.grid(True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

    plt.tight_layout()
    return plt.gcf()


def plot_mrs_basis(mrs, plot_spec=False, ppmlim=(0.0, 4.5), normalise=False):
    """Plot the formatted basis and optionally the FID from an mrs object

    :param mrs: MRS object
    :type mrs: fsl_mrs.core.mrs.MRS
    :param plot_spec: If True plot the spectrum on same axes, defaults to False
    :type plot_spec: bool, optional
    :param ppmlim: Chemical shift plotting range, defaults to (0.0, 4.5)
    :type ppmlim: tuple, optional
    :param normalise: If True normalise spectrum to max of basis, defaults to False
    :type normalise: bool, optional
    :return: Figure object
    """
    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim)

    n_met = len(mrs.names)
    if n_met <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_met))
    elif n_met <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_met))
    elif n_met > 20:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_met))

    ax = plt.gca()
    ax.set_prop_cycle('color', colors)

    max_basis = []
    for idx, n in enumerate(mrs.names):
        toplot = np.real(FID2Spec(mrs.basis[:, idx]))[first:last]
        ax.plot(mrs.getAxes(ppmlim=ppmlim),
                toplot,
                label=n)
        max_basis.append(toplot.max())

    if plot_spec:
        spec = np.real(mrs.get_spec(ppmlim=ppmlim))
        if normalise:
            spec *= np.max(max_basis) / spec.max()
        ax.plot(mrs.getAxes(ppmlim=ppmlim),
                spec,
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
    plt.grid(True, axis='x', which='major', color='k', linestyle='--', linewidth=.3)
    plt.grid(True, axis='x', which='minor', color='k', linestyle=':', linewidth=.3)

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
def plotly_spectrum(mrs, res, ppmlim=None, proj='real', metabs=None, phs=(0, 0)):
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
            # preds.append(FID2Spec(pred(mrs, res, m, add_baseline=False)))
            preds.append(FID2Spec(res.predictedFID(mrs, mode=m, noBaseline=True)))
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

    fig = go.Figure()

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    fig.add_trace(trace4)

    fig.update_layout(template='plotly_white')

    fig.update_xaxes(title_text='Chemical shift (ppm)',
                     tick0=2, dtick=.5,
                     range=[ppmlim[1], ppmlim[0]])

    fig.update_yaxes(zeroline=True,
                     zerolinewidth=1,
                     zerolinecolor='Gray',
                     showgrid=False, showticklabels=False,
                     range=[ymin, ymax])

    fig.layout.update({'height': 600})

    return fig


def plotly_fit(mrs, res, ppmlim=None, proj='real', metabs=None, phs=(0, 0)):
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
            preds.append(FID2Spec(res.predictedFID(mrs, mode=m, noBaseline=True)))
            # preds.append(FID2Spec(pred(mrs, res, m, add_baseline=False)))
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


def plotly_avg_fit(mrs_list, res_list, ppmlim=None):

    if ppmlim is None:
        ppmlim = res_list[0].ppmlim

    all_specs = []
    all_pred = []
    for mrs, res in zip(mrs_list, res_list):
        all_specs.append(mrs.FID)
        all_pred.append(res.pred)

    from fsl_mrs.utils import preproc as proc

    fids, _, _ = proc.phase_freq_align(
        all_specs,
        mrs_list[0].bandwidth, mrs_list[0].centralFrequency)
    pred, _, _ = proc.phase_freq_align(
        all_pred,
        mrs_list[0].bandwidth, mrs_list[0].centralFrequency,
        target=np.asarray(fids).mean(axis=0))

    specs = [FID2Spec(x) for x in fids]
    avg_spec = np.asarray(specs).mean(axis=0).real
    plus1sd_spec = avg_spec + 1 * np.asarray(specs).std(axis=0).real
    minus1sd_spec = avg_spec - 1 * np.asarray(specs).std(axis=0).real
    avg_fit = FID2Spec(np.asarray(pred).mean(axis=0)).real
    axis = mrs_list[0].getAxes()

    # y-axis range
    minval = min((np.min(avg_spec), np.min(plus1sd_spec), np.min(minus1sd_spec), np.min(avg_fit)))
    maxval = max((np.max(avg_spec), np.max(plus1sd_spec), np.max(minus1sd_spec), np.max(avg_fit)))
    ymin = minval - minval / 2
    ymax = maxval + maxval / 30

    trace1 = go.Scatter(x=axis, y=avg_spec,
                        mode='lines',
                        name='Mean data',
                        line=dict(color='rgb(67,67,67)', width=1.5),
                        )

    trace2 = go.Scatter(
        x=axis.tolist() + axis.tolist()[::-1],
        y=plus1sd_spec.tolist() + minus1sd_spec.tolist()[::-1],
        fill='toself',
        mode='lines',
        name='±1SD',
        line=dict(color='rgb(100,100,100)', width=0.5))

    trace3 = go.Scatter(x=axis, y=avg_fit,
                        mode='lines',
                        name='Mean fit',
                        line=dict(color='rgb(253,59,59)', width=2),
                        )

    fig = go.Figure()
    fig.update_layout(template='plotly_white')
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)

    fig.update_xaxes(title_text='Chemical shift (ppm)',
                     tick0=2, dtick=.5,
                     range=[ppmlim[1], ppmlim[0]])

    fig.update_yaxes(zeroline=True,
                     zerolinewidth=1,
                     zerolinecolor='Gray',
                     showgrid=False, showticklabels=False,
                     range=[ymin, ymax])

    fig.layout.update({'height': 600})
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


def plot_general_corr(corr_mat, labels, title='Correlation', nan_diag=True):
    """_summary_

    _extended_summary_

    :param corr_mat: _description_
    :type corr_mat: _type_
    :param labels: _description_
    :type labels: _type_
    :param title: _description_, defaults to 'Correlation'
    :type title: str, optional
    :param nan_diag: _description_, defaults to True
    :type nan_diag: bool, optional
    :return: _description_
    :rtype: _type_
    """
    fig = go.Figure()
    if nan_diag:
        np.fill_diagonal(corr_mat, np.nan)
    corrabs = np.abs(corr_mat)

    fig.add_trace(
        go.Heatmap(
            z=corr_mat,
            x=labels,
            y=labels,
            colorscale='Picnic',
            zmid=0))

    fig.update_layout(
        template='plotly_white',
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
                        args=[{"z": [corr_mat], "colorscale": 'Picnic'}],
                        label="Real",
                        method="restyle"
                    ),
                    dict(
                        args=[{"z": [corrabs], "colorscale": 'Picnic'}],
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


def plot_corr(res, corr=None, title='Correlation'):
    """Plot the correlation matrix of fitting parameters from fit results

    :param res: FSL-MRS results object
    :type res: fsl_mrs.utils.results.FitRes
    :param corr: Optionally override correlation matrix, defaults to None
    :type corr: np.array, optional
    :param title: Plot title, defaults to 'Correlation'
    :type title: str, optional
    :return: Plotly figure
    :rtype: plotly.graph_objs.Figure
    """
    if corr is None:
        corr = res.mcmc_cor

    return plot_general_corr(
        corr,
        res.original_metabs,
        title=title)


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


def plot_real_imag(mrs, res, ppmlim=None):
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

    if ppmlim is None:
        ppmlim = mrs.default_ppm_range

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


def plot_indiv_stacked(mrs, res, ppmlim=None):

    if ppmlim is None:
        ppmlim = res.ppmlim

    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim)

    n_met = len(mrs.names)
    if n_met <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_met))
    elif n_met <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_met))
    elif n_met > 20:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_met))

    def format_c_string(colour):
        return f'rgb({colour[0] * 255},{colour[1] * 255},{colour[2] * 255})'

    colors = {met: format_c_string(color) for met, color in zip(mrs.names, colors)}
    colors.update({'data': 'rgb(67,67,67)'})

    line_size = dict(data=.5,
                     indiv=2)
    fig = go.Figure()
    axis = mrs.getAxes()[first:last]
    y_data = np.real(FID2Spec(mrs.FID))[first:last]
    trace1 = go.Scatter(x=axis, y=y_data,
                        mode='lines',
                        name='data',
                        line=dict(color=colors['data'], width=line_size['data']))
    fig.add_trace(trace1)

    for i, metab in enumerate(mrs.names):
        # y_fit = np.real(FID2Spec(pred(mrs, res, metab)))
        y_fit = np.real(FID2Spec(res.predictedFID(mrs, mode=metab)))[first:last]
        trace2 = go.Scatter(x=axis, y=y_fit,
                            mode='lines',
                            name=metab,
                            line=dict(color=colors[metab], width=line_size['indiv']))
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


def plot_indiv(mrs, res, ppmlim=None):

    if ppmlim is None:
        ppmlim = res.ppmlim

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


def plotly_basis(mrs, ppmlim=None):
    """Plotly formatted plot of data + basis spectra for fitting report

    :param mrs: MRS object with basis loaded
    :type mrs: fsl_mrs.core.mrs.MRS
    :param ppmlim: Optional ppm limits, defaults to None
    :type ppmlim: tuple, optional
    :return: Figure
    :rtype: go.Figure
    """
    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim)

    n_met = len(mrs.names)
    if n_met <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_met))
    elif n_met <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_met))
    elif n_met > 20:
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_met))

    def format_c_string(colour):
        return f'rgb({colour[0] * 255},{colour[1] * 255},{colour[2] * 255})'

    colors = {met: format_c_string(color) for met, color in zip(mrs.names, colors)}
    colors.update({'data': 'rgb(67,67,67)'})

    line_size = dict(data=.5,
                     basis=2)
    fig = go.Figure()
    axis = mrs.getAxes()[first:last]
    max_vals = []
    for idx, metab in enumerate(mrs.names):
        toplot = np.real(FID2Spec(mrs.basis[:, idx]))[first:last]
        trace2 = go.Scatter(x=axis, y=toplot,
                            mode='lines',
                            name=metab,
                            line=dict(color=colors[metab], width=line_size['basis']))
        fig.add_trace(trace2)
        max_vals.append(toplot.max())

    y_data = np.real(FID2Spec(mrs.FID))[first:last]
    y_data *= np.max(max_vals) / y_data.max()
    trace1 = go.Scatter(x=axis, y=y_data,
                        mode='lines',
                        name='data',
                        line=dict(color=colors['data'], width=line_size['data']))
    fig.add_trace(trace1)

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
        values[1].append(np.round(lw[g], decimals=5))
        if res.model == 'free_shift':
            shift_groups = []
            for i, _ in enumerate(res.original_metabs):
                if res.metab_groups[i] == g:
                    shift_groups.append(str(np.round(shift[i], decimals=3)))
            values[2].append(', '.join(shift_groups))
        else:
            values[2].append(np.round(shift[g], decimals=3))
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
    _, p1_s = res.getPhaseParams(phi0='degrees', phi1='seconds')
    p1_ms = p1_s * 1000
    p1_pts = p1_s / (1 / res.bandwidth)

    # Fill dataframe
    df2 = pd.DataFrame()
    df2['Static phase (deg)'] = [np.round(p0, decimals=5)]
    df2['Linear phase (deg/ppm)'] = [np.round(p1, decimals=5)]
    df2['Linear phase (ms)'] = [np.round(p1_ms, decimals=3)]
    df2['Linear phase (pnts)'] = [np.round(p1_pts, decimals=2)]

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
def plotly_dynMRS(mrs_list,
                  res_list=None,
                  time_var=None,
                  ppmlim=None,
                  proj='real'):
    """
    Plot dynamic MRS data with a slider though time
    Args:
        mrs_list: list of MRS objects
        res_list : list of Results objects
        time_var: list of time variable (or bvals for dwMRS)
        ppmlim: tuple (low, high) in ppm
        proj : string (one of 'real','imag','abs','angle')

    Returns:
        plotly Figure
    """
    if ppmlim is None and res_list is None:
        ppmlim = mrs_list[0].default_ppm_range
    elif ppmlim is None:
        ppmlim = res_list[0].ppmlim

    # number of dyn time points
    n = len(mrs_list)
    if time_var is None:
        time_var = np.arange(n)
    else:
        time_var = np.asarray(time_var)

    # how to represent the complex data
    proj_funcs = {'real': np.real,
                  'imag': np.imag,
                  'angle': np.angle,
                  'abs': np.abs}
    # colors (same as SVS plotting - should these move to where we define global constants?)
    colors = {'data': 'rgb(67,67,67)',
              'pred': 'rgb(253,59,59)',
              'base': 'rgb(0,150,242)',
              'resid': 'rgb(170,170,170)'}

    # data to plot
    xaxis  = mrs_list[0].getAxes()
    ydata  = {}
    ydata['data'] = [proj_funcs[proj](mrs_list[i].get_spec()) for i in range(n)]
    if res_list is not None:
        ydata['pred'] = [proj_funcs[proj](res_list[i].pred_spec) for i in range(n)]
        ydata['base'] = [proj_funcs[proj](FID2Spec(res_list[i].baseline)) for i in range(n)]
        ydata['resid'] = [proj_funcs[proj](FID2Spec(res_list[i].residuals)) for i in range(n)]
    # TODO : add residuals/baseline to the data for plotting

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for name in ydata:
        for i, t in enumerate(time_var):
            if isinstance(t, (int, float)):
                t_str = str(t)
            else:
                t_str = np.array2string(t, floatmode='fixed', precision=1)
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color=colors[name], width=1),
                    name=f"{name} - {t_str}",
                    x=xaxis,
                    y=ydata[name][i]))

    fig.update_layout(template='plotly_white')
    fig.update_xaxes(title_text='Chemical shift (ppm)',
                     tick0=2, dtick=.5,
                     range=[ppmlim[1], ppmlim[0]])

    # guess y-axis range
    data = np.asarray(ydata['data']).flatten()
    minval, maxval = np.min(data), np.max(data)
    ymin = minval - np.abs(minval) / 2
    ymax = maxval + maxval / 30

    # update yaxes
    fig.update_yaxes(zeroline=True,
                     zerolinewidth=1,
                     zerolinecolor='Gray',
                     showgrid=False, showticklabels=True,
                     range=[ymin, ymax])

    # Make first traces visible
    for i in range(len(ydata)):
        fig.data[i * n].visible = True

    # Create and add slider steps
    steps = []
    for i in range(n):
        t_str = np.array2string(time_var[i], floatmode='fixed', precision=1)
        step = dict(
            method="restyle",
            label=f"t={t_str}",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"time_variable : {t_str}"}])

        for j in range(len(ydata)):
            step["args"][0]["visible"][i + j * n] = True
        steps.append(step)

    # Create slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "time variable: "},
        pad={"t": 50},
        steps=steps)]

    fig.update_layout(sliders=sliders)
    fig.layout.update({'height': 600})

    return fig


# ----------- Imaging
def plot_world_orient(t1file, voxfile):
    """
    Plot sagittal/coronal/axial T1 with voxel overlay in red
    """
    t1img = Image(t1file)
    vox = mrs_io.read_FID(voxfile)

    # Centre of voxel
    originvox = np.zeros(3)
    centre_mm = transform(originvox, vox.voxToWorldMat)
    voxel_corners_world = [
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]]
    voxel_corners_mm = [transform(vcw, vox.voxToWorldMat) for vcw in voxel_corners_world]

    # What indicies does this correspond to in the T1 img?
    centre_vox_t1 = transform(centre_mm, t1img.worldToVoxMat)
    centre_vox_t1_int = centre_vox_t1.astype(int)
    voxel_corners_t1 = np.asarray([transform(vcm, t1img.worldToVoxMat) for vcm in voxel_corners_mm])

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    slices = [
        t1img[centre_vox_t1_int[0], :, :],
        t1img[:, centre_vox_t1_int[1], :],
        t1img[:, :, centre_vox_t1_int[2]]]
    vox_centre = [[1, 2], [0, 2], [0, 1]]

    def plot_joined(ax, coordsx, coordsy):
        fullx = coordsx[[0, 1, 2, 3, 0, 4, 5, 1, 2, 6, 7, 4, 5, 6, 7, 3]]
        fully = coordsy[[0, 1, 2, 3, 0, 4, 5, 1, 2, 6, 7, 4, 5, 6, 7, 3]]
        ax.plot(fullx, fully, 'r')

    for idx, (ax, sli, loc) in enumerate(zip(axes, slices, vox_centre)):
        vmin = np.quantile(sli, .01)
        vmax = np.quantile(sli, .99)
        ax.imshow(sli.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

        # ax.plot(centre_vox_t1[loc[0]], centre_vox_t1[loc[1]], 'go')
        # for vertex in voxel_corners_t1:
        #     ax.plot(vertex[loc[0]], vertex[loc[1]], 'rx')

        plot_joined(ax, voxel_corners_t1[:, loc[0]], voxel_corners_t1[:, loc[1]])
        ax.hlines(centre_vox_t1_int[loc[1]], xmin=0, xmax=sli.shape[0])
        ax.vlines(centre_vox_t1_int[loc[0]], ymin=0, ymax=sli.shape[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, sli.shape[0]])
        ax.set_ylim([0, sli.shape[1]])

    return plt.gcf()
