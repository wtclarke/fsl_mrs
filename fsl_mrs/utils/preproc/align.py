#!/usr/bin/env python

# align.py - Module containing spectral registration alignment functions (align and aligndiff)
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

from fsl_mrs.utils.preproc.general import get_target_FID, add, subtract
from fsl_mrs.utils.preproc.filtering import apodize as apod
from fsl_mrs.core import MRS
from fsl_mrs.utils.misc import extract_spectrum, shift_FID
from scipy.optimize import minimize
import numpy as np


# Phase-Freq alignment functions
def align_FID(mrs, src_FID, tgt_FID, ppmlim=None, shift=True):
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
    normalisation = np.linalg.norm(tgt_FID)

    # Internal functions so they can see globals
    def shift_phase_freq(FID, phi, eps, extract=True):
        sFID = np.exp(-1j * phi) * shift_FID(mrs, FID, eps)
        if extract:
            sFID = extract_spectrum(mrs, sFID, ppmlim=ppmlim, shift=shift)
        return sFID

    def cf(p):
        phi    = p[0]  # phase shift
        eps    = p[1]  # freq shift
        FID    = shift_phase_freq(src_FID, phi, eps)
        target = extract_spectrum(mrs, tgt_FID, ppmlim=ppmlim, shift=shift)
        xx     = np.linalg.norm((FID - target) / normalisation)
        return xx
    x0 = np.array([0, 0])
    res = minimize(cf, x0, method='Powell')
    phi = res.x[0]
    eps = res.x[1]

    return phi, eps


def align_FID_diff(mrs, src_FID0, src_FID1, tgt_FID, diffType='add', ppmlim=None, shift=True):
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
    normalisation = np.linalg.norm(tgt_FID)

    # Internal functions so they can see globals
    def shift_phase_freq(FID0, FID1, phi, eps, extract=True):
        sFID = np.exp(-1j * phi) * shift_FID(mrs, FID0, eps)
        if extract:
            sFID = extract_spectrum(mrs, sFID, ppmlim=ppmlim, shift=shift)
            FID1 = extract_spectrum(mrs, FID1, ppmlim=ppmlim, shift=shift)

        if diffType.lower() == 'add':
            FIDOut = add(FID1, sFID)
        elif diffType.lower() == 'sub':
            FIDOut = subtract(FID1, sFID)
        else:
            raise ValueError('diffType must be add or sub.')

        return FIDOut

    def cf(p):
        phi    = p[0]  # phase shift
        eps    = p[1]  # freq shift
        FID    = shift_phase_freq(src_FID0, src_FID1, phi, eps)
        target = extract_spectrum(mrs, tgt_FID, ppmlim=ppmlim, shift=shift)
        xx     = np.linalg.norm((FID - target) / normalisation)
        return xx

    x0  = np.array([0.0, 0.0])
    res = minimize(cf, x0)
    phi = res.x[0]
    eps = res.x[1]

    alignedFID0 = np.exp(-1j * phi) * shift_FID(mrs, src_FID0, eps)

    return alignedFID0, phi, eps


# The functions to call
# 1) For normal FIDs
def phase_freq_align(FIDlist,
                     bandwidth,
                     centralFrequency,
                     nucleus='1H',
                     ppmlim=None,
                     niter=2,
                     apodize=0,
                     verbose=False,
                     shift=True,
                     target=None):
    """
    Algorithm:
       Average spectra
       Loop over all spectra and find best phase/frequency shifts
       Iterate

    Parameters:
    -----------
    FIDlist          : list
    bandwidth        : float (unit=Hz)
    centralFrequency : float (unit=Hz)
    ppmlim           : tuple
    niter            : int
    apodize          : float (unit=Hz)
    verbose          : bool
    shift            : apply H20 shift to ppm limit
    ref              : reference data to align to

    Returns:
    --------
    list of FID aligned to each other
    """
    all_FIDs = FIDlist.copy()

    phiOut, epsOut = np.zeros(len(FIDlist)), np.zeros(len(FIDlist))
    for iter in range(niter):
        if verbose:
            print(' ---- iteration {} ----\n'.format(iter))

        if target is None:
            target = get_target_FID(all_FIDs, target='nearest_to_mean')

        MRSargs = {'FID': target, 'bw': bandwidth, 'cf': centralFrequency, 'nucleus': nucleus}
        mrs = MRS(**MRSargs)

        if apodize > 0:
            target = apod(target, mrs.dwellTime, [apodize])

        for idx, FID in enumerate(all_FIDs):
            if verbose:
                print(f'... aligning FID number {idx}\r')

            if apodize > 0:
                FID_apod = apod(FID.copy(), mrs.dwellTime, [apodize])
            else:
                FID_apod = FID

            phi, eps = align_FID(mrs,
                                 FID_apod,
                                 target,
                                 ppmlim=ppmlim,
                                 shift=shift)

            all_FIDs[idx] = np.exp(-1j * phi) * shift_FID(mrs, FID, eps)
            phiOut[idx] += phi
            epsOut[idx] += eps
        if verbose:
            print('\n')
    return all_FIDs, phiOut, epsOut


# 2) To align spectra from different groups with optional processing applied.
def phase_freq_align_diff(FIDlist0,
                          FIDlist1,
                          bandwidth,
                          centralFrequency,
                          nucleus='1H',
                          diffType='add',
                          ppmlim=None,
                          shift=True,
                          target=None):
    """ Align subspectra from difference methods.

    Only spectra in FIDlist0 are shifted.

    Parameters:
    -----------
    FIDlist0         : list - shifted
    FIDlist1         : list - fixed
    bandwidth        : float (unit=Hz)
    centralFrequency : float (unit=Hz)
    nucleus          : str
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
        for fid0, fid1 in zip(FIDlist0, FIDlist1):
            if diffType.lower() == 'add':
                diffFIDList.append(add(fid1, fid0))
            elif diffType.lower() == 'sub':
                diffFIDList.append(subtract(fid1, fid0))
            else:
                raise ValueError('diffType must be add or sub.')
        tgt_FID = get_target_FID(diffFIDList, target='nearest_to_mean')

    # Pass to phase_freq_align
    mrs = MRS(FID=FIDlist0[0], cf=centralFrequency, bw=bandwidth, nucleus=nucleus)
    phiOut, epsOut = [], []
    alignedFIDs0 = []
    for fid0, fid1 in zip(FIDlist0, FIDlist1):
        # breakpoint()
        out = align_FID_diff(mrs, fid0, fid1, tgt_FID, diffType=diffType,
                             ppmlim=ppmlim, shift=shift)
        alignedFIDs0.append(out[0])
        phiOut.append(out[1])
        epsOut.append(out[2])

    return alignedFIDs0, FIDlist1, phiOut, epsOut


# Reporting functions
def phase_freq_align_report(inFIDs,
                            outFIDs,
                            phi,
                            eps,
                            bw,
                            cf,
                            nucleus='1H',
                            ppmlim=None,
                            shift=True,
                            html=None):
    """
    Generate phase alignment report
    """
    from fsl_mrs.utils.preproc.combine import combine_FIDs
    import plotly.graph_objects as go
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle
    from plotly.subplots import make_subplots

    # Fetch line styles
    lines, _, _ = plotStyles()

    # Make a new figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Phase', 'Shift'])

    trace1 = go.Scatter(x=np.arange(1, len(phi) + 1),
                        y=np.array(phi) * (180.0 / np.pi),
                        mode='lines',
                        name='Phase',
                        line=lines['out'])
    fig.add_trace(trace1, row=1, col=1)
    fig.layout.xaxis.update(title_text='Transient #')
    fig.layout.yaxis.update(title_text='First-order phase (degrees)')

    trace2 = go.Scatter(x=np.arange(1, len(eps) + 1),
                        y=eps,
                        mode='lines',
                        name='Shift',
                        line=lines['diff'])
    fig.add_trace(trace2, row=1, col=2)
    fig.layout.yaxis2.update(title_text='Shift (Hz)')
    fig.layout.xaxis2.update(title_text='Transient #')

    # Transpose so time dimension is first
    meanIn = combine_FIDs(inFIDs.T, 'mean')
    meanOut = combine_FIDs(outFIDs.T, 'mean')

    def toMRSobj(fid):
        return MRS(FID=fid, cf=cf, bw=bw, nucleus=nucleus)

    meanIn = toMRSobj(meanIn)
    meanOut = toMRSobj(meanOut)

    if shift:
        axis = 'ppmshift'
    else:
        axis = 'ppm'

    toPlotIn, toPlotOut = [], []
    for fid in inFIDs:
        toPlotIn.append(toMRSobj(fid))
    for fid in outFIDs:
        toPlotOut.append(toMRSobj(fid))

    def addline(fig, mrs, lim, name, linestyle):
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim, axis=axis),
                           y=np.real(mrs.get_spec(ppmlim=lim, shift=shift)),
                           mode='lines',
                           name=name,
                           line=linestyle)
        return fig.add_trace(trace)
    fig2 = go.Figure()
    for idx, fid in enumerate(toPlotIn):
        cval = np.round(255 * idx / len(toPlotIn))
        linetmp = {'color': f'rgb(0,{cval},{cval})', 'width': 1}
        fig2 = addline(fig2, fid, ppmlim, f'#{idx}', linetmp)
    fig2 = addline(fig2, meanIn, ppmlim, 'Mean - Unligned', lines['blk'])
    plotAxesStyle(fig2, ppmlim, 'Unaligned')

    fig3 = go.Figure()
    for idx, fid in enumerate(toPlotOut):
        cval = np.round(255 * idx / len(toPlotIn))
        linetmp = {'color': f'rgb(0,{cval},{cval})', 'width': 1}
        fig3 = addline(fig3, fid, ppmlim, f'#{idx}', linetmp)
    fig3 = addline(fig3, meanIn, ppmlim, 'Mean - Unligned', lines['out'])
    fig3 = addline(fig3, meanOut, ppmlim, 'Mean - Aligned', lines['blk'])
    plotAxesStyle(fig3, ppmlim, 'Aligned')

    if html is not None:
        from plotly.offline import plot
        from fsl_mrs.utils.preproc.reporting import figgroup, singleReport
        from datetime import datetime
        import os.path as op

        if op.isdir(html):
            filename = 'report_' + datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3] + '.html'
            htmlfile = op.join(html, filename)
        elif op.isdir(op.dirname(html)) and op.splitext(html)[1] == '.html':
            htmlfile = html
        else:
            raise ValueError('Report html path must be file or directory. ')

        opName = 'Align'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.align.phase_freq_align.\n'\
            + f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext='Alignment parameters.',
                               afttext='')]
        div2 = plot(fig2, output_type='div', include_plotlyjs='cdn')
        figurelist.append(figgroup(fig=div2,
                                   name='',
                                   foretext='Transients before alignment.',
                                   afttext=''))
        div3 = plot(fig3, output_type='div', include_plotlyjs='cdn')
        figurelist.append(figgroup(fig=div3,
                                   name='',
                                   foretext='Transients after alignment.',
                                   afttext=''))

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig, fig2, fig3
    else:
        return fig, fig2, fig3


def phase_freq_align_diff_report(inFIDs0,
                                 inFIDs1,
                                 outFIDs0,
                                 outFIDs1,
                                 phi,
                                 eps,
                                 bw,
                                 cf,
                                 nucleus='1H',
                                 ppmlim=None,
                                 diffType='add',
                                 shift=True,
                                 html=None):
    from fsl_mrs.utils.preproc.combine import combine_FIDs
    import plotly.graph_objects as go
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle
    from plotly.subplots import make_subplots

    # Fetch line styles
    lines, _, _ = plotStyles()

    # Make a new figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Phase', 'Shift'])

    trace1 = go.Scatter(x=np.arange(1, len(phi) + 1),
                        y=np.array(phi) * (180.0 / np.pi),
                        mode='lines',
                        name='Phase',
                        line=lines['out'])
    fig.add_trace(trace1, row=1, col=1)
    fig.layout.xaxis.update(title_text='Transient #')
    fig.layout.yaxis.update(title_text='First-order phase (degrees)')

    trace2 = go.Scatter(x=np.arange(1, len(eps) + 1),
                        y=eps,
                        mode='lines',
                        name='Shift',
                        line=lines['diff'])
    fig.add_trace(trace2, row=1, col=2)
    fig.layout.yaxis2.update(title_text='Shift (Hz)')
    fig.layout.xaxis2.update(title_text='Transient #')

    diffFIDListIn = []
    diffFIDListOut = []
    for fid0i, fid1i, fid0o, fid1o in zip(inFIDs0, inFIDs1, outFIDs0, outFIDs1):
        if diffType.lower() == 'add':
            diffFIDListIn.append(add(fid1i, fid0i))
            diffFIDListOut.append(add(fid1o, fid0o))
        elif diffType.lower() == 'sub':
            diffFIDListIn.append(subtract(fid1i, fid0i))
            diffFIDListOut.append(subtract(fid1o, fid0o))
        else:
            raise ValueError('diffType must be add or sub.')

    meanIn = combine_FIDs(diffFIDListIn, 'mean')
    meanOut = combine_FIDs(diffFIDListOut, 'mean')

    def toMRSobj(fid):
        return MRS(FID=fid, cf=cf, bw=bw, nucleus=nucleus)

    meanIn = toMRSobj(meanIn)
    meanOut = toMRSobj(meanOut)

    if shift:
        axis = 'ppmshift'
    else:
        axis = 'ppm'

    toPlotIn, toPlotOut = [], []
    for fid in diffFIDListIn:
        toPlotIn.append(toMRSobj(fid))
    for fid in diffFIDListOut:
        toPlotOut.append(toMRSobj(fid))

    def addline(fig, mrs, lim, name, linestyle):
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim, axis=axis),
                           y=np.real(mrs.get_spec(ppmlim=lim, shift=shift)),
                           mode='lines',
                           name=name,
                           line=linestyle)
        return fig.add_trace(trace)
    fig2 = go.Figure()
    for idx, fid in enumerate(toPlotIn):
        cval = np.round(255 * idx / len(toPlotIn))
        linetmp = {'color': f'rgb(0,{cval},{cval})', 'width': 1}
        fig2 = addline(fig2, fid, ppmlim, f'#{idx}', linetmp)
    fig2 = addline(fig2, meanIn, ppmlim, 'Mean - Unligned', lines['blk'])
    plotAxesStyle(fig2, ppmlim, 'Unaligned')

    fig3 = go.Figure()
    for idx, fid in enumerate(toPlotOut):
        cval = np.round(255 * idx / len(toPlotIn))
        linetmp = {'color': f'rgb(0,{cval},{cval})', 'width': 1}
        fig3 = addline(fig3, fid, ppmlim, f'#{idx}', linetmp)
    fig3 = addline(fig3, meanIn, ppmlim, 'Mean - Unligned', lines['out'])
    fig3 = addline(fig3, meanOut, ppmlim, 'Mean - Aligned', lines['blk'])
    plotAxesStyle(fig3, ppmlim, 'Aligned')

    if html is not None:
        from plotly.offline import plot
        from fsl_mrs.utils.preproc.reporting import figgroup, singleReport
        from datetime import datetime
        import os.path as op

        if op.isdir(html):
            filename = 'report_' + datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3] + '.html'
            htmlfile = op.join(html, filename)
        elif op.isdir(op.dirname(html)) and op.splitext(html)[1] == '.html':
            htmlfile = html
        else:
            raise ValueError('Report html path must be file or directory. ')

        opName = 'AlignDiff'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.align.phase_freq_align_diff.\n'\
            + f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext='Alignment parameters.',
                               afttext='')]
        div2 = plot(fig2, output_type='div', include_plotlyjs='cdn')
        figurelist.append(figgroup(fig=div2,
                                   name='',
                                   foretext='Transients before alignment.',
                                   afttext=''))
        div3 = plot(fig3, output_type='div', include_plotlyjs='cdn')
        figurelist.append(figgroup(fig=div3,
                                   name='',
                                   foretext='Transients after alignment.',
                                   afttext=''))

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig, fig2, fig3
    else:
        return fig, fig2, fig3
