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
from fsl_mrs.utils.misc import extract_spectrum, checkCFUnits, FIDToSpec, SpecToFID
from fsl_mrs.utils.preproc.shifting import pad
from fsl_mrs.utils.preproc.remove import hlsvd
from fsl_mrs.utils.preproc.filtering import apodize


def applyPhase(FID, phaseAngle):
    """
    Multiply FID by constant phase
    """
    return FID * np.exp(1j * phaseAngle)


def applyLinPhase(FID, frequency_axis, time):
    """
    Multiply spectrum by linear phase
    """
    return SpecToFID(FIDToSpec(FID) * np.exp(1j * 2 * np.pi * frequency_axis * time))


def phaseCorrect(FID, bw, cf, nucleus='1H', ppmlim=(2.8, 3.2), shift=True, use_hlsvd=False):
    """ Phase correction based on the phase of a maximum point.

    HLSVD is used to remove peaks outside the limits to flatten baseline first.

    Args:
        FID (ndarray): Time domain data
        bw (float): bandwidth
        cf (float): central frequency in Hz
        ppmlim (tuple,optional)  : Limit to this ppm range
        shift (bool,optional)    : Apply H20 shft
        use_hlsvd (bool,optional)    : Enable hlsvd step

    Returns:
        FID (ndarray): Phase corrected FID
        phaseAngle (double): shift in radians
        index (int): Index of phased point
    """

    cf = checkCFUnits(cf, units='Hz')

    if use_hlsvd:
        # Run HLSVD to remove peaks outside limits
        try:
            fid_hlsvd = hlsvd(FID, 1 / bw, cf, (ppmlim[1] + 0.5, ppmlim[1] + 3.0), limitUnits='ppm+shift')
            fid_hlsvd = hlsvd(fid_hlsvd, 1 / bw, cf, (ppmlim[0] - 3.0, ppmlim[0] - 0.5), limitUnits='ppm+shift')
        except Exception:
            fid_hlsvd = FID
            print('HLSVD in phaseCorrect failed, proceeding to phasing.')
    else:
        fid_hlsvd = FID

    # Find maximum of absolute spectrum in ppm limit
    padFID = pad(fid_hlsvd, FID.size * 3)
    MRSargs = {'FID': padFID,
               'bw': bw,
               'cf': cf,
               'nucleus': nucleus}
    mrs = MRS(**MRSargs)
    spec = extract_spectrum(mrs, padFID, ppmlim=ppmlim, shift=shift)

    maxIndex = np.argmax(np.abs(spec))
    phaseAngle = -np.angle(spec[maxIndex])

    return applyPhase(FID, phaseAngle), phaseAngle, int(np.round(maxIndex / 4))


def phaseCorrect_report(inFID,
                        outFID,
                        position,
                        bw,
                        cf,
                        nucleus='1H',
                        ppmlim=(2.8, 3.2),
                        html=None):
    """
    Generate report for phaseCorrect
    """
    # from matplotlib import pyplot as plt
    from fsl_mrs.core import MRS
    import plotly.graph_objects as go
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle

    # Turn input FIDs into mrs objects
    def toMRSobj(fid):
        return MRS(FID=fid, cf=cf, bw=bw, nucleus=nucleus)

    plotIn = toMRSobj(inFID)
    plotOut = toMRSobj(outFID)

    widelimit = (0, 6)

    # Fetch line styles
    lines, colors, _ = plotStyles()

    # Make a new figure
    fig = go.Figure()

    # Add lines to figure
    def addline(fig, mrs, lim, name, linestyle):
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim),
                           y=np.real(mrs.get_spec(ppmlim=lim)),
                           mode='lines',
                           name=name,
                           line=linestyle)
        return fig.add_trace(trace)

    fig = addline(fig, plotIn, widelimit, 'Unphased', lines['in'])
    fig = addline(fig, plotIn, ppmlim, 'Search region', lines['emph'])

    if position is None:
        # re-estimate here.
        position = np.argmax(np.abs(plotIn.get_spec(ppmlim=ppmlim)))

    # Deal with rounding errors
    if position >= len(plotIn.getAxes(ppmlim=ppmlim)):
        position = len(plotIn.getAxes(ppmlim=ppmlim)) - 1

    axis    = [plotIn.getAxes(ppmlim=ppmlim)[position]]
    y_data  = [np.real(plotIn.get_spec(ppmlim=ppmlim))[position]]
    trace = go.Scatter(x=axis, y=y_data,
                       mode='markers',
                       name='max point',
                       marker=dict(color=colors['emph'], symbol='x', size=8))
    fig.add_trace(trace)

    fig = addline(fig, plotOut, widelimit, 'Phased', lines['out'])

    # Axes layout
    plotAxesStyle(fig, widelimit, title='Phase correction summary')

    # Axes
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

        opName = 'Phase correction'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.preproc.phasing.phaseCorrect.\n'\
            + f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext='Phase correction of spectra based on maximum in the'
                                        f' range {ppmlim[0]} to {ppmlim[1]} ppm.',
                               afttext='')]

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig
    else:
        return fig


def phasta(
        data: np.ndarray,
        dwelltime: float,
        limits: tuple[int, int] | None = None,
        indices_to_use: list[int] | slice = slice(None),
        apodization: float = 0) -> tuple[np.ndarray, float]:
    """Phase correction of a FID or an array of FIDs based on LCModel's Phasta algorithm

    Phase is calculated using the mean of the array or a subset, as selected using indices_to_use

    :param data: FID or 2D array of FIDS, time is first dimension
    :type data: np.ndarray
    :param dwelltime: Dwelltime (1 / spectral bandwidth)
    :type dwelltime: float
    :param limits: Index limits to limit range over which algorithm is run, defaults to None
    :type limits: tuple, optional
    :param indices_to_use: Phase all FIDs based on mean of subset, defaults to slice(None) (use all in mean).
    :type indices_to_use: list[int] | slice, optional
    :param apodization: Apply apodization, defaults to 0 (no apodization)
    :type apodization: float, optional
    :return: Phase corrected FID array
    :rtype: np.ndarray
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    limits = slice(None) if limits is None else slice(limits[0], limits[1])

    data_apod = np.asarray([
        apodize(fid, dwelltime, apodization) for fid in data.T]).T

    deg_search = np.arange(0, 359, 1)
    range_6 = []
    spec_sum = []
    spec = FIDToSpec(
        np.mean(data_apod[:, indices_to_use], axis=-1))[limits]
    for deg in deg_search:
        re_spec = (spec * np.exp(1j * deg * np.pi / 180)).real
        range_6.append(np.sum(np.abs(re_spec - re_spec[0])**6))
        spec_sum.append(np.sum(re_spec))

    range_6 = np.asarray(range_6)
    spec_sum = np.asarray(spec_sum)
    max_deg = deg_search[spec_sum > 0][np.argmax(range_6[spec_sum > 0])]

    return data.squeeze() * np.exp(1j * max_deg * np.pi / 180), max_deg
