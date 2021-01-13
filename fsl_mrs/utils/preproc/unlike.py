# unlike.py - Outlier detection routines
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

from fsl_mrs.core import MRS
import numpy as np
from fsl_mrs.utils.preproc.general import get_target_FID
from fsl_mrs.utils.misc import extract_spectrum, FIDToSpec


def identifyUnlikeFIDs(FIDList,
                       bandwidth,
                       centralFrequency,
                       nucleus='1H',
                       sdlimit=1.96,
                       iterations=2,
                       ppmlim=None,
                       shift=True):
    """ Identify FIDs in a list that are unlike the others

    Args:
        FIDList (list of ndarray): Time domain data
        bandwidth (float)        : Bandwidth in Hz
        centralFrequency (float) : Central frequency in Hz
        sdlimit (float,optional) : Exclusion limit (number of standard deviations). Default = 3.
        iterations (int,optional): Number of iterations to use.
        ppmlim (tuple,optional)  : Limit to this ppm range
        shift (bool,optional)    : Apply H20 shft

    Returns:
        goodFIDS (list of ndarray): FIDs that passed the criteria
        badFIDS (list of ndarray): FIDs that failed the likeness critera
        rmIndicies (list of int): Indicies of those FIDs that have been removed
        metric (list of floats): Likeness metric of each FID
    """

    # Calculate the FID to compare to
    target = get_target_FID(FIDList, target='median')

    if ppmlim is not None:
        MRSargs = {'FID': target,
                   'bw': bandwidth,
                   'cf': centralFrequency,
                   'nucleus': nucleus}
        mrs = MRS(**MRSargs)

        target = extract_spectrum(mrs, target, ppmlim=ppmlim, shift=shift)
        compareList = [extract_spectrum(mrs, f, ppmlim=ppmlim, shift=shift) for f in FIDList]
    else:
        compareList = [FIDToSpec(f) for f in FIDList]
        target = FIDToSpec(target)

    # Do the comparison
    for idx in range(iterations):
        metric = []
        for data in compareList:
            metric.append(np.linalg.norm(data - target))
        metric = np.asarray(metric)
        metric_avg = np.mean(metric)
        metric_std = np.std(metric)

        goodFIDs, badFIDs, rmIndicies, keepIndicies = [], [], [], []
        for iDx, (data, m) in enumerate(zip(FIDList, metric)):
            if m > ((sdlimit * metric_std) + metric_avg) or m < (-(sdlimit * metric_std) + metric_avg):
                badFIDs.append(data)
                rmIndicies.append(iDx)
            else:
                goodFIDs.append(data)
                keepIndicies.append(iDx)

        target = get_target_FID(goodFIDs, target='median')
        if ppmlim is not None:
            target = extract_spectrum(mrs, target, ppmlim=ppmlim, shift=shift)
        else:
            target = FIDToSpec(target)

    return goodFIDs, badFIDs, keepIndicies, rmIndicies, metric.tolist()


def identifyUnlikeFIDs_report(goodFIDs,
                              badFIDs,
                              keepIndicies,
                              rmIndicies,
                              metric,
                              bw,
                              cf,
                              nucleus='1H',
                              ppmlim=(0.2, 4.2),
                              sdlimit=1.96,
                              html=None):
    import plotly.graph_objects as go
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle

    metricGd = np.array(metric)[keepIndicies]
    metricBd = np.array(metric)[rmIndicies]
    metric_avg = np.mean(metric)
    metric_std = np.std(metric)

    metricGd_SD = np.abs(metricGd - metric_avg) / metric_std
    metricBd_SD = np.abs(metricBd - metric_avg) / metric_std

    gdIndex = np.argsort(metricGd_SD)
    bdIndex = np.argsort(metricBd_SD)

    plotGood, plotBad = [], []
    gdLegend, bdLegend = [], []

    # Turn input FIDs into mrs objects
    def toMRSobj(fid):
        return MRS(FID=fid, cf=cf, bw=bw, nucleus=nucleus)

    for idx in gdIndex:
        fid = goodFIDs[idx]
        plotGood.append(toMRSobj(fid))
        gdLegend.append(f'Kept (SD={metricGd_SD[idx]:0.2f})')
    for idx in bdIndex:
        fid = badFIDs[idx]
        plotBad.append(toMRSobj(fid))
        bdLegend.append(f'Removed (SD={metricBd_SD[idx]:0.2f})')

    target = get_target_FID(goodFIDs, target='median')
    tgtmrs = toMRSobj(target)

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
    for fid, leg in zip(plotGood, gdLegend):
        fig = addline(fig, fid, ppmlim, leg, lines['out'])
    for fid, leg in zip(plotBad, bdLegend):
        fig = addline(fig, fid, ppmlim, leg, lines['emph'])
    fig = addline(fig, tgtmrs, ppmlim, 'Target', lines['blk'])

    plotAxesStyle(fig, ppmlim, title='Bad average removal summary')

    # Generate report
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

        opName = 'BadAverageRemoval'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.preproc.unlike.identifyUnlikeFIDs.\n'\
            + f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext=f'Identification of FIDs unlike others. SD limit = {sdlimit:0.2f}',
                               afttext='')]

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig
    else:
        return fig
