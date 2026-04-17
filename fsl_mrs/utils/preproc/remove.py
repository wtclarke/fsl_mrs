# remove.py - Removal of peaks, zeroing or HLSVD-based correction
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
import hlsvdpropy
from fsl_mrs.core import FIDtoMRSobj
from fsl_mrs.utils.misc import checkCFUnits, FIDToSpec, SpecToFID
from fsl_mrs.utils.constants import PPM_SHIFT
H2O_PPM_TO_TMS = PPM_SHIFT['1H']


def zero_spectrum(FID, axes, limits, limitUnits='ppmshift'):
    """Zero part of a spectrum between limits

    :param FID: FID to modify
    :type FID: nump.array
    :param axes: Metadata/axes source
    :type axes: nifti_mrs.axes.Axes
    :param limits: ppm limits
    :type limits: tuple
    :param limitUnits: Whether limits include shift ('ppm' or 'ppmshift), defaults to 'ppmshift'
    :type limitUnits: str, optional
    :return: Modified FID
    :rtype: np.array
    """
    axis_lookup = {
        'ppm': lambda axes: axes.ppmIndices(limits),
        'ppmshift': lambda axes: axes.ppmShiftIndices(limits),
        'ppm+shift': lambda axes: axes.ppmShiftIndices(limits),
        'hz': lambda axes: axes.frequencyIndices(limits)}
    try:
        indices = axis_lookup[limitUnits.lower()]
    except KeyError as exc:
        raise ValueError('limitUnits should be one of: ppm, ppmshift, ppm+shift or hz.') from exc

    spec = FIDToSpec(FID)
    spec[indices(axes)] = 0.0 + 1j * 0.0
    mod_fid = SpecToFID(spec)
    return mod_fid


def model_fid_hlsvd(FID, axes, limits=None,
                    limitUnits='ppm', numSingularValues=20):
    """Model a section of the FID using HLSVD. Optionally retain components
    only within the frequenccy/ppm limits.

    :param FID: Time domain data
    :type FID: numpy.array
    :param axes: Metadata/axes source
    :type axes: nifti_mrs.axes.Axes
    :param limits: Frequency/ppm limits, defaults to None
    :type limits: tuple, optional
    :param limitUnits: Axis that limits are given in. By Default
        in ppm, relative to receiver frequency (no shift). Can be 'Hz', 'ppm'
        or 'ppm+shift'. Defaults to 'ppm'
    :type limitUnits: str, optional
    :param numSingularValues: Max number of singular values, defaults to 20
    :type numSingularValues: int, optional
    """

    return _hlsvd(
        FID,
        axes.dwelltime,
        axes.SpectrometerFrequency * 1E6,
        limits,
        limitUnits=limitUnits,
        numSingularValues=numSingularValues)


def hlsvd(FID, axes, limits,
          limitUnits='ppm', numSingularValues=20, sparse_algo=False):
    """ Run HLSVDPRO on FID

    Args:
        FID (ndarray): Time domain data
        axes (Axes) : Metadata/axes source
        limits (tuple): Limit deletion of singular values in this range.
        limitUnits (str,optional): Axis that limits are given in. By Default
        in ppm, relative to receiver frequency (no shift). Can be 'Hz', 'ppm'
        or 'ppm+shift'.
        numSingularValues (int, optional): Max number of singular values

    Returns:
        FID (ndarray): Modified FID
    """
    sumFID = _hlsvd(
        FID,
        axes.dwelltime,
        axes.SpectrometerFrequency * 1E6,
        limits,
        limitUnits=limitUnits,
        numSingularValues=numSingularValues,
        sparse_algo=sparse_algo)

    return FID - sumFID


def _hlsvd(FID, dwelltime, centralFrequency, limits,
           limitUnits='ppm', numSingularValues=20, sparse_algo=False):
    """Run hlsvdpro on FID and return spectrum modeled by HLSVD.

    :param FID: Time domain data
    :type FID: numpy.array
    :param dwelltime: dwell time in seconds
    :type dwelltime: float
    :param centralFrequency: Central frequency in Hz
    :type centralFrequency: float
    :param limits: Frequency/ppm limits, defaults to None
    :type limits: tuple, optional
    :param limitUnits: Axis that limits are given in. By Default
        in ppm, relative to receiver frequency (no shift). Can be 'Hz', 'ppm'
        or 'ppm+shift'. Defaults to 'ppm'
    :type limitUnits: str, optional
    :param numSingularValues: Max number of singular values, defaults to 20
    :type numSingularValues: int, optional

    :return: HLSVD modeled FID
    """
    m = FID.size // 2
    r = hlsvdpropy.hlsvdpro(FID, numSingularValues, m=m, sparse=sparse_algo)
    r = hlsvdpropy.convert_hlsvd_result(r, dwelltime)
    nsv_found, singular_values, frequencies, damping_factors, amplitudes, \
        phases = r[0:6]

    # convert to np array
    frequencies = np.asarray(frequencies)
    damping_factors = np.asarray(damping_factors)
    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)

    # Limit by frequencies
    if limitUnits.lower() == 'ppm':
        centralFrequency = checkCFUnits(centralFrequency, units='MHz')
        frequencylimit = np.array(limits) * centralFrequency
    elif limitUnits.lower() == 'ppm+shift':
        centralFrequency = checkCFUnits(centralFrequency, units='MHz')
        frequencylimit = (np.array(limits) - H2O_PPM_TO_TMS) * centralFrequency
    elif limitUnits.lower() == 'hz':
        frequencylimit = limits
    else:
        raise ValueError('limitUnits should be one of: ppm, ppm+shift or hz.')
    limitIndicies = (frequencies > frequencylimit[0]) & \
                    (frequencies < frequencylimit[1])

    sumFID = np.zeros(FID.shape, dtype=np.complex128)
    timeAxis = np.linspace(0, dwelltime * (FID.size - 1), FID.size)

    for use, f, d, a, p in zip(limitIndicies,
                               frequencies,
                               damping_factors,
                               amplitudes,
                               phases):
        if use:
            sumFID += a * np.exp((timeAxis / d)
                                 + 1j * 2 * np.pi
                                 * (f * timeAxis + p / 360.0))
    return sumFID


def hlsvd_report(in_mrs,
                 out_mrs,
                 limits,
                 limitUnits='ppm',
                 plotlim=(0.2, 6),
                 html=None):
    """
    Generate hlsvd report
    """
    import plotly.graph_objects as go
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle

    plotDiff = FIDtoMRSobj(out_mrs.FID - in_mrs.FID, in_mrs._axes_obj)

    if limitUnits.lower() == 'ppm':
        limits = np.array(limits) + H2O_PPM_TO_TMS
    elif limitUnits.lower() == 'ppm+shift':
        pass
    elif limitUnits.lower() == 'hz':
        limits = (np.array(limits) / (in_mrs.centralFrequency / 1E6)) + \
            H2O_PPM_TO_TMS
    else:
        raise ValueError('limitUnits should be one of: ppm, ppm+shift or hz.')

    # Fetch line styles
    lines, colors, _ = plotStyles()

    # Make a new figure
    fig = go.Figure()

    # Add lines to figure
    def addline(fig, mrs, lim, name, linestyle):
        trace = go.Scatter(x=mrs.getAxes(limits=lim),
                           y=np.real(mrs.get_spec(ppmlim=lim)),
                           mode='lines',
                           name=name,
                           line=linestyle)
        return fig.add_trace(trace)

    fig = addline(fig, in_mrs, plotlim, 'Uncorrected', lines['in'])
    fig = addline(fig, in_mrs, limits, 'Limits', lines['emph'])
    fig = addline(fig, out_mrs, plotlim, 'Corrected', lines['out'])
    fig = addline(fig, plotDiff, plotlim, 'Difference', lines['diff'])

    # Axes layout
    plotAxesStyle(fig, plotlim, title='HLSVD summary')

    # Axes
    if html is not None:
        from plotly.offline import plot
        from fsl_mrs.utils.preproc.reporting import figgroup, singleReport
        from datetime import datetime
        import os.path as op

        if op.isdir(html):
            filename = 'report_' + \
                       datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3] + '.html'
            htmlfile = op.join(html, filename)
        elif op.isdir(op.dirname(html)) and op.splitext(html)[1] == '.html':
            htmlfile = html
        else:
            raise ValueError('Report html path must be file or directory. ')

        opName = 'HLSVD'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.preproc.remove.HLSVD.\n' + \
                     f'Generated at {timestr} on {datestr}.'

        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext='HLSVD removal of peaks in the range'
                                        f' {limits[0]:0.1f} to'
                                        f' {limits[1]:0.1f} ppm.',
                               afttext='')]

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig
    else:
        return fig
