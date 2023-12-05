# shifting.py - Shifting routines
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.core import MRS
from fsl_mrs.utils.misc import extract_spectrum


def timeshift(FID, dwelltime, shiftstart, shiftend, samples=None):
    """ Shift data on time axis

    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwell time in seconds
        shiftstart (float): Shift start point in seconds
        shiftend (float): Shift end point in seconds
        samples (int, optional): Resample to this number of points

    Returns:
        FID (ndarray): Shifted FID
    """
    originalAcqTime = dwelltime * (FID.size - 1)
    originalTAxis = np.linspace(0, originalAcqTime, FID.size)
    if samples is None:
        newDT = dwelltime
    else:
        totalacqTime = originalAcqTime - shiftstart + shiftend
        newDT = totalacqTime / samples
    newTAxis = np.arange(originalTAxis[0] + shiftstart, originalTAxis[-1] + shiftend, newDT)
    FID = np.interp(newTAxis, originalTAxis, FID, left=0.0 + 1j * 0.0, right=0.0 + 1j * 0.0)

    return FID, newDT


def freqshift(FID, dwelltime, shift):
    """ Shift data on frequency axis

    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwelltime in seconds
        shift (float): shift in Hz

    Returns:
        FID (ndarray): Shifted FID
    """
    tAxis = np.linspace(0, dwelltime * FID.size, FID.size)
    phaseRamp = 2 * np.pi * tAxis * shift
    FID = FID * np.exp(1j * phaseRamp)
    return FID


def shiftToRef(FID, target, bw, cf, nucleus='1H', ppmlim=(2.8, 3.2), shift=True):
    '''Find a maximum and shift that maximum to a reference position.

    :param FID: FID
    :param float target: reference position in ppm
    :param float bw: Bandwidth or spectral width in Hz.
    :param float cf: Central or spectrometer frequency (MHz)
    :param str nucleus: Nucleus string, defaults to 1H
    :param ppmlim: Search range for peak maximum
    :param bool shift: If True (default) ppm values include shift

    :return: Shifted FID
    :return: Shifted amount in ppm
    '''

    # Find maximum of absolute spectrum in ppm limit
    padFID = pad(FID, FID.size * 3)
    MRSargs = {'FID': padFID,
               'bw': bw,
               'cf': cf,
               'nucleus': nucleus}
    mrs = MRS(**MRSargs)
    spec = extract_spectrum(mrs, padFID, ppmlim=ppmlim, shift=shift)
    if shift:
        extractedAxis = mrs.getAxes(ppmlim=ppmlim)
    else:
        extractedAxis = mrs.getAxes(ppmlim=ppmlim, axis='ppm')

    maxIndex = np.argmax(np.abs(spec))
    shiftAmount = extractedAxis[maxIndex] - target
    shiftAmountHz = shiftAmount * mrs.centralFrequency / 1E6

    return freqshift(FID, 1 / bw, -shiftAmountHz), shiftAmount


def truncate(FID, k, first_or_last='last'):
    """
    Truncate parts of a FID

    Parameters:
    -----------
    FID           : array-like
    k             : int (number of timepoints to remove)
    first_or_last : either 'first' or 'last' (which bit to truncate)

    Returns:
    --------
    array-like
    """
    FID_trunc = FID.copy()

    if first_or_last == 'first':
        return FID_trunc[k:]
    elif first_or_last == 'last':
        return FID_trunc[:-k]
    else:
        raise ValueError("Last parameter must either be 'first' or 'last'")


def pad(FID, k, first_or_last='last'):
    """
    Pad parts of a FID

    Parameters:
    -----------
    FID           : array-like
    k             : int (number of timepoints to add)
    first_or_last : either 'first' or 'last' (which bit to pad)

    Returns:
    --------
    array-like
    """
    FID_pad = FID.copy()

    if first_or_last == 'first':
        return np.pad(FID_pad, (k, 0))
    elif first_or_last == 'last':
        return np.pad(FID_pad, (0, k))
    else:
        raise ValueError("Last parameter must either be 'first' or 'last'")


def shift_report(inFID,
                 outFID,
                 inHdr,
                 outHdr,
                 ppmlim=(0.2, 4.2),
                 html=None,
                 function='shift'):
    """
    Generate report
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle

    plotIn = MRS(FID=inFID, header=inHdr)
    plotOut = MRS(FID=outFID, header=outHdr)

    # Fetch line styles
    lines, _, _ = plotStyles()

    # Make a new figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Spectra', 'FID'])

    # Add lines to figure
    trace1 = go.Scatter(x=plotIn.getAxes(ppmlim=ppmlim),
                        y=np.real(plotIn.get_spec(ppmlim=ppmlim)),
                        mode='lines',
                        name='Original',
                        line=lines['in'])
    trace2 = go.Scatter(x=plotOut.getAxes(ppmlim=ppmlim),
                        y=np.real(plotOut.get_spec(ppmlim=ppmlim)),
                        mode='lines',
                        name='Shifted',
                        line=lines['out'])
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)

    # Add lines to figure
    trace3 = go.Scatter(x=plotIn.getAxes(axis='time'),
                        y=np.real(plotIn.FID),
                        mode='lines',
                        name='Original',
                        line=lines['emph'])
    trace4 = go.Scatter(x=plotOut.getAxes(axis='time'),
                        y=np.real(plotOut.FID),
                        mode='lines',
                        name='Shifted',
                        line=lines['diff'])
    fig.add_trace(trace3, row=1, col=2)
    fig.add_trace(trace4, row=1, col=2)

    # Axes layout
    plotAxesStyle(fig, ppmlim, title='Shift summary')
    fig.layout.xaxis2.update(title_text='Time (s)')
    fig.layout.yaxis2.update(zeroline=True,
                             zerolinewidth=1,
                             zerolinecolor='Gray',
                             showgrid=False,
                             showticklabels=False)

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

        operation, function, description = reportStrings(function)

        opName = operation
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = f'Report for fsl_mrs.utils.preproc.shifting.{function}.\n'\
            + f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext=f'{description}',
                               afttext='')]

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig
    else:
        return fig


def reportStrings(funcName):
    if funcName.lower() == 'timeshift':
        operation = 'Time domain shift'
        description = 'Interpolation in timedomain.'
    elif funcName.lower() == 'freqshift':
        operation = 'Frequency domain shift'
        description = 'Fixed shift in frequency domain.'
    elif funcName.lower() == 'shifttoref':
        operation = 'Shift to ref'
        description = 'Frequency shift to reference peak (max in range).'
    elif funcName.lower() == 'truncate':
        operation = 'Truncate'
        description = 'Truncation in time domain.'
    elif funcName.lower() == 'pad':
        operation = 'Zero Pad'
        description = 'Zeropadding in time domain.'
    elif funcName.lower() == 'shift':  # Generic
        operation = 'Shift'
        funcName = '####'
        description = 'Unspecified shift operation.'
    else:
        raise ValueError(f'{funcName} not recognised as function.')

    return operation, funcName, description

# def shift_report(inFID,outFID,hdr,ppmlim = (0.2,4.2)):
#     from matplotlib import pyplot as plt
#     from fsl_mrs.utils.plotting import styleSpectrumAxes

#     toMRSobj = lambda fid : MRS(FID=fid,header=hdr)
#     plotIn = toMRSobj(inFID)
#     plotOut = toMRSobj(outFID)

#     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,7))

#     ax1.plot(plotIn.getAxes(ppmlim=ppmlim),np.real(plotIn.get_spec(ppmlim=ppmlim)),'k',label='Original', linewidth=2)
#     ax1.plot(plotOut.getAxes(ppmlim=ppmlim),np.real(plotOut.get_spec(ppmlim=ppmlim)),'r',label='Shifted', linewidth=2)
#     styleSpectrumAxes(ax=ax1)
#     ax1.legend()

#     ax2.plot(plotIn.getAxes(axis='time'),np.real(plotIn.FID),'k',label='Original', linewidth=2)
#     ax2.plot(plotOut.getAxes(axis='time'),np.real(plotOut.FID),'r--',label='Shifted', linewidth=2)
#     # styleSpectrumAxes(ax=ax2)
#     ax2.legend()
#     ax2.set_yticks([0.0])
#     ax2.set_ylabel('Re(signal) (a.u.)')
#     ax2.set_xlabel('Time (s)')

#     ax2.autoscale(enable=True, axis='x', tight=False)

#     plt.rcParams.update({'font.size': 12})
#     plt.show()
