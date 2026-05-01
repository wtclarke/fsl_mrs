# shifting.py - Shifting routines
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.core import MRS
from nifti_mrs.axes import Axes
from fsl_mrs.utils.misc import extract_spectrum


def timeshift(FID: np.ndarray,
              axes: Axes,
              shiftstart: float,
              shiftend: float,
              samples: int = None) -> tuple[np.ndarray, float]:
    """ Shift data on time axis

    Args:
        FID (ndarray): Time domain data
        axes (Axes): Metadata/axes source
        shiftstart (float): Shift start point in seconds
        shiftend (float): Shift end point in seconds
        samples (int, optional): Resample to this number of points

    Returns:
        FID (ndarray): Shifted FID
        newDT (float): New dwell time
    """
    if samples is None:
        newDT = axes.dwelltime
    else:
        totalacqTime = axes.timeAxis[-1] - axes.timeAxis[0] - shiftstart + shiftend
        newDT = totalacqTime / samples
    newTAxis = np.arange(axes.timeAxis[0] + shiftstart, axes.timeAxis[-1] + shiftend, newDT)
    FID = np.interp(newTAxis, axes.timeAxis, FID, left=0.0 + 1j * 0.0, right=0.0 + 1j * 0.0)

    return FID, newDT


def freqshift(FID: np.ndarray,
              axes: Axes,
              shift: float) -> np.ndarray:
    """ Shift data on frequency axis

    Args:
        FID (ndarray): Time domain data
        axes (Axes): Metadata/axes source
        shift (float): shift in Hz

    Returns:
        FID (ndarray): Shifted FID
    """
    phaseRamp = 2 * np.pi * axes.timeAxis * shift
    FID = FID * np.exp(1j * phaseRamp)
    return FID


def freqshift_array(
        fid_array: np.ndarray,
        axes: Axes,
        shift_array: np.ndarray | float) -> np.ndarray:
    """Apply shifts to a grid of data without looping

    :param fid_array: ND array of FIDs. Last dimension is time.
    :type fid_array: np.ndarray
    :param axes: Metadata/axes source
    :type axes: MRS
    :param shift_array: Either a single value or an array matching fid_array spatial size
    :type shift_array: np.ndarray | float
    :return: Shifted FIDs
    :rtype: np.ndarray
    """
    if isinstance(shift_array, np.ndarray)\
            and shift_array.shape != fid_array.shape[:-1]:
        raise ValueError('shift_array must be float or array matching spatial size of fid_array.')

    if isinstance(shift_array, float):
        phaseRamp = 2 * np.pi * axes.timeAxis * shift_array
    else:
        phaseRamp = 2 * np.pi * axes.timeAxis * shift_array[..., np.newaxis]

    return fid_array * np.exp(1j * phaseRamp)


def shiftToRef(FID: np.ndarray,
               target: float,
               axes: Axes,
               ppmlim: tuple = (2.8, 3.2),
               shift: bool = True) -> tuple[np.ndarray, float]:
    '''Find a maximum and shift that maximum to a reference position.

    :param FID: FID
    :param float target: reference position in ppm
    :param Axes axes: Metadata/axes source
    :param ppmlim: Search range for peak maximum
    :param bool shift: If True (default) ppm values include shift

    :return: Shifted FID
    :return: Shifted amount in ppm
    '''

    # Find maximum of absolute spectrum in ppm limit
    padFID = pad(FID, FID.size * 3)
    pad_mrs = MRS.from_axes(padFID, axes)
    spec = extract_spectrum(pad_mrs, padFID, ppmlim=ppmlim, shift=shift)
    if shift:
        extractedAxis = pad_mrs.getAxes(limits=ppmlim)
    else:
        extractedAxis = pad_mrs.getAxes(limits=ppmlim, axis='ppm')

    maxIndex = np.argmax(np.abs(spec))
    shiftAmount = extractedAxis[maxIndex] - target
    shiftAmountHz = shiftAmount * pad_mrs.centralFrequency / 1E6

    return freqshift(FID, pad_mrs.axes, -shiftAmountHz), shiftAmount


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


def shift_report(in_mrs,
                 out_mrs,
                 ppmlim=(0.2, 4.2),
                 html=None,
                 function='shift'):
    """
    Generate report
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle

    # Fetch line styles
    lines, _, _ = plotStyles()

    # Make a new figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Spectra', 'FID'])

    # Add lines to figure
    trace1 = go.Scatter(x=in_mrs.getAxes(limits=ppmlim),
                        y=np.real(in_mrs.get_spec(ppmlim=ppmlim)),
                        mode='lines',
                        name='Original',
                        line=lines['in'])
    trace2 = go.Scatter(x=out_mrs.getAxes(limits=ppmlim),
                        y=np.real(out_mrs.get_spec(ppmlim=ppmlim)),
                        mode='lines',
                        name='Shifted',
                        line=lines['out'])
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)

    # Add lines to figure
    trace3 = go.Scatter(x=in_mrs.getAxes(axis='time'),
                        y=np.real(in_mrs.FID),
                        mode='lines',
                        name='Original',
                        line=lines['emph'])
    trace4 = go.Scatter(x=out_mrs.getAxes(axis='time'),
                        y=np.real(out_mrs.FID),
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

#     ax1.plot(plotIn.getAxes(limits=ppmlim),np.real(plotIn.get_spec(ppmlim=ppmlim)),'k',label='Original', linewidth=2)
#     ax1.plot(plotOut.getAxes(limits=ppmlim),np.real(plotOut.get_spec(ppmlim=ppmlim)),'r',label='Shifted', linewidth=2)
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
