# filtering.py - Routines for FID filtering
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from scipy.optimize import minimize


def apodize(FID, dwelltime, broadening, filter='exp'):
    """ Apodize FID

    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwelltime in seconds
        broadening (tuple,float): apodisation in Hz
        filter (str,optional):'exp','l2g'

    Returns:
        FID (ndarray): Apodised FID
    """
    # Deal with non-array arguments (for user friendliness)
    if isinstance(broadening, (float, int)):
        broadening = [broadening, ]

    # If any values are zero just return (unfiltered) FID
    if not all(broadening):
        return FID

    taxis = np.linspace(0, dwelltime * (FID.size - 1), FID.size)
    if filter == 'exp':
        Tl = 1 / broadening[0]
        window = np.exp(-taxis / Tl)
    elif filter == 'l2g':
        Tl = 1 / broadening[0]
        Tg = 1 / broadening[1]
        window = np.exp(taxis / Tl) * np.exp(taxis**2 / Tg**2)
    else:
        print('Filter not recognised, should be "exp" or "l2g".')
        window = 1
    return window * FID


def apodize_report(inFID,
                   outFID,
                   bw,
                   cf,
                   nucleus='1H',
                   plotlim=(0.2, 6),
                   html=None):
    """
    Generate report
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

    fig = addline(fig, plotIn, plotlim, 'Uncorrected', lines['in'])
    fig = addline(fig, plotOut, plotlim, 'Corrected', lines['out'])

    # Axes layout
    plotAxesStyle(fig, plotlim, title='Apodization summary')

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

        opName = 'Apodization'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.preproc.filtering.apodize.\n'\
            + f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext='Apodization of spectra.',
                               afttext='')]

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig
    else:
        return fig


def calc_aprox_t2decay(
        fids: np.ndarray,
        dwelltime: float) -> float:
    """Fits an exponential decay to mean of data to extract a rough T2*.

    :param fids: Array of FID data
    :type fids: np.ndarray
    :param dwelltime: dwelltime (1/bandwidth) of FIDs
    :type dwelltime: float
    :return: Apodisation amount in hertz
    :rtype: float
    """
    timeaxis = np.arange(
        0,
        dwelltime * fids.shape[-1],
        dwelltime)

    avg_data = np.mean(np.abs(fids), axis=0)
    avg_data /= avg_data[0]

    def exp_model_offset(p):
        return p[0] + p[2] * np.exp(-timeaxis * p[1] * np.pi)

    def loss(x):
        return np.linalg.norm(avg_data - exp_model_offset(x))

    xout = minimize(
        loss,
        [0, 10, 1],
        bounds=((0, 1), (0.1, 100), (0, 2))).x

    # print(xout)
    # import matplotlib.pyplot as plt
    # plt.plot(timeaxis, avg_data)
    # plt.plot(timeaxis, exp_model_offset(xout))
    # plt.show()

    return xout[1]
