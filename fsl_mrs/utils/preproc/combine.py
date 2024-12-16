# combine.py - Module containing functions for combining FIDs, includes coil combination
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np


def dephase(FIDlist):
    """
      Uses first data point of each FID to dephase each FID in list
      Returns a list of dephased FIDs
    """
    return [fid * np.exp(-1j * np.angle(fid[0])) for fid in FIDlist]


class CovarianceEstimationError(Exception):
    """Raised when coil covariance can't be estimated from the data.
    """
    pass


def estimate_noise_cov(FIDs, prop=.1):
    """Estimate noise covariance from the noise points at the end of a FID

    Covariance is calculated across the last dimension of the input FID array.

    :param FIDs: Array of FIDs ((optional) repeats * timepoints x N coils/FIDS)
    :type FIDs: numpy.array
    :param prop: proportion of FID used to estimate covariance, defaults to .1
    :type prop: float, optional
    :return: covariance matrix
    :rtype: np.ndarray
    """
    # Estimate noise covariance
    start = int((1 - prop) * FIDs.shape[-2])
    selected_points = FIDs[..., start:, :].reshape(-1, FIDs.shape[-1])
    # Raise warning if not enough samples
    if selected_points.shape[0] < 10 * FIDs.shape[-1]:
        raise CovarianceEstimationError(
            f'You have far too few points ({selected_points.shape[0]}) '
            f'to calculate an {FIDs.shape[-1]} element covariance.'
            ' Disable prewhitening.')
    elif selected_points.shape[0] < 1E5:
        print(
            'You may not have enough samples to accurately estimate the noise covariance, '
            '10^5 samples recommended.')
    return np.cov(selected_points, rowvar=False)


def prewhiten(FIDlist, prop=.1, C=None):
    """Pre-whiten data (remove inter-coil covariances)

    Either using supplied covariance matrix (C)
    or by estimating covariance from noise points at the end of the FID.

    :param FIDs: Array of FIDs ((optional) repeats * timepoints x N coils/FIDS)
    :type FIDs: numpy.array
    :param prop: proportion of FID used to estimate covariance, defaults to .1
    :type prop: float, optional
    :param C: Supplied covariance matrix, defaults to None
    :type C: np.array, optional
    :return FIDs: Pre-whitened data
    :rtype: numpy.array
    :return W: Pre-whitening matrix
    :rtype: numpy.array
    :return C: Estimated covariance matrix
    :rtype: numpy.array
    """
    FIDs = np.asarray(FIDlist, dtype=complex)
    if C is None:
        C = estimate_noise_cov(FIDs, prop)

    D, V = np.linalg.eigh(C, UPLO='U')  # UPLO = 'U' to match matlab implementation
    # Pre-whitening matrix
    W = V @ np.diag(1 / np.sqrt(D))
    # Pre-whitened data
    FIDs = FIDs @ W
    return FIDs, W, C


def svd_reduce(FIDlist, W=None, C=None, return_alpha=False):
    """Combine different channels by the wSVD method

    Based on C.T. Rodgers and M.D. Robson, Magn Reson Med 63:881-891, 2010

    :param FIDlist: Array of FIDs (timepoints x N coils)
    :type FIDlist: np.array or list
    :param W: Pre-whitening matrix, defaults to None
    :type W: np.array, optional
    :param C: Coil covariance matrix, defaults to None
    :type C: np.array, optional
    :param return_alpha: Optionally return the coil combination weights, defaults to False
    :type return_alpha: bool, optional
    :return FID: Coil combined FID
    :rtype: np.array
    :return alpha: Coil combination weights incorporating prewhitening, only returned if return_alpha=True
    :rtype: np.array
    :return scaledAmps: Coil combination weights (no prewhitening), only returned if return_alpha=True
    :rtype: np.array
    """
    FIDs  = np.asarray(FIDlist)
    U, S, V = np.linalg.svd(FIDs, full_matrices=False)
    # U,S,V = svds(FIDs,k=1)  #this is much faster but fails for two coil case

    # nCoils = FIDs.shape[1]
    # svdQuality = ((S[0] / np.linalg.norm(S)) * np.sqrt(nCoils) - 1) / (np.sqrt(nCoils) - 1)

    # get arbitrary amplitude
    iW = np.eye(FIDs.shape[1])
    if W is not None:
        iW = np.linalg.inv(W)
    amp = V[0, :] @ iW

    # arbitrary scaling here such that the first coil weight is real and positive
    svdRescale = np.linalg.norm(amp) * (amp[0] / np.abs(amp[0]))

    # combined channels
    FID = U[:, 0] * S[0] * svdRescale

    if return_alpha:
        # sensitivities per channel
        # alpha = amp/svdRescale # equivalent to svdCoilAmplitudes in matlab implementation

        # Instead incorporate the effect of the whitening stage as well.
        if C is None:
            C = np.eye(FIDs.shape[1])
        scaledAmps = (amp / svdRescale).conj().T
        alpha = np.linalg.inv(C) @ scaledAmps * svdRescale.conj() * svdRescale
        return FID, alpha, scaledAmps
    else:
        return FID


def weightedCombination(FIDlist, weights):
    """
    Combine different FIDS with different complex weights

    Parameters:
    -----------
    FIDlist      : list of FIDs
    weights      : complex weights

    Returns:
    --------
    array-like (FID)
    """
    if isinstance(FIDlist, list):
        FIDlist  = np.asarray(FIDlist)
    if isinstance(weights, list):
        weights = np.asarray(weights)
    # combine channels
    FID = np.sum(FIDlist * weights[None, :], axis=1)

    return FID


def combine_FIDs(FIDlist, method, do_prewhiten=False, do_dephase=False, weights=None, cov=None):
    """Combine FIDs (either from multiple coils or multiple averages)

    :param FIDlist: list of FIDs or array with time dimension first
    :type FIDlist: list
    :param method: one of 'mean', 'svd', 'svd_weights', 'weighted'
    :type method: str
    :param do_prewhiten: If true noise whitening is performed before combination, defaults to False
    :type do_prewhiten: bool, optional
    :param do_dephase: Phase is removed before combination, defaults to False
    :type do_dephase: bool, optional
    :param weights: Combine using supplied complex weights,  method must = weighted, defaults to None
    :type weights: list or np.array, optional
    :param cov: covariance matrix for noise correlation between FIDs, defaults to None
    :type cov: np.ndarray, optional
    :return: Combined FID signal
    :rtype: numpy.array
    """

    if isinstance(FIDlist, list):
        FIDlist = np.asarray(FIDlist).T

    # Pre-whitening
    pre_w_mat = None
    if do_prewhiten:
        FIDlist, pre_w_mat, cov = prewhiten(FIDlist, C=cov)

    # Dephasing
    if do_dephase:
        FIDlist   = dephase(FIDlist)

    # Combining channels
    if method == 'mean':
        return np.mean(FIDlist, axis=-1).T
    elif method == 'svd':
        return svd_reduce(FIDlist, pre_w_mat)
    elif method == 'svd_weights':
        return svd_reduce(FIDlist, pre_w_mat, cov, return_alpha=True)
    elif method == 'weighted':
        return weightedCombination(FIDlist, weights)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Should be either 'mean', 'svd', 'svd_weights', or 'weighted'.")


def combine_FIDs_report(inFIDs,
                        outFID,
                        bw,
                        cf,
                        nucleus='1H',
                        ncha=2,
                        ppmlim=(0.0, 6.0),
                        method='not specified',
                        dim=None,
                        html=None):
    """ Take list of FIDs that are passed to combine and output

    If uncombined data it will display ncha channels (default 2).
    """
    from fsl_mrs.core import MRS
    import plotly.graph_objects as go
    from fsl_mrs.utils.preproc.reporting import plotStyles, plotAxesStyle
    from matplotlib.pyplot import cm

    def toMRSobj(fid):
        return MRS(FID=fid, cf=cf, bw=bw, nucleus=nucleus)

    # Assemble data to plot
    toPlotIn = []
    colourVecIn = []
    legendIn = []
    if isinstance(inFIDs, list):
        for idx, fid in enumerate(inFIDs):
            if inFIDs[0].ndim > 1 and dim == 'DIM_COIL':
                toPlotIn.extend([toMRSobj(f) for f in fid[:, :ncha].T])
                colourVecIn.extend([idx / len(inFIDs)] * ncha)
                legendIn.extend([f'FID #{idx}: CHA #{jdx}' for jdx in range(ncha)])
            else:
                toPlotIn.append(toMRSobj(fid))
                colourVecIn.append(idx / len(inFIDs))
                legendIn.append(f'FID #{idx}')
    else:
        toPlotIn.extend([toMRSobj(f) for f in inFIDs[:, :ncha].T])
        colourVecIn.extend([float(jdx) / ncha for jdx in range(ncha)])
        if inFIDs.ndim > 1 and dim == 'DIM_COIL':
            legendIn.extend([f'FID #0: CHA #{jdx}' for jdx in range(ncha)])
        elif inFIDs.ndim > 1:
            legendIn.extend([f'FID #{jdx}' for jdx in range(ncha)])

    toPlotOut = []
    legendOut = []
    if outFID.ndim > 1:
        toPlotOut.extend([toMRSobj(f) for f in outFID[:, :ncha].T])
        legendOut.extend([f'Combined: CHA #{jdx}' for jdx in range(ncha)])
    else:
        toPlotOut.append(toMRSobj(outFID))
        legendOut.append('Combined')

    def addline(fig, mrs, lim, name, linestyle):
        trace = go.Scatter(x=mrs.getAxes(ppmlim=lim),
                           y=np.real(mrs.get_spec(ppmlim=lim)),
                           mode='lines',
                           name=name,
                           line=linestyle)
        return fig.add_trace(trace)

    lines, colors, _ = plotStyles()
    colors = cm.Spectral(np.array(colourVecIn).ravel())

    fig = go.Figure()
    for idx, fid in enumerate(toPlotIn):
        cval = np.round(255 * colors[idx, :])
        linetmp = {'color': f'rgb({cval[0]},{cval[1]},{cval[2]})', 'width': 1}
        fig = addline(fig, fid, ppmlim, legendIn[idx], linetmp)

    for idx, fid in enumerate(toPlotOut):
        fig = addline(fig, fid, ppmlim, legendOut[idx], lines['blk'])
    plotAxesStyle(fig, ppmlim, 'Combined')

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

        opName = 'Combination'
        timestr = datetime.now().strftime("%H:%M:%S")
        datestr = datetime.now().strftime("%d/%m/%Y")
        headerinfo = 'Report for fsl_mrs.utils.preproc.combine.combine_FIDs.\n'\
            + f'Generated at {timestr} on {datestr}.'
        # Figures
        div = plot(fig, output_type='div', include_plotlyjs='cdn')
        figurelist = [figgroup(fig=div,
                               name='',
                               foretext=f'Combination of spectra. Method = {method}',
                               afttext='')]

        singleReport(htmlfile, opName, headerinfo, figurelist)
        return fig
    else:
        return fig

#  Matplotlib
# def combine_FIDs_report(inFIDs,outFID,hdr,fileout=None,ncha=2):
#     """ Take list of FIDs that are passed to combine and output

#     If uncombined data it will display ncha channels (default 2).
#     """
#     from matplotlib import pyplot as plt
#     from fsl_mrs.core import MRS
#     from fsl_mrs.utils.plotting import styleSpectrumAxes

#     toMRSobj = lambda fid : MRS(FID=fid,header=hdr)

#     # Assemble data to plot
#     toPlotIn = []
#     if isinstance(inFIDs,list):
#         for fid in inFIDs:
#             if inFIDs[0].ndim>1:
#                 toPlotIn.extend([toMRSobj(f) for f in fid[:,:ncha].T])
#             else:
#                 toPlotIn.append(toMRSobj(fid))

#     elif inFIDs.ndim>1:
#         toPlotIn.extend([toMRSobj(f) for f in inFIDs[:,:ncha].T])

#     toPlotOut = []
#     if outFID.ndim>1:
#         toPlotOut.extend([toMRSobj(f) for f in outFID[:,:ncha].T])
#     else:
#         toPlotOut.append(toMRSobj(outFID))

#     ppmlim = (0.0,6.0)
#     ax = plt.gca()
#     colourvec = [[i/len(toPlotIn)]*ncha for i in range(len(inFIDs))]
#     colors = plt.cm.Spectral(np.array(colourvec).ravel())
#     style = ['--']*len(colors)
#     ax.set_prop_cycle(color =colors,linestyle=style)
#     for fid in toPlotIn:
#         ax.plot(fid.getAxes(ppmlim=ppmlim),np.real(fid.get_spec(ppmlim=ppmlim)))
#     for fid in toPlotOut:
#         ax.plot(fid.getAxes(ppmlim=ppmlim),np.real(fid.get_spec(ppmlim=ppmlim)),'k-')
#     styleSpectrumAxes(ax)
#     plt.show()
