# dyn_based_proc.py - Module containing processing functions based on dynamic fitting
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2021 University of Oxford
# SHBASECOPYRIGHT
import os.path as op

import numpy as np

from fsl_mrs.dynamic import dynMRS
from fsl_mrs.utils import preproc as proc
from fsl_mrs.core import NIFTI_MRS
from fsl_mrs.utils.preproc.align import phase_freq_align_report
from fsl_mrs.utils.preproc.nifti_mrs_proc import update_processing_prov

config_file_path = op.dirname(__file__)


def align_by_dynamic_fit(data, basis, fitargs={}):
    """Phase and frequency alignment based on dynamic fitting

    This function performs phase and frequency alignment based on the fsl-mrs
    dynamic fitting tool. All parameters are kept constant except eps (frequency)
    and phi0 (zero-order phase).

    Only one higher encoding dimension can be handled currently.

    A single basis may be used or a list with a size equal to the size of the aligned
    dimension.

    The data should be approximately aligned before using this function.

    :param data: NIfTI-MRS object with a single dimension to align along
    :type data: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param basis: Basis to fit, or list of basis matching the dimension to align.
    :type basis: fsl_mrs.core.basis.Basis or list
    :param fitargs: kw arguments to pass to fitting, defaults to {}
    :type fitargs: dict, optional
    :return: Tuple with the aligned data, shift, and phase
    :rtype: tuple
    """
    if not isinstance(data, NIFTI_MRS):
        raise TypeError('Data must be a NIFTI_MRS object.')
    if data.ndim < 5:
        raise ValueError('Data must have at leaset one higher diemnsion.')
    if not np.isclose(np.prod(data.shape[4:]), data.shape[4:]).any():
        raise ValueError('This function can only handle one non-singleton higher dimension.')

    if not isinstance(basis, list) or len(basis) == 1:
        mrslist = data.mrs(basis=basis)
    elif len(basis) == len(data.mrs()):
        mrslist = data.mrs()
        for mrs, bb in zip(mrslist, basis):
            mrs.basis = bb
    else:
        raise TypeError('basis must either be a single Basis object or a list the length of the alignment dim.')

    tval = np.arange(0, len(mrslist))

    dyn = dynMRS(
        mrslist,
        tval,
        config_file=op.join(config_file_path, 'align_model.py'),
        **fitargs)

    init = dyn.initialise(indiv_init='mean', verbose=False)
    dyn_res = dyn.fit(init=init, verbose=False)

    def correctfid(fid, eps, phi):
        hz = eps * 1 / (2 * np.pi)
        fid_shift = proc.freqshift(fid, data.dwelltime, hz)
        fid_phased = proc.applyPhase(fid_shift, phi)
        return fid_phased

    eps = dyn_res.dataframe_mapped.eps_0.to_numpy()
    phi = dyn_res.dataframe_mapped.Phi_0_0.to_numpy()

    aligned_obj = data.copy()
    generator = data.iterate_over_dims()
    for (dd, idx), ei, pi in zip(generator, eps, phi):
        aligned_obj[idx] = correctfid(dd, ei, pi)

    # Update processing prov
    processing_info = f'{__name__}.align_by_dynamic_fit, '
    processing_info += f'fitargs={fitargs}.'

    update_processing_prov(aligned_obj, 'Frequency and phase correction', processing_info)

    return aligned_obj, eps, phi


def align_by_dynamic_fit_report(indata, aligned_data, eps, phi, ppmlim=(0.0, 4.2), html=None):
    """Report for dynamic fitting alignment

    :param indata: NIfTI-MRS before alignment
    :type indata: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param aligned_data: NIfTI-MRS after alignment
    :type aligned_data: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param eps: Shifts applied
    :type eps: list
    :param phi: Phases applied
    :type phi: list
    :param ppmlim: PPM limit, defaults to (0.0, 4.2)
    :type ppmlim: tuple, optional
    :param html: Path to html to write, if None nothing writen. Defaults to None
    :type html: Str, optional
    :return: tuple of figures
    :rtype: tuple
    """

    inFIDs = np.asarray([mrs.FID for mrs in indata.mrs()])
    outFIDs = np.asarray([mrs.FID for mrs in aligned_data.mrs()])

    bw = indata.bandwidth
    cf = indata.spectrometer_frequency[0] * 1E6
    nucleus = indata.nucleus[0]

    return phase_freq_align_report(
        inFIDs,
        outFIDs,
        phi,
        eps,
        bw,
        cf,
        nucleus=nucleus,
        ppmlim=ppmlim,
        html=html)
