'''mrsi.py - Module containing preprocessing functions for mrsi

Author: William Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 20123 University of Oxford
SHBASECOPYRIGHT
'''

import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize

from fsl.data.image import Image

from fsl_mrs.utils.preproc import freqshift, pad
from fsl_mrs.utils.misc import FIDToSpec


def mrsi_phase_corr(data, mask=None, ppmlim=None):
    """Phase correct MRSI data by maximising the real part across limits.

    :param data: MRSI data
    :type data: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param mask: Mask to select certain fids to include, defaults to None
    :type mask: fsl.data.image.Image, optional
    :param ppmlim: ppm limits over which to run phase algorithm, defaults to None
    :type ppmlim: tuple, optional
    :return: Returns phased MRSI data and Image containing phases applied in degrees
    :rtype: NIFTI_MRS, Image
    """
    if mask is None:
        mask = np.ones(data.shape[:3]).astype(bool)
    else:
        mask = mask[:].astype(bool)

    if data.ndim > 4:
        limits = data.mrs()[0].mrs_from_average().ppmlim_to_range(ppmlim, shift=True)
    else:
        limits = data.mrs().mrs_from_average().ppmlim_to_range(ppmlim, shift=True)

    phs_array = np.zeros(data.shape[:3] + data.shape[4:])
    phs = np.zeros(data.shape[:3])
    out = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=False):
        dd[mask, :], phs[mask] = phase_corr_max_real(
            dd[mask, :],
            limits=limits)

        out[idx] = dd
        phs_array[idx[:3] + idx[4:]] = phs * 180 / np.pi

    return out, Image(phs_array, xform=data.voxToWorldMat)


def phase_corr_max_real(fids, limits=None):
    """Phase correction of multiple FIDs based on maximising the real part fo the spectrum
    Optionally define limits between which to maximise.

    :param fids: list of FIDs
    :type fids: list or np.array
    :param limits: limits over which to maximise real part, index of array, defaults to None
    :type limits: tuple, optional
    :return: Phased FIDs, array of applied phases
    :rtype: (np.array, np.array)
    """

    if limits is None:
        limits = (0, fids.shape[1])

    phases = []
    for fid in fids:
        def phase_fun(x):
            return -np.sum(FIDToSpec(fid * np.exp(1j * x))[limits[0]:limits[1]].real)
        phases.append(minimize(phase_fun, 1E-3).x[0])  # Init with non-zero

    return np.stack([fid * np.exp(1j * x) for fid, x in zip(fids, phases)]), np.asarray(phases)


def mrsi_freq_align(data, mask=None, zpad_factor=1):
    """Frequency align MRSI data uing cross correlation to mean FID.

    :param data: MRSI data
    :type data: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param mask: Mask to select certain fids to include, defaults to None
    :type mask: fsl.data.image.Image, optional
    :param zpad_factor: Zeropadding applied to fid before xcorrelation, defaults to 1, 0 disables
    :type zpad_factor: int, optional
    :return: Returns shifted MRSI data and Image containing shifts applied in Hz
    :rtype: NIFTI_MRS, Image
    """
    if mask is None:
        mask = np.ones(data.shape[:3]).astype(bool)
    else:
        mask = mask[:].astype(bool)

    shift_array = np.zeros(data.shape[:3] + data.shape[4:])
    shifts = np.zeros(data.shape[:3])
    out = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=False):
        dd[mask, :], shifts[mask] = xcorr_align(
            dd[mask, :],
            data.dwelltime,
            zpad_factor=zpad_factor)

        out[idx] = dd
        shift_array[idx[:3] + idx[4:]] = shifts

    return out, Image(shift_array, xform=data.voxToWorldMat)


def xcorr_align(fids_in, dwelltime, zpad_factor=1):
    """Align fids using cross correlation to mean

    :param fids_in: Array of FIDs, transients x timedomain
    :type fids_in: numpy.ndarray
    :param dwelltime: spectral dwell time (1/bandwidth) in s.
    :type dwelltime: float
    :param zpad_factor: Zeropadding applied to fid before xcorrelation, defaults to 1, 0 disables
    :type zpad_factor: int
    :returns: tuple containing shifted FIDs, shifts in Hz
    """
    def zpad(x):
        return pad(x, fids_in.shape[1] * zpad_factor, 'last')

    def prep_spec(x):
        return np.abs(FIDToSpec(zpad(x)))

    padded_mean_spec = prep_spec(fids_in.mean(axis=0))

    shifts = []
    for fid in fids_in:
        shifts.append(
            np.argmax(correlate(prep_spec(fid), padded_mean_spec, mode='same')))
    shifts = np.asarray(shifts)
    shifts -= int(fids_in.shape[1] * 0.5 * (1 + zpad_factor))
    bandwidth = 1 / dwelltime
    shifts_hz = - shifts.astype(float) * bandwidth / (fids_in.shape[1] * (1 + zpad_factor))

    return np.stack([freqshift(fid, dwelltime, s) for fid, s in zip(fids_in, shifts_hz)]), shifts_hz


def lipid_removal_l2(data, beta=1E-5, lipid_mask=None, lipid_basis=None):
    """Lipid removal using the L2-regularised 'reconstruction' approach.

    The user must specify one of lipid_mask or lipid_basis.
    Currently only implemented for MRSI data with no higher NIfTI dimensions.

    Originally published by Bilgic et al in jMRI 2014 doi: 10.1002/jmri.24365
    The code is broadly a port of the matlab demo hosted by the original authors at
    https://martinos.org/~berkin/software.html

    :param data: MRSI data
    :type data: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param beta: regularisation scaling parameter, defaults to 1E-5
    :type beta: float, optional
    :param lipid_mask: Mask to select voxels only containing lipid signals, defaults to None
    :type lipid_mask: fsl.data.image.Image, optional
    :param lipid_basis: Array of lipid FIDS, fist dim should be time, defaults to None
    :type lipid_basis: np.array, optional
    :return: Data with lipids removed
    :rtype: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    """
    if data.ndim > 4:
        raise ValueError('Data cannot have higher NIfTI dimensions.')

    # Assemble a lipid basis from masked region or the direct input
    if lipid_basis is not None:
        if lipid_basis.shape[0] != data.shape[3]:
            raise ValueError(
                'Lipid basis must have a first dimension size equal to the spectral dimension. '
                f'Current size = {lipid_basis.shape[0]}, expected = data.shape[3]')
    elif lipid_mask is not None:
        if not isinstance(lipid_mask, Image):
            raise TypeError('lipid_mask should be an fslpy Image object.')
        lipid_mask = lipid_mask[:].astype(bool)
        if not any(lipid_mask.ravel()):
            raise ValueError('Mask image must contain some selected voxels.')

        lipid_basis = data[:][lipid_mask, :].T
        assert lipid_basis.shape[0] == data.shape[3]
    else:
        raise TypeError('One of lipid_mask or lipid_basis must be specified. Both are set to None.')

    # Calculate the inverted matrix
    lipid_inv = np.linalg.inv(np.eye(lipid_basis.shape[0]) + beta * (lipid_basis @ lipid_basis.T.conj()))

    # Apply voxel-wise
    reduced_lipid_img = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        reduced_lipid_img[idx] = lipid_inv @ dd

    return reduced_lipid_img
