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
from fsl_mrs.core.nifti_mrs import NIFTI_MRS


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

    fids, phases = phase_corr_max_real(
        data[:][mask, :],
        limits=data.mrs().mrs_from_average().ppmlim_to_range(ppmlim, shift=True))

    new_array = data[:].copy()
    new_array[mask, :] = fids

    phase_array = np.zeros(data.shape[:3])
    phase_array[mask] = phases.ravel() * 180 / np.pi

    return NIFTI_MRS(new_array, header=data.header), Image(phase_array, xform=data.voxToWorldMat)


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

    fids_shifted, shifts = xcorr_align(
        data[:][mask, :],
        data.dwelltime,
        zpad_factor=zpad_factor)

    new_array = data[:].copy()
    new_array[mask, :] = fids_shifted

    shift_array = np.zeros(data.shape[:3])
    shift_array[mask] = shifts

    return NIFTI_MRS(new_array, header=data.header), Image(shift_array, xform=data.voxToWorldMat)


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
