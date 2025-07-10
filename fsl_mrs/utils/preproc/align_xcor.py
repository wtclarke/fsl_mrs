"""Routines for spectral alignment using cross correlation

see fsl_mrs.utils.preproc.align for spectral registration approaches

Copyright W Clarke, University of Oxford, 2025.
"""

import numpy as np
from scipy.signal import correlate

from fsl_mrs.utils.preproc import freqshift, pad, apodize
from fsl_mrs.utils.misc import FIDToSpec


def xcorr_align(
        fids_in: np.ndarray,
        dwelltime: float,
        target: np.ndarray | None = None,
        zpad_factor: int = 1,
        apodize_hz: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """Align FIDs using cross correlation of magnitude spectrum

    By default aligns to the mean of all FIDs. Optionally pass a target

    :param fids_in: Array of FIDs, transients x timedomain
    :type fids_in: numpy.ndarray
    :param dwelltime: spectral dwell time (1/bandwidth) in s.
    :type dwelltime: float
    :param target: Alignment target FID, defaults to None. Zero-pad will be applied to target
    :type target: np.ndarray | None, optional
    :param zpad_factor: Zeropadding applied to fid before xcorrelation, defaults to 1, 0 disables
    :type zpad_factor: int
    :param apodize_hz: Apodization to apply to FIDs (not target), defaults to 0
    :type apodize_hz: float, optional
    :return: Returns shifted FIDs and the shift in hertz
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    def zpad(x):
        return pad(x, fids_in.shape[1] * zpad_factor, 'last')

    def prep_spec(x):
        x = zpad(x)
        x = apodize(
            x,
            dwelltime,
            apodize_hz)
        return np.abs(FIDToSpec(x))

    if target is None:
        target = prep_spec(fids_in.mean(axis=0))
    else:
        if target.size != fids_in.shape[1]:
            raise ValueError(f'Shape of target {target.size} must match input {fids_in.shape[1]}.')
        target = np.abs(FIDToSpec(zpad(target)))

    shifts = []
    for fid in fids_in:
        shifts.append(
            np.argmax(correlate(prep_spec(fid), target, mode='same')))
    shifts = np.asarray(shifts)
    shifts -= int(fids_in.shape[1] * 0.5 * (1 + zpad_factor))
    bandwidth = 1 / dwelltime
    shifts_hz = - shifts.astype(float) * bandwidth / (fids_in.shape[1] * (1 + zpad_factor))

    return np.stack([freqshift(fid, dwelltime, s) for fid, s in zip(fids_in, shifts_hz)]), shifts_hz
