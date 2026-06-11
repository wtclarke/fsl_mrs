"""Routines for spectral alignment using cross correlation

see fsl_mrs.utils.preproc.align for spectral registration approaches

Copyright W Clarke, University of Oxford, 2025.
"""

import numpy as np
from scipy.signal import correlate, correlation_lags

from fsl_mrs.utils.preproc import freqshift, pad, apodize, applyPhase
from fsl_mrs.utils.misc import FIDToSpec
from nifti_mrs.axes import Axes


def xcorr_align(
        fids_in: np.typing.NDArray[np.complexfloating],
        axes: Axes,
        target: np.ndarray | None = None,
        zpad_factor: int = 1,
        apodize_hz: float = 0,
        ppmlim: None | tuple[float, float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align FIDs using cross correlation of complex spectrum

    By default aligns to the mean of all FIDs. Optionally pass a target

    :param fids_in: Array of FIDs, transients x timedomain
    :type fids_in: numpy.ndarray
    :param axes: Axes object
    :type axes: nifti_mrs.axes.Axes
    :param target: Alignment target FID, defaults to None. Zero-pad will be applied to target
    :type target: np.ndarray | None, optional
    :param zpad_factor: Zeropadding applied to fid before xcorrelation, defaults to 1, 0 disables
    :type zpad_factor: int
    :param apodize_hz: Apodization to apply to FIDs (not target), defaults to 0
    :type apodize_hz: float, optional
    :param ppmlim: Run alignment over limited ppm range, defaults to None
    :type ppmlim: None | tuple[float, float], optional
    :return: Returns shifted FIDs, the shift in hertz and phase in radians
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    def zpad(x):
        return pad(x, fids_in.shape[1] * zpad_factor, 'last')

    padded_axes = Axes(npoints=zpad(fids_in[0]).shape[0],
                       ResonantNucleus=axes.ResonantNucleus,
                       SpectrometerFrequency=axes.SpectrometerFrequency,
                       dwelltime=axes.dwelltime)
    indices = padded_axes.ppmShiftIndices(ppmlim)

    def prep_spec(x):
        x = zpad(x)
        x = apodize(
            x,
            padded_axes.timeAxis,
            apodize_hz)
        return FIDToSpec(x)[indices]

    # If the target is not defined, use the average of the input FIDs
    if target is None:
        target = prep_spec(fids_in.mean(axis=0))
    else:
        if target.size != fids_in.shape[1]:
            raise ValueError(f'Shape of target {target.size} must match input {fids_in.shape[1]}.')
        target = FIDToSpec(zpad(target))[indices]

    shifts = []
    phases = []
    for fid in fids_in:
        prepped = prep_spec(fid)
        xc = correlate(prepped, target, mode='same')
        max_index = np.argmax(np.abs(xc))
        lag = correlation_lags(len(prepped), len(target), mode='same')[max_index]
        shifts.append(lag)
        phases.append(-np.angle(xc[max_index]))

    shifts = np.asarray(shifts)
    phases = np.asarray(phases)

    # Calculate shifts in Hz
    shifts_hz = - shifts * np.diff(padded_axes.frequencyAxis[:2])

    # Apply correction
    def correct(x, shift, phase):
        return applyPhase(
            freqshift(x, axes, shift),
            phase)

    corrected = np.stack([
            correct(fid, shi, phs) for fid, shi, phs in zip(fids_in, shifts_hz, phases)])

    return corrected, shifts_hz, phases
