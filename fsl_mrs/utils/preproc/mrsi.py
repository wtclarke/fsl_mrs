'''mrsi.py - Module containing preprocessing functions for mrsi

Author: William Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 20123 University of Oxford
SHBASECOPYRIGHT
'''

import numpy as np
from scipy.optimize import minimize

from fsl.data.image import Image

from fsl_mrs.utils.preproc.align_xcor import xcorr_align
from fsl_mrs.utils.preproc.shifting import freqshift_array
from fsl_mrs.utils.preproc.filtering import calc_aprox_t2decay
from fsl_mrs.utils.preproc.phasing import phasta
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.core.nifti_mrs import NIFTI_MRS
from fsl_mrs.core.basis import Basis
from fsl_mrs.utils.preproc.filtering import apodize


def mrsi_phase_corr(
        data: NIFTI_MRS,
        method: str = 'phasta',
        mask: Image | None = None,
        ppmlim: tuple[float, float] | None = None,
        apodize: str | float = "auto",
        higher_dim_index: int | slice = slice(None)) -> NIFTI_MRS:
    """Run phase correction on MRSI

    Implements either a simple maximise real over limits or LCModel's Phasta algorithm

    :param data: MRSI data to correct.
    :type data: NIFTI_MRS
    :param method: "phasta" or "max-real, defaults to 'phasta'
    :type method: str, optional
    :param mask: Image to select voxels to process, defaults to None
    :type mask: Image | None, optional
    :param ppmlim: Limit correction to ppm range, defaults to None
    :type ppmlim: tuple[float, float] | None, optional
    :param apodize: Apodization filter (in hertz), defaults to "auto"
    :type apodize: str | float, optional
    :param higher_dim_index: For phasta algorithm use a sub-set of a single higher dimension, defaults to slice(None)
    :type higher_dim_index: list[int] | slice, optional
    :return: Phase corrected MRSI
    :rtype: NIFTI_MRS
    """
    if mask is None:
        mask = np.ones(data.shape[:3]).astype(bool)
    else:
        mask = mask[:].astype(bool)

    if data.ndim > 4:
        limits = data.mrs()[0].mrs_from_average().ppmlim_to_range(ppmlim, shift=True)
    else:
        limits = data.mrs().mrs_from_average().ppmlim_to_range(ppmlim, shift=True)

    if apodize == "auto":
        apodize = calc_aprox_t2decay(
            np.moveaxis(data[:][mask, :], 1, -1).reshape(-1, data.shape[3]),
            data.dwelltime
        )
        print(f'Setting apodization filter to {apodize:0.1f} Hz.')
    elif not (isinstance(apodize, (float, int)) and apodize >= 0):
        raise ValueError('Apodize should be a value >= 0.')

    phs_array = np.zeros(data.shape[:3] + data.shape[4:])
    out = data.copy()
    for dd, idx in data.iterate_over_spatial():
        if not mask[idx[:3]]:
            continue
        if method.lower() == "max-real":
            out[idx], phs = phase_corr_max_real(
                dd,
                data.dwelltime,
                limits=limits,
                apodization=apodize)

            phs_array[idx[:3] + idx[4:]] = phs * 180 / np.pi
        elif method.lower() == "phasta":
            out[idx], phs_array[idx[:3] + idx[4:]] = phasta(
                dd,
                data.dwelltime,
                limits=limits,
                apodization=apodize,
                indices_to_use=higher_dim_index
            )
        else:
            raise ValueError('method must be "phasta" or "max-real"')

    return out, Image(phs_array, xform=data.voxToWorldMat)


def phase_corr_max_real(
        fids: np.ndarray,
        dwelltime: float,
        limits=None,
        apodization: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """Phase correction of multiple FIDs based on maximising the real part of the spectrum
    Optionally define limits between which to maximise.

    :param fids: list of FIDs
    :type fids: list or np.array
    :param dwelltime: Dwelltime (1 / spectral bandwidth)
    :type dwelltime: float
    :param limits: limits over which to maximise real part, index of array, defaults to None
    :type limits: tuple, optional
    :param apodization: Apply apodization, defaults to 0 (no apodization)
    :type apodization: float, optional
    :return: Phased FIDs, array of applied phases
    :rtype: (np.array, np.array)
    """

    data_apod = [apodize(fid, dwelltime, apodization) for fid in fids]

    limits = slice(None) if limits is None else slice(limits[0], limits[1])

    phases = []
    for fid in data_apod:
        def phase_fun(x):
            return -np.sum(FIDToSpec(fid * np.exp(1j * x))[limits].real)
        phases.append(minimize(phase_fun, 1E-3).x[0])  # Init with non-zero

    return np.stack([fid * np.exp(1j * x) for fid, x in zip(fids, phases)]), np.asarray(phases)


def mrsi_freq_align(
        data: NIFTI_MRS,
        target: None | NIFTI_MRS | Basis = None,
        basis_ignore: list[str] = [],
        mask: Image = None,
        zpad_factor: int = 1,
        apodize: str | float = "auto",
        higher_dimensions: str | int = "separate") -> tuple[NIFTI_MRS, Image]:
    """Frequency align MRSI data using cross correlation.

    Align either to mean, to a provided target, or to a basis spectrum
    A target FID must be a single FID with no higher dimensions.
    The spectra to use in a Basis target can be reduced using `basis_ignore`.

    :param data: MRSI data
    :type data: NIFTI_MRS
    :param target: Select what the target is.
        None = mean of all voxels within mask.
        If a Basis object is passed, alignment will be done to the unbroadened basis.
        If a NIFTI_MRS object is passed, alignment will be done to the passed spectrum.
        Defaults to None
    :type target: None | NIFTI_MRS | Basis, optional
    :param basis_ignore: List of basis spectra to remove from a Basis object target.
        Defaults to empty list, i.e. uses all the basis spectra
    :type basis_ignore: list[str], optional
    :param mask: If provided only voxels in mask will be aligned, defaults to None
    :type mask: Image, optional
    :param zpad_factor: Multiples of zero padding applied to FID before alignment, defaults to 1, 0 disables
    :type zpad_factor: int, optional
    :param apodize: Amount of apodization to apply in hertz, defaults to "auto" which estimates amount.
    :type apodize: str | float, optional
    :param higher_dimensions: How to handle higher dimensions.
        "separate" runs alignment on each higher index separately.
        "combine" runs alignment on all indices together.
        Passing an index (int) indicates the result of that index should be applied to all others.
        Defaults to "separate"
    :type higher_dimensions: str | int, optional
    :return: Returns shifted MRSI data and Image containing shifts applied in Hz
    :rtype: tuple[NIFTI_MRS, Image]
    """
    # Handle target
    if isinstance(target, Basis):
        target = np.sum(
            target.get_formatted_basis(
                data.bandwidth,
                data.shape[3],
                ignore=basis_ignore
            ), axis=-1)
    elif isinstance(target, NIFTI_MRS):
        if not np.isclose(target.dwelltime, data.dwelltime)\
                or not np.isclose(target.shape[3], data.shape[3]):
            raise ValueError('Target must have the same dwell time and number of points as data.')

        if np.prod(target.shape[4:]) > 1:
            raise ValueError('Target must not have any higher dimensions.')

        if target.shape[:3] != (1, 1, 1):
            raise ValueError('Target must be single voxel.')
        else:
            target = target[0, 0, 0, :]
    elif target is None:
        pass
    else:
        raise TypeError('target must be a NIFTI_MRS or Basis object, or None.')

    if mask is None:
        mask = np.ones(data.shape[:3]).astype(bool)
    else:
        mask = mask[:].astype(bool)

    # Calculate automatic apodization amount
    if apodize == "auto":
        apodize = calc_aprox_t2decay(
            np.moveaxis(data[:][mask, :], 1, -1).reshape(-1, data.shape[3]),
            data.dwelltime
        )
        print(f'Setting apodization filter to {apodize:0.1f} Hz.')
    elif not (isinstance(apodize, (float, int)) and apodize >= 0):
        raise ValueError('Apodize should be a value >= 0.')

    # Define nested function to avoid repeating lots of options for each case
    def xcorr_align_worker(dat):
        return xcorr_align(
            dat,
            data.dwelltime,
            target=target,
            zpad_factor=zpad_factor,
            apodize_hz=apodize)

    shift_array = np.zeros(data.shape[:3] + data.shape[4:])
    if higher_dimensions == "separate":
        out = data.copy()
        shifts = np.zeros(data.shape[:3])
        for dd, idx in data.iterate_over_dims(iterate_over_space=False):
            dd[mask, :], shifts[mask] = xcorr_align_worker(dd[mask, :])
            out[idx] = dd
            shift_array[idx[:3] + idx[4:]] = shifts
        return out, Image(shift_array, xform=data.voxToWorldMat)
    elif isinstance(higher_dimensions, int):
        if higher_dimensions >= data.shape[4]:
            raise ValueError('higher_dimensions index must be < data.shape[4]')
        out = data.copy()
        shifts = np.zeros(data.shape[:3])
        _, shifts[mask] = xcorr_align_worker(data[:][mask, :, higher_dimensions])
        for dd, idx in data.iterate_over_dims(iterate_over_space=False):
            out[idx] = freqshift_array(
                dd,
                data.dwelltime,
                shifts)
            shift_array[idx[:3] + idx[4:]] = shifts
        return out, Image(shift_array, xform=data.voxToWorldMat)

    elif higher_dimensions == "combine":
        out = data[:].copy()
        tmp, shifts = xcorr_align_worker(
            np.moveaxis(data[:][mask, :], 1, -1).reshape(-1, data.shape[3]))

        out[mask, :] = np.moveaxis(tmp.reshape((np.sum(mask),) + data.shape[4:] + (data.shape[3],)), -1, 1)
        if shift_array.ndim == 3:
            shift_array[mask] = shifts.reshape((np.sum(mask),) + data.shape[4:])
        else:
            shift_array[mask, :] = shifts.reshape((np.sum(mask),) + data.shape[4:])
        return NIFTI_MRS(out, header=data.header), Image(shift_array, xform=data.voxToWorldMat)
    else:
        raise ValueError('higher_dimensions must be "separate", "combine" or an integer index.')


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
    if data.ndim > 4 and np.prod(data.shape[4:]) > 1.0:
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
