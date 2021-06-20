"""
Tools for manipulating FSL-MRS basis sets.
Includes conversion of LCModel basis sets to FSL-MRS JSON format

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
Copyright (C) 2021 University of Oxford
"""
from pathlib import Path

import numpy as np

from fsl_mrs.utils import mrs_io
from fsl_mrs.core import MRS
from fsl_mrs.core.basis import Basis
from fsl_mrs.utils.qc import idPeaksCalcFWHM
from fsl_mrs.utils.misc import ts_to_ts, InsufficentTimeCoverageError


class IncompatibleBasisError(Exception):
    pass


def convert_lcm_basis(path_to_basis, output_location=None):
    """Converts an existing LCModel basis set (.BASIS) file to FSL format (a directory of json files).

    The generated FSL format will only contain a subset of the information that it normaly does
    (e.g. no sequence info), and won't hold all LCModel fields either.

    :param path_to_basis: Path to existing LCModel basis
    :type path_to_basis: str or pathlib.Path
    :param output_location: Name of output directory, defaults to match input name and location
    :type output_location: str or pathlib.Path, optional.
    """
    if isinstance(path_to_basis, str):
        path_to_basis = Path(path_to_basis)

    # 1. Read LCModel basis
    basis = mrs_io.read_basis(path_to_basis)

    # 2. Write to new location
    sim_info = f'Converted from {str(path_to_basis)}'
    if output_location is None:
        basis.save(path_to_basis.stem, info_str=sim_info)
    else:
        basis.save(output_location, info_str=sim_info)


def add_basis(fid, name, cf, bw, target, scale=False, width=None, sim_info='Manually added'):
    """Add an additional basis spectrum to an existing FSL formatted basis set.

    Optionally rescale the norm of the new FID to the mean of the existing ones.

    :param fid: Fid to add
    :type fid: np.ndarray, complex
    :param name: Name of metabolite basis
    :type name: str
    :param cf: Central frequency in MHz
    :type cf: float
    :param bw: Bandwidth in Hz
    :type bw: float
    :param target: Target basis set
    :type target: str or pathlib.Path
    :param scale: Rescale the fid so its norm is the mean of the norms of the
     other basis spectra, defaults to False
    :type scale: bool, optional
    :param width: Width (fwhm in Hz) of added basis, defaults to None. If None it will be estimated.
    :type width: float, optional
    :param sim_info: String added to the meta.SimVersion field, defaults to 'Manually added'
    :type sim_info: str, optional
    """
    # 1. Check that target exists
    target = Path(target)
    if not target.is_dir():
        raise NotADirectoryError('Target must be a directory of FSL-MRS basis (json) files')

    # 2. Resample new basis to the same raster as the target
    # Can't use the central frequency as a way to align as the absolute frequency is effectively arbitrary
    target_basis = mrs_io.read_basis(target)
    target_dt = target_basis.original_dwell
    try:
        resampled_fid = ts_to_ts(fid,
                                 1 / bw,
                                 target_dt,
                                 target_basis.original_points)
    except InsufficentTimeCoverageError:
        raise IncompatibleBasisError('The new basis FID covers too little time, try padding.')

    # 3. Scale if requested
    if scale:
        norms = []
        for b in target_basis.original_basis_array.T:
            norms.append(np.linalg.norm(b))
        resampled_fid *= np.mean(norms) / np.linalg.norm(resampled_fid)

    # 4. Width calculation if needed.
    if width is None:
        mrs = MRS(FID=resampled_fid, cf=cf, bw=bw)
        mrs.check_FID(repair=True)
        width, _, _ = idPeaksCalcFWHM(mrs)

    # 5. Add to existing basis
    target_basis.add_fid_to_basis(resampled_fid, name, width=width)

    # 6. Write to json without overwriting existing files
    target_basis.save(target, info_str=sim_info)


def shift_basis(basis, name, amount):
    """Shift a particular metabolite basis spectrum on the chemical shift axis.

    :param basis: Unmodified basis set
    :type basis: fsl_mrs.core.basis.Basis
    :param name: Metabolite name to shift
    :type name: str
    :param amount: Chemical shift amount in ppm
    :type amount: float
    :return: Modified basis
    :rtype: fsl_mrs.core.basis.Basis
    """
    index = basis.names.index(name)

    original_fid = basis.original_basis_array[:, index]

    amount_in_hz = amount * basis.cf
    t = basis.original_time_axis
    shifted = original_fid * np.exp(-1j * 2 * np.pi * t * amount_in_hz)

    basis.update_fid(shifted, name)

    return basis


def rescale_basis(basis, name, target_scale=None):
    """Rescale a single basis FID in place to either the mean of the set or a particular target

    :param basis: Unmodified basis set
    :type basis: fsl_mrs.core.basis.Basis
    :param name: Metabolite name to SCALE
    :type name: str
    :param target_scale: Norm of slected basis will be scaled to this target, defaults to None = mean of other FIDs
    :type target_scale: float, optional
    :return: Basis object with new scaling
    :rtype: fsl_mrs.core.basis.Basis
    """
    index = basis.names.index(name)
    indexed_fid = basis.original_basis_array[:, index]
    all_except_index = np.delete(basis.original_basis_array, index, axis=1)

    if target_scale:
        indexed_fid *= target_scale / np.linalg.norm(indexed_fid)
    else:
        indexed_fid *= np.linalg.norm(np.mean(all_except_index, axis=1), axis=0) / np.linalg.norm(indexed_fid)

    basis.update_fid(indexed_fid, name)

    return basis


def difference_basis_sets(basis_1, basis_2, add_or_subtract='add', missing_metabs='raise'):
    """Add or subtract basis sets to form a set of difference spectra

    :param basis_1: Basis set 1
    :type basis_1: fsl_mrs.core.basis.Basis
    :param basis_2: Basis set 2
    :type basis_2: fsl_mrs.core.basis.Basis
    :param add_or_subtract: Add ('add') or subtract ('sub') basis sets, defaults to 'add'
    :type add_or_subtract: str, optional
    :param missing_metabs: Behaviour when mismatched basis sets are found.
        It 'raise' a IncompatibleBasisError is raised, if 'ignore' the mismatched
        basis will be skipped. Defaults to 'raise'
    :type missing_metabs: str, optional
    :return: Difference basis spectra
    :rtype: fsl_mrs.core.basis.Basis
    """

    if missing_metabs == 'raise':
        for name in basis_1.names:
            if name not in basis_2.names:
                raise IncompatibleBasisError(f'{name} does not occur in basis_2.')

        for name in basis_2.names:
            if name not in basis_1.names:
                raise IncompatibleBasisError(f'{name} does not occur in basis_1.')

    difference_spec = []
    names = []
    headers = []
    for b1, name in zip(basis_1.original_basis_array.T, basis_1.names):
        if name in basis_2.names:
            index = basis_2.names.index(name)
            if add_or_subtract == 'add':
                diff = b1 + basis_2.original_basis_array[:, index]
            elif add_or_subtract == 'sub':
                diff = b1 - basis_2.original_basis_array[:, index]

            difference_spec.append(diff)
            names.append(name)
            headers.append({'dwelltime': basis_2.original_dwell,
                            'bandwidth': basis_2.original_bw,
                            'centralFrequency': basis_2.cf,
                            'fwhm': basis_2.basis_fwhm[index]})

    return Basis(np.asarray(difference_spec), names, headers)
