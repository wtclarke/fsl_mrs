'''
FSL-MRS test script

Test core basis class.

Copyright Will Clarke, University of Oxford, 2021
'''
from pathlib import Path

import numpy as np
import pytest

import fsl_mrs.utils.mrs_io.fsl_io as fslio
from fsl_mrs.core import basis as basis_mod
from fsl_mrs.utils.mrs_io.lcm_io import EncryptedBasisError

testsPath = Path(__file__).parent
fsl_basis_path = testsPath / 'testdata' / 'mrs_io' / 'basisset_FSL'
lcm_basis_path = testsPath / 'testdata' / 'mrs_io' / 'basisset_LCModel.BASIS'
lcm_basis_official = testsPath / 'testdata' / 'mrs_io' / 'gamma_press_te20_3t_v1.basis'
lcm_basis_encrypted = testsPath / 'testdata' / 'mrs_io' / 'press_te25_3t_v3.basis'


def test_load_and_constructors():

    basis, names, bhdrs = fslio.readFSLBasisFiles(fsl_basis_path)
    manual = basis_mod.Basis(basis, names, bhdrs)

    from_file = basis_mod.Basis.from_file(fsl_basis_path)

    assert isinstance(manual, basis_mod.Basis)
    assert isinstance(from_file, basis_mod.Basis)

    assert np.allclose(manual.original_basis_array, from_file.original_basis_array)

    assert manual.cf == 123.2189956
    assert manual.original_bw == 4000.0
    assert manual.original_dwell == 0.00025
    assert manual.original_points == 2048
    assert manual.n_metabs == 21
    assert manual.original_basis_array.shape == (2048, 21)
    assert manual.original_basis_array.dtype == complex

    assert from_file.cf == 123.2189956
    assert from_file.original_bw == 4000.0
    assert from_file.original_dwell == 0.00025
    assert from_file.original_points == 2048
    assert from_file.n_metabs == 21
    assert from_file.original_basis_array.shape == (2048, 21)
    assert from_file.original_basis_array.dtype == complex

    # Test 1D FID
    manual1D = basis_mod.Basis(basis[:, 0], [names[0]], [bhdrs[0]])
    assert manual1D.cf == 123.2189956
    assert manual1D.original_bw == 4000.0
    assert manual1D.original_dwell == 0.00025
    assert manual1D.original_points == 2048
    assert manual1D.n_metabs == 1
    assert manual1D.original_basis_array.shape == (2048, 1)
    assert manual1D.original_basis_array.dtype == complex

    # Test transposed array
    manual = basis_mod.Basis(basis.T, names, bhdrs)
    assert manual.cf == 123.2189956
    assert manual.original_bw == 4000.0
    assert manual.original_dwell == 0.00025
    assert manual.original_points == 2048
    assert manual.n_metabs == 21
    assert manual.original_basis_array.shape == (2048, 21)
    assert manual.original_basis_array.dtype == complex


def test_nuc_in_hdr():
    basis, names, bhdrs = fslio.readFSLBasisFiles(fsl_basis_path)
    manual = basis_mod.Basis(basis, names, bhdrs)

    assert manual.nucleus == "1H"

    for hdr in bhdrs:
        hdr['nucleus'] = "1H"
    manual = basis_mod.Basis(basis, names, bhdrs)
    assert manual.nucleus == "1H"

    for hdr in bhdrs:
        hdr['nucleus'] = "31P"
    manual = basis_mod.Basis(basis, names, bhdrs)
    assert manual.nucleus == "31P"

    with pytest.raises(ValueError):
        for hdr in bhdrs:
            hdr['nucleus'] = "not_a_nuc"
        manual = basis_mod.Basis(basis, names, bhdrs)


def test_lcm_load():
    # Test LCModel basis load
    lcm = basis_mod.Basis.from_file(lcm_basis_path)
    assert lcm.cf == 123.261703
    assert np.isclose(lcm.original_bw, 4000.0)
    assert np.isclose(lcm.original_dwell, 0.00025)
    assert lcm.original_points == 4096
    assert lcm.n_metabs == 21
    assert lcm.original_basis_array.shape == (4096, 21)
    assert lcm.original_basis_array.dtype == complex

    lcm = basis_mod.Basis.from_file(lcm_basis_official)
    assert lcm.cf == 123.199997
    assert lcm.original_points == 9872
    assert lcm.n_metabs == 17
    assert lcm.original_basis_array.shape == (9872, 17)
    assert lcm.original_basis_array.dtype == complex

    with pytest.raises(EncryptedBasisError) as exc_info:
        lcm = basis_mod.Basis.from_file(lcm_basis_encrypted)
    assert exc_info.type is EncryptedBasisError
    assert exc_info.value.args[0] ==\
        'This is an encrypted LCModel basis. '\
        'Please consider using a basis specific to your sequence.'


def test_save(tmp_path):
    # Test saving of basis set

    original = basis_mod.Basis.from_file(fsl_basis_path)
    original.save(tmp_path / 'new_basis')

    saved = basis_mod.Basis.from_file(tmp_path / 'new_basis')

    assert np.allclose(original.original_basis_array, saved.original_basis_array)


def test_formatting():
    original = basis_mod.Basis.from_file(fsl_basis_path)

    with pytest.raises(basis_mod.BasisHasInsufficentCoverage) as exc_info:
        original.get_formatted_basis(2000, 2048)
    assert exc_info.type is basis_mod.BasisHasInsufficentCoverage
    assert exc_info.value.args[0] == 'The basis spectra covers too little time. '\
                                     'Please reduce the dwelltime, number of points or pad this basis.'

    basis = original.get_formatted_basis(2000, 1024)
    assert basis.shape == (1024, 21)

    basis = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'])
    assert basis.shape == (1024, 19)

    basis = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'], scale_factor=100)
    assert np.isclose(np.linalg.norm(np.mean(basis, axis=1)), 100)

    names = original.get_formatted_names(ignore=['Ins', 'Cr'])
    assert 'Ins' not in names
    assert 'Cr' not in names

    basis = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'], scale_factor=100, indept_scale=['Mac'])
    index = original.get_formatted_names(ignore=['Ins', 'Cr']).index('Mac')
    assert np.isclose(np.linalg.norm(np.mean(np.delete(basis, index, axis=1), axis=1)), 100)
    assert np.isclose(np.linalg.norm(basis[:, index]), 100)

    # Test rescale
    rescale = original.get_rescale_values(2000, 1024, ignore=['Ins', 'Cr'], scale_factor=100)
    no_scale = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'])
    assert np.isclose(np.linalg.norm(np.mean(no_scale * rescale[0], axis=1)), 100)


def test_formatting_linear_interp():
    original = basis_mod.Basis.from_file(fsl_basis_path)
    original.use_fourier_interp = False

    with pytest.raises(basis_mod.BasisHasInsufficentCoverage) as exc_info:
        original.get_formatted_basis(2000, 2048)
    assert exc_info.type is basis_mod.BasisHasInsufficentCoverage
    assert exc_info.value.args[0] == 'The basis spectra covers too little time. '\
                                     'Please reduce the dwelltime, number of points or pad this basis.'

    basis = original.get_formatted_basis(2000, 1024)
    assert basis.shape == (1024, 21)

    basis = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'])
    assert basis.shape == (1024, 19)

    basis = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'], scale_factor=100)
    assert np.isclose(np.linalg.norm(np.mean(basis, axis=1)), 100)

    names = original.get_formatted_names(ignore=['Ins', 'Cr'])
    assert 'Ins' not in names
    assert 'Cr' not in names

    basis = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'], scale_factor=100, indept_scale=['Mac'])
    index = original.get_formatted_names(ignore=['Ins', 'Cr']).index('Mac')
    assert np.isclose(np.linalg.norm(np.mean(np.delete(basis, index, axis=1), axis=1)), 100)
    assert np.isclose(np.linalg.norm(basis[:, index]), 100)

    # Test rescale
    rescale = original.get_rescale_values(2000, 1024, ignore=['Ins', 'Cr'], scale_factor=100)
    no_scale = original.get_formatted_basis(2000, 1024, ignore=['Ins', 'Cr'])
    assert np.isclose(np.linalg.norm(np.mean(no_scale * rescale[0], axis=1)), 100)


def test_add_fid():
    original = basis_mod.Basis.from_file(fsl_basis_path)

    original.add_fid_to_basis(np.ones((original.original_points,), dtype=complex), 'test', width=5.0)

    assert 'test' in original.names
    assert original.basis_fwhm[-1] == 5.0
    assert original.n_metabs == 22


def test_remove_fid():
    original = basis_mod.Basis.from_file(fsl_basis_path)

    original.remove_fid_from_basis('NAA')

    assert len(original.names) == 20
    assert 'NAA' not in original.names
    assert original.n_metabs == 20
    assert len(original.basis_fwhm) == 20


def test_add_peak():
    original = basis_mod.Basis.from_file(fsl_basis_path)

    original.add_peak(0, 1, 'test', gamma=1, sigma=10)

    assert 'test' in original.names
    assert original.n_metabs == 22


def test_add_mm():
    original_no_mod = basis_mod.Basis.from_file(fsl_basis_path)
    orig_names = original_no_mod.names
    original = basis_mod.Basis.from_file(fsl_basis_path)
    original.add_default_MM_peaks(gamma=10, sigma=10)
    assert original.n_metabs == 26
    assert original.names == orig_names + ['MM09', 'MM12', 'MM14', 'MM17', 'MM21']

    original.add_water_peak(gamma=10, sigma=10, name='myh20')
    assert original.n_metabs == 27
    assert original.names == orig_names + ['MM09', 'MM12', 'MM14', 'MM17', 'MM21', 'myh20']


def test_update():
    original = basis_mod.Basis.from_file(fsl_basis_path)

    original.update_fid(np.ones((original.original_points,), complex), 'Mac')

    index = original.names.index('Mac')
    assert np.allclose(original._raw_fids[:, index], np.ones((original.original_points,), complex))


def test_update_nucleus():
    original = basis_mod.Basis.from_file(fsl_basis_path)

    original.nucleus = "31P"
    assert original.nucleus == "31P"

    with pytest.raises(
            ValueError,
            match="Nucleus string must be in format of '1H', '31P', '23Na' etc."):
        original.nucleus = "1234RR"
