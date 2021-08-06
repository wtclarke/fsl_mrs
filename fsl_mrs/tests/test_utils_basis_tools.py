'''
FSL-MRS test script

Test basis tools.

Copyright Will Clarke, University of Oxford, 2021
'''
from pathlib import Path

import pytest
import numpy as np

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils import basis_tools
from fsl_mrs.utils.mrs_io import fsl_io

testsPath = Path(__file__).parent
fsl_basis_path = testsPath / 'testdata' / 'fsl_mrs' / 'steam_basis'
lcm_basis_path = testsPath / 'testdata' / 'basis_tools' / '3T_slaser_32vespa_1250.BASIS'
extra_basis = testsPath / 'testdata' / 'basis_tools' / 'macSeed.json'


def test_convert_lcmodel(tmp_path):
    out_loc = tmp_path / 'test_basis'
    basis_tools.convert_lcm_basis(lcm_basis_path, out_loc)

    basis = mrs_io.read_basis(lcm_basis_path)
    new_basis = mrs_io.read_basis(out_loc)

    assert basis.names == new_basis.names
    assert np.allclose(basis.original_basis_array, new_basis.original_basis_array)
    assert np.isclose(basis.original_bw, new_basis.original_bw)
    assert np.isclose(basis.cf, new_basis.cf)


def test_add_basis():
    basis = mrs_io.read_basis(fsl_basis_path)

    mac_in = fsl_io.readJSON(extra_basis)
    fid = np.asarray(mac_in['basis_re']) + 1j * np.asarray(mac_in['basis_im'])
    cf = mac_in['basis_centre']
    bw = 1 / mac_in['basis_dwell']

    with pytest.raises(basis_tools.IncompatibleBasisError) as exc_info:
        basis_tools.add_basis(fid, 'mac2', cf, bw, basis)

    assert exc_info.type is basis_tools.IncompatibleBasisError
    assert exc_info.value.args[0] == "The new basis FID covers too little time, try padding."

    new_basis = basis_tools.add_basis(fid, 'mac1', cf, bw, basis, pad=True)
    index = new_basis.names.index('mac1')
    assert 'mac1' in new_basis.names

    fid_pad = np.pad(fid, (0, fid.size))
    new_basis = basis_tools.add_basis(fid_pad, 'mac2', cf, bw, basis)
    index = new_basis.names.index('mac2')
    assert 'mac2' in new_basis.names
    assert np.allclose(new_basis._raw_fids[:, index], fid_pad[0::2])

    new_basis = basis_tools.add_basis(fid_pad, 'mac3', cf, bw, basis, scale=True, width=10)
    index = new_basis.names.index('mac3')
    assert 'mac3' in new_basis.names
    assert new_basis.basis_fwhm[index] == 10

    new_basis = basis_tools.add_basis(fid_pad, 'mac4', cf, bw, basis, width=10, conj=True)
    index = new_basis.names.index('mac4')
    assert 'mac4' in new_basis.names
    assert np.allclose(new_basis._raw_fids[:, index], fid_pad[0::2].conj())


def test_shift():
    basis = mrs_io.read_basis(fsl_basis_path)

    shifted = basis_tools.shift_basis(mrs_io.read_basis(fsl_basis_path), 'NAA', 1.0)

    index = basis.names.index('NAA')
    amount_in_hz = 1.0 * basis.cf
    t = basis.original_time_axis
    t -= t[0]
    shifted_fid = basis.original_basis_array[:, index] * np.exp(-1j * 2 * np.pi * t * amount_in_hz)
    assert np.allclose(shifted.original_basis_array[:, index], shifted_fid)


def test_rescale():
    basis = mrs_io.read_basis(fsl_basis_path)

    index = basis.names.index('Mac')
    indexed_fid = basis.original_basis_array[:, index]
    original_scale = np.linalg.norm(indexed_fid)

    basis = basis_tools.rescale_basis(basis, 'Mac')
    indexed_fid = basis.original_basis_array[:, index]
    new_scale = np.linalg.norm(indexed_fid)
    assert new_scale != original_scale

    basis = basis_tools.rescale_basis(basis, 'Mac', target_scale=1.0)
    indexed_fid = basis.original_basis_array[:, index]
    new_scale = np.linalg.norm(indexed_fid)
    assert np.isclose(new_scale, 1.0)


basis_on = testsPath / 'testdata' / 'basis_tools' / 'low_res_off'
basis_off = testsPath / 'testdata' / 'basis_tools' / 'low_res_on'


def test_add_sub():
    basis_1 = mrs_io.read_basis(basis_off)
    basis_2 = mrs_io.read_basis(basis_on)

    # Test addition
    new = basis_tools.difference_basis_sets(basis_1, basis_2)
    assert np.allclose(
        new.original_basis_array,
        basis_1.original_basis_array + basis_2.original_basis_array)

    # Test subtraction
    new = basis_tools.difference_basis_sets(basis_1, basis_2, add_or_subtract='sub')
    assert np.allclose(
        new.original_basis_array,
        basis_1.original_basis_array - basis_2.original_basis_array)

    # Test missmatched
    basis_1.remove_fid_from_basis('NAA')
    new = basis_tools.difference_basis_sets(basis_1, basis_2, missing_metabs='ignore')
    assert new.n_metabs == 2

    with pytest.raises(basis_tools.IncompatibleBasisError) as exc_info:
        new = basis_tools.difference_basis_sets(basis_1, basis_2, missing_metabs='raise')
        assert exc_info.type is basis_tools.IncompatibleBasisError
        assert exc_info.value.args[0] == "NAA does not occur in basis_1."


def test_conj():
    basis = mrs_io.read_basis(basis_off)
    basis_conj = basis_tools.conjugate_basis(mrs_io.read_basis(basis_off))
    assert np.allclose(basis_conj.original_basis_array, basis.original_basis_array.conj())

    basis_conj = basis_tools.conjugate_basis(mrs_io.read_basis(basis_off), name='NAA')
    index = basis_conj.names.index('NAA')
    assert np.allclose(basis_conj.original_basis_array[:, index], basis.original_basis_array[:, index].conj())
