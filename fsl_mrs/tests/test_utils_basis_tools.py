'''
FSL-MRS test script

Test basis tools.

Copyright Will Clarke, University of Oxford, 2021
'''
from pathlib import Path
from shutil import copytree

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


def test_add_basis(tmp_path):
    out_loc = tmp_path / 'test_basis'
    copytree(fsl_basis_path, out_loc)

    mac_in = fsl_io.readJSON(extra_basis)
    fid = np.asarray(mac_in['basis_re']) + 1j * np.asarray(mac_in['basis_im'])
    cf = mac_in['basis_centre']
    bw = 1 / mac_in['basis_dwell']

    with pytest.raises(basis_tools.IncompatibleBasisError) as exc_info:
        basis_tools.add_basis(fid, 'mac2', cf, bw, out_loc)

    assert exc_info.type is basis_tools.IncompatibleBasisError
    assert exc_info.value.args[0] == "The new basis FID covers too little time, try padding."
    fid_pad = np.pad(fid, (0, fid.size))
    basis_tools.add_basis(fid_pad, 'mac2', cf, bw, out_loc)

    new_basis = mrs_io.read_basis(out_loc)
    index = new_basis.names.index('mac2')
    assert 'mac2' in new_basis.names
    assert np.allclose(new_basis._raw_fids[:, index], fid_pad[0::2])

    basis_tools.add_basis(fid_pad, 'mac3', cf, bw, out_loc, scale=True, width=10)
    new_basis = mrs_io.read_basis(out_loc)
    index = new_basis.names.index('mac3')
    assert 'mac3' in new_basis.names
    assert new_basis.basis_fwhm[index] == 10


def test_shift():
    basis = mrs_io.read_basis(fsl_basis_path)

    shifted = basis_tools.shift_basis(mrs_io.read_basis(fsl_basis_path), 'NAA', 1.0)

    index = basis.names.index('NAA')
    amount_in_hz = 1.0 * basis.cf
    t = basis.original_time_axis
    shifted_fid = basis.original_basis_array[:, index] * np.exp(-1j * 2 * np.pi * t * amount_in_hz)
    assert np.allclose(shifted.original_basis_array[:, index], shifted_fid)
