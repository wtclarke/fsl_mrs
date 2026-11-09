'''FSL-MRS test script

Test io functions

Copyright Will Clarke, University of Oxford, 2021'''

import numpy as np
import os.path as op
import pytest
from pathlib import Path

from fsl_mrs import read_FID, read_basis
import fsl_mrs.utils.mrs_io.fsl_io as fslio
import fsl_mrs.utils.mrs_io.jmrui_io as jmruiio
from fsl_mrs.utils.mrs_io.main import _check_datatype, IncompatibleBasisFormat
from fsl_mrs.core.basis import Basis


testsPath = op.dirname(__file__)
SVSTestData = {'nifti': op.join(testsPath, 'testdata/mrs_io/metab.nii'),
               'raw': op.join(testsPath, 'testdata/mrs_io/metab.RAW'),
               'txt': op.join(testsPath, 'testdata/mrs_io/metab.txt')}

headerReqFields = ['centralFrequency', 'bandwidth', 'dwelltime']


def test_read_FID_SVS():
    # Test the loading of the three types of data we handle for SVS data
    # nifti + json
    # .raw
    # .txt

    data_nifti = read_FID(SVSTestData['nifti'])
    data_raw = read_FID(SVSTestData['raw'])
    data_txt = read_FID(SVSTestData['txt'])

    # Check that the data from each of these matches - it should they are all the same bit of data.
    datamean = np.mean([data_nifti[:],
                        data_raw[:],
                        data_txt[:]], axis=0)

    assert np.isclose(data_nifti[:], datamean).all()
    assert np.isclose(data_raw[:], datamean).all()
    assert np.isclose(data_txt[:], datamean).all()

    # # Check that the headers each contain the required fields
    # for r in headerReqFields:
    #     assert r in header_nifti
    #     assert r in header_raw
    #     assert r in header_txt

    #     headerMean = np.mean([header_nifti[r], header_raw[r], header_txt[r]])
    #     assert np.isclose(header_nifti[r], headerMean)
    #     assert np.isclose(header_raw[r], headerMean)
    #     assert np.isclose(header_txt[r], headerMean)

# TODO: Make MRSI test function (and find data)
# def test_read_FID_MRSI()


BasisTestData = {
    'fsl': op.join(testsPath, 'testdata/mrs_io/basisset_FSL'),
    'fsl_nuc': op.join(testsPath, 'testdata/mrs_io/basisset_FSL_nuc'),  # Includes a basis_nucleus field (set to 31P)
    'fsl_seq_nuc': op.join(testsPath, 'testdata/mrs_io/basisset_FSL_seq_nuc'),   # Includes a seq->nucleus field (31P)
    'raw': op.join(testsPath, 'testdata/mrs_io/basisset_LCModel_raw'),
    'txt': op.join(testsPath, 'testdata/mrs_io/basisset_JMRUI'),
    'mrui': op.join(testsPath, 'testdata/mrs_io/basisset_mrui'),
    'txt_single': op.join(testsPath, 'testdata/mrs_io/basis_set_jMRUI.txt'),
    'lcm': op.join(testsPath, 'testdata/mrs_io/basisset_LCModel.BASIS')}


def test_read_Basis() -> None:
    # Test the loading of the four types of data we handle for basis specta
    # fsl_mrs - folder of json
    # lcmodel - .basis file
    # lcmodel - folder of .raw
    # jmrui - folder of .txt

    with pytest.raises(IncompatibleBasisFormat) as exc_info:
        _ = read_basis(BasisTestData['raw'])

    assert exc_info.type is IncompatibleBasisFormat
    assert exc_info.value.args[0] == "LCModel raw files don't contain enough information"\
                                     " to generate a Basis object. Please use fsl_mrs.utils.mrs_io"\
                                     ".lcm_io.read_basis_files to load the partial information."

    basis_fsl = read_basis(BasisTestData['fsl'])
    basis_txt = read_basis(BasisTestData['txt'])
    basis_mrui = read_basis(BasisTestData['mrui'])
    basis_txt_single = read_basis(BasisTestData['txt_single'])
    basis_lcm = read_basis(BasisTestData['lcm'])

    # Check each returns a basis object
    assert isinstance(basis_fsl, Basis)
    assert isinstance(basis_txt, Basis)
    assert isinstance(basis_mrui, Basis)
    assert isinstance(basis_txt_single, Basis)
    assert isinstance(basis_lcm, Basis)

    # lcm basis file is zeropadded by a factor of 2
    # Test that all contain the same amount of data.
    assert basis_fsl.original_points == 2048
    assert basis_txt.original_points == 2048
    assert basis_mrui.original_points == 1024
    assert basis_txt_single.original_points == 2048
    assert basis_lcm.original_points == (2 * 2048)

    # Test that the number of names match the amount of data
    numNames = 21
    assert len(basis_fsl.names) == numNames
    assert len(basis_txt.names) == numNames
    assert len(basis_mrui.names) == 13
    assert len(basis_txt_single.names) == 17
    assert len(basis_lcm.names) == numNames


def test_read_mruiBasis_files() -> None:
    mruifiles = sorted(Path(BasisTestData['mrui']).glob('*.mrui'))
    basis = jmruiio.read_mruiBasis_files(mruifiles)

    assert isinstance(basis, Basis)
    assert basis.original_basis_array.shape == (1024, 13)
    assert basis.names == [file.stem for file in mruifiles]
    assert np.isclose(basis.cf, 127728513.0 / 1E6)
    assert np.isclose(basis.original_bw, 2000)
    assert np.isclose(basis.original_dwell, 0.0005)
    assert basis.basis_fwhm == [None, ] * 13
    assert basis.nucleus == '1H'


def test_fslBasisRegen():
    pointsToGen = 100
    basis_fsl = read_basis(BasisTestData['fsl'])
    basis_fsl2 = fslio.readFSLBasisFiles(BasisTestData['fsl'],
                                         readoutShift=4.65,
                                         bandwidth=4000,
                                         points=pointsToGen)

    assert basis_fsl2.names == basis_fsl.names
    assert basis_fsl2.original_bw == basis_fsl.original_bw
    assert np.allclose(basis_fsl2.original_basis_array, basis_fsl.original_basis_array[:pointsToGen, :], atol=1E-2)


def test_check_datatype():
    '''Check various paths through _check_datatype'''

    assert _check_datatype(Path('fake/path/test.RAW')) == ('RAW', '.RAW')
    assert _check_datatype(Path('fake/path/test.H2O')) == ('RAW', '.H2O')
    assert _check_datatype(Path('fake/path/test.raw')) == ('RAW', '.raw')
    assert _check_datatype(Path('fake/path/test.h2o')) == ('RAW', '.h2o')

    assert _check_datatype(Path('fake/path/test.txt')) == ('TXT', '.txt')

    assert _check_datatype(Path('fake/path/test.nii')) == ('NIFTI', '.nii')
    assert _check_datatype(Path('fake/path/test.nii.gz')) == ('NIFTI', '.nii.gz')
    assert _check_datatype(Path('fake/path/test.blah.nii.gz')) == ('NIFTI', '.blah.nii.gz')
    assert _check_datatype(Path('fake/path/test.blah.nii')) == ('NIFTI', '.blah.nii')

    assert _check_datatype(Path('fake/../../nasty/path/test.nii.gz')) == ('NIFTI', '.nii.gz')


def test_fsl_io_save_load_basis(tmp_path):
    """Test the read and write basis functions for the fsl io module."""

    basis = fslio.readFSLBasisFiles(BasisTestData['fsl'])
    assert basis.original_basis_array.shape == (2048, 21)
    assert np.iscomplexobj(basis.original_basis_array)
    assert len(basis.names) == basis.original_basis_array.shape[1]
    assert basis.cf == 123218995.6 / 1E6
    assert basis.original_bw == 4000
    assert basis.original_dwell == 0.00025
    assert basis.basis_fwhm[0] == 2
    assert basis.nucleus == "1H"
    assert basis.axes.ppmshift == 4.65

    basis.save(tmp_path)
    assert (tmp_path / (basis.names[0] + '.json')).exists()

    nbasis = fslio.readFSLBasisFiles(tmp_path)
    assert np.allclose(nbasis.original_basis_array[:, 0], basis.original_basis_array[:, 0])
    assert nbasis.names[0] == basis.names[0]
    assert nbasis.cf == basis.cf
    assert nbasis.original_bw == basis.original_bw
    assert nbasis.original_dwell == basis.original_dwell
    assert nbasis.basis_fwhm == basis.basis_fwhm
    assert nbasis.nucleus == basis.nucleus
    assert nbasis.axes.ppmshift == basis.axes.ppmshift


def test_fsl_io_save_load_basis_nucleus(tmp_path):

    # With nucleus information
    # Test that read directly ["basis"]["basis_nucleus"] works
    basis = fslio.readFSLBasisFiles(BasisTestData['fsl_nuc'])
    assert basis.nucleus == "31P"

    # Test that read from ["seq"]["Nucleus"] works
    basis = fslio.readFSLBasisFiles(BasisTestData['fsl_seq_nuc'])
    assert basis.nucleus == "31P"

    basis.save(tmp_path)
    nbasis = fslio.readFSLBasisFiles(tmp_path)
    assert nbasis.cf == basis.cf
    assert nbasis.original_bw == basis.original_bw
    assert nbasis.original_dwell == basis.original_dwell
    assert nbasis.basis_fwhm == basis.basis_fwhm
    assert nbasis.nucleus == basis.nucleus
    assert nbasis.nucleus == "31P"
    assert nbasis.axes.ppmshift == 4.65


def test_load_symlink(tmp_path):
    from fsl_mrs.utils.misc import create_rel_symlink
    create_rel_symlink(SVSTestData['nifti'], tmp_path, 'test1')

    import os.path as op
    assert op.islink(tmp_path / 'test1.nii')
    assert read_FID(tmp_path / 'test1.nii').shape == (1, 1, 1, 4096)
