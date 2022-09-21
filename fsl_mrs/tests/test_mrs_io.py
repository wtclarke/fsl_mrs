'''FSL-MRS test script

Test io functions

Copyright Will Clarke, University of Oxford, 2021'''

import numpy as np
import os.path as op
import pytest
from pathlib import Path

import fsl_mrs.utils.mrs_io as mrsio
import fsl_mrs.utils.mrs_io.fsl_io as fslio
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

    data_nifti = mrsio.read_FID(SVSTestData['nifti'])
    data_raw = mrsio.read_FID(SVSTestData['raw'])
    data_txt = mrsio.read_FID(SVSTestData['txt'])

    # Check that the data from each of these matches - it should they are all the same bit of data.
    datamean = np.mean([data_nifti.data,
                        data_raw.data,
                        data_txt.data], axis=0)

    assert np.isclose(data_nifti.data, datamean).all()
    assert np.isclose(data_raw.data, datamean).all()
    assert np.isclose(data_txt.data, datamean).all()

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


BasisTestData = {'fsl': op.join(testsPath, 'testdata/mrs_io/basisset_FSL'),
                 'raw': op.join(testsPath, 'testdata/mrs_io/basisset_LCModel_raw'),
                 'txt': op.join(testsPath, 'testdata/mrs_io/basisset_JMRUI'),
                 'txt_single': op.join(testsPath, 'testdata/mrs_io/basis_set_jMRUI.txt'),
                 'lcm': op.join(testsPath, 'testdata/mrs_io/basisset_LCModel.BASIS')}


def test_read_Basis():
    # Test the loading of the four types of data we handle for basis specta
    # fsl_mrs - folder of json
    # lcmodel - .basis file
    # lcmodel - folder of .raw
    # jmrui - folder of .txt

    with pytest.raises(IncompatibleBasisFormat) as exc_info:
        _ = mrsio.read_basis(BasisTestData['raw'])

    assert exc_info.type is IncompatibleBasisFormat
    assert exc_info.value.args[0] == "LCModel raw files don't contain enough information"\
                                     " to generate a Basis object. Please use fsl_mrs.utils.mrs_io"\
                                     ".lcm_io.read_basis_files to load the partial information."

    basis_fsl = mrsio.read_basis(BasisTestData['fsl'])
    basis_txt = mrsio.read_basis(BasisTestData['txt'])
    basis_txt_single = mrsio.read_basis(BasisTestData['txt_single'])
    basis_lcm = mrsio.read_basis(BasisTestData['lcm'])

    # Check each returns a basis object
    assert isinstance(basis_fsl, Basis)
    assert isinstance(basis_txt, Basis)
    assert isinstance(basis_txt_single, Basis)
    assert isinstance(basis_lcm, Basis)

    # lcm basis file is zeropadded by a factor of 2
    # Test that all contain the same amount of data.
    assert basis_fsl.original_points == 2048
    assert basis_txt.original_points == 2048
    assert basis_txt_single.original_points == 2048
    assert basis_lcm.original_points == (2 * 2048)

    # Test that the number of names match the amount of data
    numNames = 21
    assert len(basis_fsl.names) == numNames
    assert len(basis_txt.names) == numNames
    assert len(basis_txt_single.names) == 17
    assert len(basis_lcm.names) == numNames


def test_fslBasisRegen():
    pointsToGen = 100
    basis_fsl = mrsio.read_basis(BasisTestData['fsl'])
    basis_fsl2, names_fsl2, headers_fsl2 = fslio.readFSLBasisFiles(BasisTestData['fsl'],
                                                                   readoutShift=4.65,
                                                                   bandwidth=4000,
                                                                   points=pointsToGen)
    basis_fsl2 = Basis(basis_fsl2, names_fsl2, headers_fsl2)

    assert basis_fsl2.names == basis_fsl.names
    assert basis_fsl2.original_bw == basis_fsl.original_bw
    assert np.allclose(basis_fsl2.original_basis_array, basis_fsl.original_basis_array[:pointsToGen, :])


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

    basis, names, hdrs = fslio.readFSLBasisFiles(BasisTestData['fsl'])
    assert basis.shape == (2048, 21)
    assert np.iscomplexobj(basis)
    assert len(names) == basis.shape[1]
    assert hdrs[0]['centralFrequency'] == 123218995.6
    assert hdrs[0]['bandwidth'] == 4000
    assert hdrs[0]['dwelltime'] == 0.00025
    assert hdrs[0]['fwhm'] == 2

    fslio.write_fsl_basis_file(basis[:, 0], names[0], hdrs[0], tmp_path)
    assert (tmp_path / (names[0] + '.json')).exists()

    nbasis, nnames, nhdr = fslio.readFSLBasisFiles(tmp_path)
    assert np.allclose(nbasis[:, 0], basis[:, 0])
    assert nnames[0] == names[0]
    assert nhdr[0] == hdrs[0]
