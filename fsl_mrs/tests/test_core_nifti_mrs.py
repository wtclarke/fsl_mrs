'''FSL-MRS test script

Test the NIFTI-MRS class implementation

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
from pathlib import Path

from fsl_mrs.core import NIFTI_MRS
from fsl_mrs.utils import mrs_io

# Files
testsPath = Path(__file__).parent
data = {'metab': testsPath / 'testdata/fsl_mrs/metab.nii.gz',
        'unprocessed': testsPath / 'testdata/fsl_mrs_preproc/metab_raw.nii.gz',
        'water': testsPath / 'testdata/fsl_mrs/wref.nii.gz',
        'basis': testsPath / 'testdata/fsl_mrs/steam_basis'}


def test_nifti_mrs():
    obj = NIFTI_MRS(data['unprocessed'])

    assert obj.nifti_mrs_version == '0.2'
    assert obj.shape == (1, 1, 1, 4096, 32, 64)
    assert obj.dwelltime == 8.33e-05
    assert obj.nucleus == ['1H']
    assert obj.spectrometer_frequency == [297.219948]
    assert obj.bandwidth == 1 / obj.dwelltime
    assert obj.dim_tags == ['DIM_COIL', 'DIM_DYN', None]

    copy_obj = obj.copy(remove_dim='DIM_DYN')
    assert copy_obj.shape == (1, 1, 1, 4096, 32)
    assert copy_obj.dim_tags == ['DIM_COIL', None, None]
    assert copy_obj.hdr_ext.dimensions == 5
    assert isinstance(copy_obj, NIFTI_MRS)


def test_nifti_mrs_gen_mrs():
    obj = NIFTI_MRS(data['unprocessed'])

    for mrs in obj.generate_mrs(dim='DIM_COIL',
                                basis_file=str(data['basis']),
                                ref_data=str(data['water'])):
        assert len(mrs) == 32
        assert mrs[0].FID.shape == (4096,)
        assert mrs[0].basis.shape == (4096, 20)
        break

    basis = mrs_io.read_basis(str(data['basis']))
    for mrs in obj.generate_mrs(dim='DIM_DYN',
                                basis=basis,
                                ref_data=str(data['water'])):
        assert len(mrs) == 64
        assert mrs[0].FID.shape == (4096,)
        assert mrs[0].basis.shape == (4096, 20)
        break


def test_nifti_mrs_mrs():
    obj = NIFTI_MRS(data['metab'])

    assert obj.mrs().FID.shape == (4095, )
