'''FSL-MRS test script

Test the NIFTI-MRS class implementation

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
from pathlib import Path

import numpy as np

from fsl_mrs.core import NIFTI_MRS
from fsl_mrs.core.nifti_mrs import gen_new_nifti_mrs
from fsl_mrs.utils import mrs_io

# Files
testsPath = Path(__file__).parent
data = {'metab': testsPath / 'testdata/fsl_mrs/metab.nii.gz',
        'unprocessed': testsPath / 'testdata/fsl_mrs_preproc/metab_raw.nii.gz',
        'water': testsPath / 'testdata/fsl_mrs/wref.nii.gz',
        'basis': testsPath / 'testdata/fsl_mrs/steam_basis'}


def test_nifti_mrs():
    obj = NIFTI_MRS(data['unprocessed'])

    assert obj.mrs_nifti_version == '0.2'
    assert obj.shape == (1, 1, 1, 4096, 32, 64)
    assert obj.dwelltime == 8.33e-05
    assert obj.nucleus == ['1H']
    assert obj.spectrometer_frequency == [297.219948]
    assert obj.bandwidth == 1 / obj.dwelltime
    assert obj.dim_tags == ['DIM_COIL', 'DIM_DYN', None]
    assert isinstance(obj.hdr_ext, dict)

    hdr_ext = obj.hdr_ext
    hdr_ext['bogus'] = 'test'
    obj.hdr_ext = hdr_ext
    assert 'bogus' in obj.hdr_ext
    assert obj.dim_position('DIM_DYN') == 5

    assert obj.copy(remove_dim='DIM_DYN').shape == (1, 1, 1, 4096, 32)


def test_nifti_mrs_save(tmp_path):
    obj = NIFTI_MRS(data['metab'])
    obj.save(tmp_path / 'out')

    assert (tmp_path / 'out.nii.gz').exists()


def test_nifti_mrs_generator():
    obj = NIFTI_MRS(data['unprocessed'])

    for gen_data, slice_idx in obj.iterate_over_dims():
        assert gen_data.shape == (1, 1, 1, 4096)
        assert slice_idx == (slice(None, None, None),
                             slice(None, None, None),
                             slice(None, None, None),
                             slice(None, None, None),
                             0, 0)
        break

    for gen_data, slice_idx in obj.iterate_over_dims(dim='DIM_DYN'):
        assert gen_data.shape == (1, 1, 1, 4096, 64)
        assert slice_idx == (slice(None, None, None),
                             slice(None, None, None),
                             slice(None, None, None),
                             slice(None, None, None),
                             0, slice(None, None, None))
        break

    for gen_data, slice_idx in obj.iterate_over_dims(dim='DIM_DYN', iterate_over_space=True):
        assert gen_data.shape == (4096, 64)
        assert slice_idx == (0, 0, 0, slice(None, None, None), 0, slice(None, None, None))
        break

    for gen_data, slice_idx in obj.iterate_over_dims(dim='DIM_DYN', iterate_over_space=True,
                                                     reduce_dim_index=True):
        assert gen_data.shape == (4096, 64)
        assert slice_idx == (0, 0, 0, slice(None, None, None), 0)
        break


def test_nifti_mrs_gen_mrs():
    obj = NIFTI_MRS(data['unprocessed'])

    for mrs in obj.generate_mrs(dim='DIM_COIL',
                                basis_file=str(data['basis']),
                                ref_data=str(data['water'])):
        assert len(mrs) == 32
        assert mrs[0].FID.shape == (4096,)
        assert mrs[0].basis.shape == (4096, 20)
        break

    basis, names, basishdr = mrs_io.read_basis(str(data['basis']))
    for mrs in obj.generate_mrs(dim='DIM_DYN',
                                basis=basis, names=names, basis_hdr=basishdr[0],
                                ref_data=str(data['water'])):
        assert len(mrs) == 64
        assert mrs[0].FID.shape == (4096,)
        assert mrs[0].basis.shape == (4096, 20)
        break


def test_nifti_mrs_mrs():
    obj = NIFTI_MRS(data['metab'])

    assert obj.mrs().FID.shape == (4095, )


def test_gen_new_nifti_mrs(tmp_path):
    data = np.zeros((1, 1, 1, 1024, 4), dtype=np.complex64)
    affine = np.eye(4)
    nmrs = gen_new_nifti_mrs(data,
                             1 / 2000.0,
                             128.0,
                             nucleus='1H',
                             affine=affine,
                             dim_tags=['DIM_COIL', None, None])

    assert nmrs.shape == (1, 1, 1, 1024, 4)
    assert nmrs.dwelltime == 1 / 2000.0
    assert nmrs.nucleus == ['1H']
    assert nmrs.spectrometer_frequency == [128.0]
    assert nmrs.bandwidth == 2000.0
    assert nmrs.dim_tags == ['DIM_COIL', None, None]
    assert isinstance(nmrs.hdr_ext, dict)

    nmrs.save(tmp_path / 'out')

    assert (tmp_path / 'out.nii.gz').exists()
