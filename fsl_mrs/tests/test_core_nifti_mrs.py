'''FSL-MRS test script

Test the NIFTI-MRS class implementation

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
from pathlib import Path

import pytest
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


def test_add_remove_field():

    nmrs = NIFTI_MRS(data['unprocessed'])

    with pytest.raises(ValueError) as exc_info:
        nmrs.remove_hdr_field('SpectrometerFrequency')

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == 'You cannot remove the required metadata.'

    with pytest.raises(ValueError) as exc_info:
        nmrs.remove_hdr_field('ResonantNucleus')

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == 'You cannot remove the required metadata.'

    with pytest.raises(ValueError) as exc_info:
        nmrs.remove_hdr_field('dim_5')

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == 'Modify dimension headers through dedicated methods.'

    with pytest.raises(ValueError) as exc_info:
        nmrs.add_hdr_field('dim_5_header', {'p1': [1, 2, 3]})

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == 'Modify dimension headers through dedicated methods.'

    nmrs.add_hdr_field('RepetitionTime', 5.0)
    assert 'RepetitionTime' in nmrs.hdr_ext
    assert nmrs.hdr_ext['RepetitionTime'] == 5.0

    nmrs.remove_hdr_field('RepetitionTime')
    assert 'RepetitionTime' not in nmrs.hdr_ext


def test_set_dim_info():
    nmrs = NIFTI_MRS(data['unprocessed'])
    nmrs.set_dim_info('DIM_DYN', 'my info')
    assert nmrs.hdr_ext['dim_6_info'] == 'my info'


def test_set_dim_header():
    nmrs = NIFTI_MRS(data['unprocessed'])
    with pytest.raises(ValueError) as exc_info:
        nmrs.set_dim_header('DIM_DYN', {'my_hdr': np.arange(10).tolist()})
    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == 'New dim header length must be 64'

    nmrs.set_dim_header('DIM_DYN', {'my_hdr': np.arange(64).tolist()})
    assert nmrs.hdr_ext['dim_6_header'] == {'my_hdr': np.arange(64).tolist()}
