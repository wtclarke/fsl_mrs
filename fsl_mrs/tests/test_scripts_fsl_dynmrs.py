'''FSL-MRS test script

Test the dynamic MRS fitting script fsl_dynmrs

Author: William Clarke      <william.clarke@ndcn.ox.ac.uk>
        Vasilis Karlaftis   <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2026 University of Oxford
'''
from subprocess import run, CalledProcessError
from pathlib import Path
import re

import pytest
import numpy as np

import fsl_mrs.utils.synthetic as syn
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl_mrs.core import basis
from fsl.data.image import Image
from fsl_mrs.utils.validate_results import compare_folders


testsPath = Path(__file__).parent
model_path = testsPath / 'testdata/dynamic/simple_linear_model.py'


@pytest.fixture
def fixed_ratio_data(tmp_path, request):
    is_mrsi = getattr(request, "param", False)

    FID_basis1 = syn.syntheticFID(chemicalshift=[1, ], amplitude=[1], noisecovariance=[[0]], damping=[3])
    FID_basis2 = syn.syntheticFID(chemicalshift=[3, ], amplitude=[1], noisecovariance=[[0]], damping=[3])
    bset = basis.Basis(
        np.stack((FID_basis1[0][0], FID_basis2[0][0]), axis=1),
        ['Met1', 'Met2'],
        axes=FID_basis1[2])
    bset.basis_fwhm = [3 * np.pi, 3 * np.pi]

    if is_mrsi:
        spatial_dims = (2, 2, 2)
        nvoxels = np.prod(spatial_dims)
        FID1 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[1, 1], noisecovariance=0.01*np.eye(nvoxels),
                                coilamps=np.ones(nvoxels).tolist(),
                                coilphase=np.random.random(nvoxels) * 2 * np.pi)
        FID2 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[2, 2], noisecovariance=0.01*np.eye(nvoxels),
                                coilamps=np.ones(nvoxels).tolist(),
                                coilphase=np.random.random(nvoxels) * 2 * np.pi)
        fid1 = np.stack(FID1[0]).reshape(spatial_dims + (2048,))
        fid2 = np.stack(FID2[0]).reshape(spatial_dims + (2048,))
    else:
        FID1 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[1, 1], noisecovariance=[[0.01]])
        FID2 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[2, 2], noisecovariance=[[0.01]])
        fid1 = FID1[0][0].reshape((1, 1, 1, 2048))
        fid2 = FID2[0][0].reshape((1, 1, 1, 2048))

    data = np.stack((fid1, fid2), axis=-1)
    data = np.conj(data)
    nmrs = gen_nifti_mrs(
        data,
        FID1[2].dwelltime,
        FID1[2].SpectrometerFrequency,
        dim_tags=['DIM_DYN', None, None])

    time_var = np.arange(2)

    # Save
    data_path = tmp_path / 'data.nii.gz'
    basis_path = tmp_path / 'basis'
    tv_path = tmp_path / 'time_var.csv'

    nmrs.save(data_path)
    bset.save(basis_path)
    np.savetxt(tv_path, time_var, delimiter=',')

    return data_path, basis_path, tv_path


def test_fixtures_svs(fixed_ratio_data):
    assert fixed_ratio_data[0].exists()
    assert fixed_ratio_data[1].exists()
    assert fixed_ratio_data[2].exists()


@pytest.mark.parametrize("fixed_ratio_data", [True], indirect=True)
def test_fixtures_mrsi(fixed_ratio_data):
    assert fixed_ratio_data[0].exists()
    assert fixed_ratio_data[1].exists()
    assert fixed_ratio_data[2].exists()


def test_fsl_dynmrs(fixed_ratio_data, tmp_path):
    data_str = str(fixed_ratio_data[0])
    basis_str = str(fixed_ratio_data[1])
    tv_str = str(fixed_ratio_data[2])
    model_str = str(model_path)

    run(['fsl_dynmrs',
         '--data', data_str,
         '--basis', basis_str,
         '--dyn_config', model_str,
         '--time_variables', tv_str,
         '--baseline_order', '0',
         '--output', str(tmp_path / 'dyn_res'),
         '--report'])

    assert (tmp_path / 'dyn_res').exists()
    assert (tmp_path / 'dyn_res' / 'dyn_cov.csv').exists()
    assert (tmp_path / 'dyn_res' / 'init_results.csv').exists()
    assert (tmp_path / 'dyn_res' / 'dyn_results.csv').exists()
    assert (tmp_path / 'dyn_res' / 'mapped_parameters.csv').exists()
    assert (tmp_path / 'dyn_res' / 'free_parameters.csv').exists()
    assert (tmp_path / 'dyn_res' / 'options.txt').exists()
    assert (tmp_path / 'dyn_res' / 'report.html').exists()


def test_dynmrs_spline(fixed_ratio_data, tmp_path):
    data_str = str(fixed_ratio_data[0])
    basis_str = str(fixed_ratio_data[1])
    tv_str = str(fixed_ratio_data[2])
    model_str = str(model_path)
    run(['fsl_dynmrs',
         '--data', data_str,
         '--basis', basis_str,
         '--dyn_config', model_str,
         '--time_variables', tv_str,
         '--baseline', 'spline, flexible',
         '--output', str(tmp_path / 'dyn_res_spline'),
         '--report'])

    assert (tmp_path / 'dyn_res_spline').exists()
    assert (tmp_path / 'dyn_res_spline' / 'dyn_cov.csv').exists()
    assert (tmp_path / 'dyn_res_spline' / 'init_results.csv').exists()
    assert (tmp_path / 'dyn_res_spline' / 'dyn_results.csv').exists()
    assert (tmp_path / 'dyn_res_spline' / 'mapped_parameters.csv').exists()
    assert (tmp_path / 'dyn_res_spline' / 'free_parameters.csv').exists()
    assert (tmp_path / 'dyn_res_spline' / 'options.txt').exists()
    assert (tmp_path / 'dyn_res_spline' / 'report.html').exists()


def test_dynmrs_models(fixed_ratio_data, tmp_path):
    data_str = str(fixed_ratio_data[0])
    basis_str = str(fixed_ratio_data[1])
    tv_str = str(fixed_ratio_data[2])
    model_str = str(model_path)

    def gen_cmd(out_path):
        return ['fsl_dynmrs',
                '--data', data_str,
                '--basis', basis_str,
                '--dyn_config', model_str,
                '--time_variables', tv_str,
                '--baseline', 'spline, flexible',
                '--output', str(tmp_path / out_path),
                '--report']

    run(gen_cmd('voigt'))
    assert (tmp_path / 'voigt' / 'dyn_results.csv').exists()
    run(gen_cmd('lorentzian') + ['--lorentzian',])
    assert (tmp_path / 'lorentzian' / 'dyn_results.csv').exists()
    run(gen_cmd('lorentzianfs') + ['--lorentzian', '--free_shift'])
    assert (tmp_path / 'lorentzianfs' / 'dyn_results.csv').exists()
    run(gen_cmd('fs') + ['--free_shift',])
    assert (tmp_path / 'fs' / 'dyn_results.csv').exists()
    run(gen_cmd('inversion') + ['--inversion_model',])
    assert (tmp_path / 'inversion' / 'dyn_results.csv').exists()


@pytest.mark.parametrize("fixed_ratio_data", [True], indirect=True)
def test_fsl_dynmrs_spatial_mask(fixed_ratio_data, tmp_path):
    data_str = str(fixed_ratio_data[0])
    basis_str = str(fixed_ratio_data[1])
    tv_str = str(fixed_ratio_data[2])
    model_str = str(model_path)

    data = Image(data_str)

    # Test wrong shape but nonzero mask
    wrong_shape = np.array(data.shape[:3]) + 1
    wrong_shape_mask = np.ones(tuple(wrong_shape), dtype=np.uint8)
    wrong_shape_mask[0, 0, :] = 0
    wrong_shape_path = tmp_path / 'wrong_shape_mask.nii.gz'
    Image(wrong_shape_mask, xform=data.header.get_sform(), header=data.header).save(wrong_shape_path)
    with pytest.raises(CalledProcessError) as excinfo:
        run(['fsl_dynmrs',
             '--data', data_str,
             '--basis', basis_str,
             '--dyn_config', model_str,
             '--time_variables', tv_str,
             '--baseline_order', '0',
             '--spatial-mask', str(wrong_shape_path),
             '--output', str(tmp_path / 'dyn_res_wrong_shape'),
             '--report'],
            check=True,
            capture_output=True,
            text=True)
    assert 'Spatial mask shape' in excinfo.value.stderr

    # Test correct shape but zero mask
    empty_mask = np.zeros(data.shape[:3], dtype=np.uint8)
    empty_path = tmp_path / 'empty_mask.nii.gz'
    Image(empty_mask, xform=data.header.get_sform(), header=data.header).save(empty_path)
    with pytest.raises(CalledProcessError) as excinfo:
        run(['fsl_dynmrs',
             '--data', data_str,
             '--basis', basis_str,
             '--dyn_config', model_str,
             '--time_variables', tv_str,
             '--baseline_order', '0',
             '--spatial-mask', str(empty_path),
             '--output', str(tmp_path / 'dyn_res_empty'),
             '--report'],
            check=True,
            capture_output=True,
            text=True)
    assert 'Spatial mask is empty' in excinfo.value.stderr

    # Test correct shape and nonzero mask
    valid_mask = np.ones(data.shape[:3], dtype=np.uint8)
    valid_mask[0, 0, :] = 0
    valid_path = tmp_path / 'valid_mask.nii.gz'
    Image(valid_mask, xform=data.header.get_sform(), header=data.header).save(valid_path)
    run(['fsl_dynmrs',
         '--data', data_str,
         '--basis', basis_str,
         '--dyn_config', model_str,
         '--time_variables', tv_str,
         '--baseline_order', '0',
         '--spatial-mask', str(valid_path),
         '--output', str(tmp_path / 'dyn_res_valid'),
         '--parallel', 'off',
         '--report'])

    assert (tmp_path / 'dyn_res_valid').exists()
    assert (tmp_path / 'dyn_res_valid' / 'mean').exists()
    assert (tmp_path / 'dyn_res_valid' / 'var').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels').exists()
    assert not (tmp_path / 'dyn_res_valid' / 'voxels' / '0_0_0').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1' / 'dyn_cov.csv').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1' / 'init_results.csv').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1' / 'dyn_results.csv').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1' / 'mapped_parameters.csv').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1' / 'free_parameters.csv').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1' / 'options.txt').exists()
    assert (tmp_path / 'dyn_res_valid' / 'voxels' / '1_1_1' / 'report.html').exists()


@pytest.mark.parametrize("fixed_ratio_data", [True], indirect=True)
def test_fsl_dynmrs_spatial_index(fixed_ratio_data, tmp_path):
    data_str = str(fixed_ratio_data[0])
    basis_str = str(fixed_ratio_data[1])
    tv_str = str(fixed_ratio_data[2])
    model_str = str(model_path)

    # Test call without spatial_index
    run(['fsl_dynmrs',
         '--data', data_str,
         '--basis', basis_str,
         '--dyn_config', model_str,
         '--time_variables', tv_str,
         '--baseline_order', '0',
         '--output', str(tmp_path / 'dyn_res_all'),
         '--mean_mrsi',
         '--parallel', 'local',
         '--report'])

    assert (tmp_path / 'dyn_res_all').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean').exists()
    assert (tmp_path / 'dyn_res_all' / 'var').exists()
    assert (tmp_path / 'dyn_res_all' / 'voxels').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel' / 'dyn_cov.csv').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel' / 'init_results.csv').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel' / 'dyn_results.csv').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel' / 'mapped_parameters.csv').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel' / 'free_parameters.csv').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel' / 'options.txt').exists()
    assert (tmp_path / 'dyn_res_all' / 'mean_voxel' / 'report.html').exists()

    # Test multiple calls with spatial_index
    pattern = re.compile(r'^\d+_\d+_\d+$')
    voxels = [p for p in (tmp_path / 'dyn_res_all' / 'voxels').iterdir()
              if p.is_dir() and pattern.match(p.name)]
    for i in voxels:
        idx = i.name.split('_')
        run(['fsl_dynmrs',
             '--data', data_str,
             '--basis', basis_str,
             '--dyn_config', model_str,
             '--time_variables', tv_str,
             '--baseline_order', '0',
             '--output', str(tmp_path / 'dyn_res_split'),
             '--spatial-index', idx[0], idx[1], idx[2],
             '--report'])

    assert compare_folders(tmp_path / 'dyn_res_all' / 'voxels', tmp_path / 'dyn_res_split' / 'voxels', subdir=True)
