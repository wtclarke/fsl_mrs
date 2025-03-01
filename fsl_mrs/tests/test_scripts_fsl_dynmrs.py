'''FSL-MRS test script

Test the dynamic MRS fitting script fsl_dynmrs

Copyright Will Clarke, University of Oxford, 2022'''
from subprocess import run
from pathlib import Path

import pytest
import numpy as np

import fsl_mrs.utils.synthetic as syn
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl_mrs.core import basis

testsPath = Path(__file__).parent
model_path = testsPath / 'testdata/dynamic/simple_linear_model.py'


@pytest.fixture
def fixed_ratio_data(tmp_path):
    FID_basis1 = syn.syntheticFID(chemicalshift=[1, ], amplitude=[1], noisecovariance=[[0]], damping=[3])
    FID_basis2 = syn.syntheticFID(chemicalshift=[3, ], amplitude=[1], noisecovariance=[[0]], damping=[3])
    FID_basis1[1]['fwhm'] = 3 * np.pi
    FID_basis2[1]['fwhm'] = 3 * np.pi
    bset = basis.Basis(
        np.stack((FID_basis1[0][0], FID_basis2[0][0]), axis=1),
        ['Met1', 'Met2'],
        [FID_basis1[1], FID_basis2[1]])

    FID1 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[1, 1], noisecovariance=[[0.01]])
    FID2 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[2, 2], noisecovariance=[[0.01]])

    fid1 = FID1[0][0].reshape((1, 1, 1, 2048))
    fid2 = FID2[0][0].reshape((1, 1, 1, 2048))
    data = np.stack((fid1, fid2), axis=-1)

    data = np.conj(data)
    nmrs = gen_nifti_mrs(
        data,
        FID1[1]['dwelltime'],
        FID1[1]['centralFrequency'],
        dim_tags=['DIM_DYN', None, None])

    time_var = np.arange(2)

    # Save
    basis_path = tmp_path / 'basis'
    data_path = tmp_path / 'data.nii.gz'
    tv_path = tmp_path / 'time_var.csv'

    nmrs.save(data_path)
    bset.save(basis_path)
    np.savetxt(tv_path, time_var, delimiter=',')

    return data_path, basis_path, tv_path


def test_fixtures(fixed_ratio_data):
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
