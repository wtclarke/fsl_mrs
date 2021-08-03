'''FSL-MRS test script

Test the main mrsi fitting script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
from pathlib import Path

# Files
testsPath = Path(__file__).parent
data = {'metab': testsPath / 'testdata/fsl_mrsi/FID_Metab.nii.gz',
        'water': testsPath / 'testdata/fsl_mrsi/FID_ref.nii.gz',
        'basis': testsPath / 'testdata/fsl_mrsi/3T_slaser_32vespa_1250.BASIS',
        'mask': testsPath / 'testdata/fsl_mrsi/small_mask.nii.gz',
        'seg_wm': testsPath / 'testdata/fsl_mrsi/mrsi_seg_wm.nii.gz',
        'seg_gm': testsPath / 'testdata/fsl_mrsi/mrsi_seg_gm.nii.gz',
        'seg_csf': testsPath / 'testdata/fsl_mrsi/mrsi_seg_csf.nii.gz'}


def test_fsl_mrsi(tmp_path):

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out'),
                           '--h2o', data['water'],
                           '--TE', '30',
                           '--TR', '2.0',
                           '--add_MM',
                           '--mask', data['mask'],
                           '--tissue_frac',
                           data['seg_wm'],
                           data['seg_gm'],
                           data['seg_csf'],
                           '--overwrite',
                           '--combine', 'Cr', 'PCr'])

    assert (tmp_path / 'fit_out/fit').exists()
    assert (tmp_path / 'fit_out/qc').exists()
    assert (tmp_path / 'fit_out/uncertainties').exists()
    assert (tmp_path / 'fit_out/concs').exists()
    assert (tmp_path / 'fit_out/nuisance').exists()

    assert (tmp_path / 'fit_out/concs/raw/NAA.nii.gz').exists()
    assert (tmp_path / 'fit_out/concs/molality/NAA.nii.gz').exists()
    assert (tmp_path / 'fit_out/uncertainties/NAA_sd.nii.gz').exists()
    assert (tmp_path / 'fit_out/qc/NAA_snr.nii.gz').exists()
    assert (tmp_path / 'fit_out/fit/fit.nii.gz').exists()

    assert (tmp_path / 'fit_out/nuisance/p0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/p1.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/shift_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/combined_lw_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/gamma_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/sigma_group0.nii.gz').exists()


def test_fsl_mrsi_noh2o(tmp_path):

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out'),
                           '--add_MM',
                           '--mask', data['mask'],
                           '--overwrite',
                           '--combine', 'Cr', 'PCr'])

    assert (tmp_path / 'fit_out/fit').exists()
    assert (tmp_path / 'fit_out/qc').exists()
    assert (tmp_path / 'fit_out/uncertainties').exists()
    assert (tmp_path / 'fit_out/concs').exists()
    assert (tmp_path / 'fit_out/nuisance').exists()

    assert (tmp_path / 'fit_out/concs/raw/NAA.nii.gz').exists()
    assert (tmp_path / 'fit_out/concs/internal/NAA.nii.gz').exists()
    assert (tmp_path / 'fit_out/uncertainties/NAA_sd.nii.gz').exists()
    assert (tmp_path / 'fit_out/qc/NAA_snr.nii.gz').exists()
    assert (tmp_path / 'fit_out/fit/fit.nii.gz').exists()

    assert (tmp_path / 'fit_out/nuisance/p0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/p1.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/shift_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/combined_lw_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/gamma_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/sigma_group0.nii.gz').exists()