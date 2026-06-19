'''FSL-MRS test script

Test the main mrsi fitting script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
import json
from pathlib import Path
import re
import nibabel as nib
import numpy as np
from fsl_mrs.utils.validate_results import compare_folders

# Files
testsPath = Path(__file__).parent
data = {'metab': testsPath / 'testdata/fsl_mrsi/FID_Metab.nii.gz',
        'water': testsPath / 'testdata/fsl_mrsi/FID_ref.nii.gz',
        'basis': testsPath / 'testdata/fsl_mrsi/3T_slaser_32vespa_1250_wmm',
        'mask': testsPath / 'testdata/fsl_mrsi/small_mask.nii.gz',
        'seg_wm': testsPath / 'testdata/fsl_mrsi/mrsi_seg_wm.nii.gz',
        'seg_gm': testsPath / 'testdata/fsl_mrsi/mrsi_seg_gm.nii.gz',
        'seg_csf': testsPath / 'testdata/fsl_mrsi/mrsi_seg_csf.nii.gz'}


def test_fsl_mrsi(tmp_path):

    print(' '.join(['fsl_mrsi',
                    '--data', str(data['metab']),
                    '--basis', str(data['basis']),
                    '--output', str(tmp_path / 'fit_out'),
                    '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
                    '--h2o', str(data['water']),
                    '--TE', '30',
                    '--TR', '2.0',
                    '--mask', str(data['mask']),
                    '--parallel-workers', '1',
                    '--parallel-batch-size-multiple', '2',
                    '--slow-fit-log-threshold', '0.000001',
                    '--tissue_frac',
                    str(data['seg_wm']),
                    str(data['seg_gm']),
                    str(data['seg_csf']),
                    '--output_correlations',
                    '--overwrite',
                    '--combine', 'Cr', 'PCr']))

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out'),
                           '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
                           '--h2o', data['water'],
                           '--TE', '30',
                           '--TR', '2.0',
                           '--mask', data['mask'],
                           '--parallel-workers', '1',
                           '--parallel-batch-size-multiple', '2',
                           '--slow-fit-log-threshold', '0.000001',
                           '--tissue_frac',
                           data['seg_wm'],
                           data['seg_gm'],
                           data['seg_csf'],
                           '--output_correlations',
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
    assert (tmp_path / 'fit_out/qc/fit_failed.nii.gz').exists()
    assert (tmp_path / 'fit_out/fit/fit.nii.gz').exists()
    assert (tmp_path / 'fit_out/mrsi.tree').exists()

    assert (tmp_path / 'fit_out/nuisance/p0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/p1.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/shift_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/combined_lw_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/gamma_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/sigma_group0.nii.gz').exists()

    assert (tmp_path / 'fit_out/misc/metabolite_groups.json').exists()
    assert (tmp_path / 'fit_out/misc/mrs_fit_parameters.json').exists()
    assert (tmp_path / 'fit_out/misc/fit_correlations.nii.gz').exists()
    assert (tmp_path / 'fit_out/misc/fit_times.nii.gz').exists()
    assert (tmp_path / 'fit_out/misc/slow_fits.jsonl').exists()
    with open(tmp_path / 'fit_out/misc/slow_fits.jsonl') as log_file:
        slow_fit_logs = [json.loads(line) for line in log_file]
    assert len(slow_fit_logs) == 6
    assert {'voxel_index', 'elapsed_seconds', 'scipy_minimize_output'} <= set(slow_fit_logs[0])

    mask = np.asanyarray(nib.load(data['mask']).dataobj)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, 2)
    mask = mask != 0
    fit_failed_img = nib.load(tmp_path / 'fit_out/qc/fit_failed.nii.gz')
    fit_failed = np.asanyarray(fit_failed_img.dataobj)
    assert fit_failed_img.get_data_dtype() == np.uint8
    assert set(np.unique(fit_failed)).issubset({0, 1})

    expected_failed = np.zeros(mask.shape, dtype=bool)
    for slow_fit_log in slow_fit_logs:
        index = tuple(int(ind) for ind in slow_fit_log['voxel_index'])
        success = slow_fit_log['scipy_minimize_result']['success']
        expected_failed[index] = success in (False, 'False')
    assert np.array_equal(fit_failed.astype(bool), expected_failed)

    fit_times = np.asanyarray(nib.load(tmp_path / 'fit_out/misc/fit_times.nii.gz').dataobj)
    assert np.all(fit_times[mask] > 0)
    assert np.all(fit_times[~mask] == 0)


def test_fsl_mrsi_partial_internal_reference_failure(tmp_path):
    """Test that even if you have a partially / not-fit internal reference
    the command runs and returns output (all or partially zeros)."""

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_ref_out'),
                           '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
                           '--mask', data['mask'],
                           '--parallel-workers', '1',
                           '--parallel-batch-size-multiple', '2',
                           '--internal_ref', 'Tau',
                           '--overwrite',
                           '--combine', 'Cr', 'PCr'])

    assert (tmp_path / 'fit_ref_out/concs/raw/Tau.nii.gz').exists()
    assert (tmp_path / 'fit_ref_out/concs/internal/Tau.nii.gz').exists()
    assert (tmp_path / 'fit_ref_out/qc/fit_failed.nii.gz').exists()
    assert (tmp_path / 'fit_ref_out/misc/fit_times.nii.gz').exists()

    mask = np.asanyarray(nib.load(data['mask']).dataobj)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, 2)
    mask = mask != 0

    tau_internal = np.asanyarray(
        nib.load(tmp_path / 'fit_ref_out/concs/internal/Tau.nii.gz').dataobj)
    assert np.all(np.isfinite(tau_internal[mask]))


def test_fsl_mrsi_models(tmp_path):

    def gen_cmd(out_path):
        return ['fsl_mrsi',
                '--data', data['metab'],
                '--basis', data['basis'],
                '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
                '--mask', data['mask'],
                '--overwrite',
                '--combine', 'Cr', 'PCr',
                '--output', str(tmp_path / out_path)]

    subprocess.run(gen_cmd('voigt'))
    assert (tmp_path / 'voigt/concs/raw/NAA.nii.gz').exists()
    subprocess.run(gen_cmd('lorentzian') + ['--lorentzian',])
    assert (tmp_path / 'lorentzian/concs/raw/NAA.nii.gz').exists()
    subprocess.run(gen_cmd('lorentzianfs') + ['--lorentzian', '--free_shift'])
    assert (tmp_path / 'lorentzianfs/concs/raw/NAA.nii.gz').exists()
    subprocess.run(gen_cmd('fs') + ['--free_shift',])
    assert (tmp_path / 'fs/concs/raw/NAA.nii.gz').exists()


def test_default_mm_warning(tmp_path, capfd):
    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out'),
                           '--h2o', data['water'],
                           '--TE', '30',
                           '--TR', '2.0',
                           '--mask', data['mask'],
                           '--tissue_frac',
                           data['seg_wm'],
                           data['seg_gm'],
                           data['seg_csf'],
                           '--output_correlations',
                           '--overwrite',
                           '--combine', 'Cr', 'PCr'])
    out, _ = capfd.readouterr()
    pattern = re.compile(
        re.escape(
            'Default macromolecules (MM09, MM12, MM14, MM17, MM21) are present in the '
            'basis set.\n'
            'However they are not all listed in the --metab_groups.\n'
            'It is recommended that all default MM are assigned their own group.\n'
            'E.g. Use --metab_groups MM09 MM12 MM14 MM17 MM21\n'))
    assert pattern.match(out) is not None


def test_fsl_mrsi_noh2o(tmp_path):

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out'),
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
    assert (tmp_path / 'fit_out/mrsi.tree').exists()

    assert (tmp_path / 'fit_out/nuisance/p0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/p1.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/shift_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/combined_lw_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/gamma_group0.nii.gz').exists()
    assert (tmp_path / 'fit_out/nuisance/sigma_group0.nii.gz').exists()

    assert (tmp_path / 'fit_out/misc/metabolite_groups.json').exists()
    assert (tmp_path / 'fit_out/misc/mrs_fit_parameters.json').exists()


def test_alt_ref(tmp_path):

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out'),
                           '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
                           '--h2o', data['water'],
                           '--TE', '30',
                           '--TR', '2.0',
                           '--mask', data['mask'],
                           '--tissue_frac',
                           data['seg_wm'],
                           data['seg_gm'],
                           data['seg_csf'],
                           '--output_correlations',
                           '--overwrite',
                           '--combine', 'Cr', 'PCr',
                           '--wref_metabolite', 'PCho', 'GPC',
                           '--ref_protons', '3',
                           '--ref_int_limits', '3.0', '3.4'])

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


# TODO create a separate test for possible baseline arguments
def test_baseline_options(tmp_path):

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out1'),
                           '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
                           '--h2o', data['water'],
                           '--TE', '30',
                           '--TR', '2.0',
                           '--mask', data['mask'],
                           '--tissue_frac',
                           data['seg_wm'],
                           data['seg_gm'],
                           data['seg_csf'],
                           '--overwrite',
                           '--combine', 'Cr', 'PCr',
                           '--baseline', 'polynomial, 4'])

    assert (tmp_path / 'fit_out1/concs/raw/NAA.nii.gz').exists()

    subprocess.check_call(['fsl_mrsi',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', str(tmp_path / 'fit_out2'),
                           '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
                           '--h2o', data['water'],
                           '--TE', '30',
                           '--TR', '2.0',
                           '--mask', data['mask'],
                           '--tissue_frac',
                           data['seg_wm'],
                           data['seg_gm'],
                           data['seg_csf'],
                           '--overwrite',
                           '--combine', 'Cr', 'PCr',
                           '--baseline_order', '4'])

    assert (tmp_path / 'fit_out2/concs/raw/NAA.nii.gz').exists()

    # Fit time maps are wall-clock diagnostics and are expected to differ
    # between otherwise equivalent runs.
    (tmp_path / 'fit_out1/misc/fit_times.nii.gz').unlink()
    (tmp_path / 'fit_out2/misc/fit_times.nii.gz').unlink()

    assert compare_folders((tmp_path / 'fit_out2'), (tmp_path / 'fit_out1'), subdir=True)
