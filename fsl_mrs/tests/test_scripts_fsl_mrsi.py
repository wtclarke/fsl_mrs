'''FSL-MRS test script

Test the main mrsi fitting script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
from pathlib import Path
import re

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

    assert (tmp_path / 'fit_out/misc/metabolite_groups.json').exists()
    assert (tmp_path / 'fit_out/misc/mrs_fit_parameters.json').exists()
    assert (tmp_path / 'fit_out/misc/fit_correlations.nii.gz').exists()


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


def test_baseline_options(tmp_path):

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
                           '--overwrite',
                           '--combine', 'Cr', 'PCr',
                           '--baseline', 'polynomial, 3'])

    assert (tmp_path / 'fit_out/concs/raw/NAA.nii.gz').exists()

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
                           '--overwrite',
                           '--combine', 'Cr', 'PCr',
                           '--baseline', 'spline, flexible'])

    assert (tmp_path / 'fit_out/concs/raw/NAA.nii.gz').exists()

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
                           '--overwrite',
                           '--combine', 'Cr', 'PCr',
                           '--baseline_order', '4'])

    assert (tmp_path / 'fit_out/concs/raw/NAA.nii.gz').exists()
