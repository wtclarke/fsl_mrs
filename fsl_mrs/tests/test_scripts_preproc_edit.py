'''FSL-MRS test script

Test the edited svs preprocessing script

Copyright Will Clarke, University of Oxford, 2021'''

import subprocess
from pathlib import Path

from fsl_mrs.utils import mrs_io

testsPath = Path(__file__).parent
data = testsPath / 'testdata/fsl_mrs_preproc_edit'
t1 = str(testsPath / 'testdata/svs_segment/T1.anat/T1_biascorr.nii.gz')


def test_preproc_defaults(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_internal.nii.gz')
    wrefq = str(data / 'wref_quant.nii.gz')
    ecc = str(data / 'wref_internal.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc_edit',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--hlsvd',
         '--leftshift', '2',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'diff.nii.gz').exists()
    assert (tmp_path / 'edit_0.nii.gz').exists()
    assert (tmp_path / 'edit_1.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()
    assert (tmp_path / 'options.txt').exists()
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()


def test_preproc(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_internal.nii.gz')
    wrefq = str(data / 'wref_quant.nii.gz')
    ecc = str(data / 'wref_internal.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc_edit',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--align_ppm_dynamic', '1.8', '4.2',
         '--align_ppm_edit', '2.5', '4.2',
         '--hlsvd',
         '--leftshift', '2',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'diff.nii.gz').exists()
    assert (tmp_path / 'edit_0.nii.gz').exists()
    assert (tmp_path / 'edit_1.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()
    assert (tmp_path / 'options.txt').exists()
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()


def test_preproc_noavg(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_internal.nii.gz')
    wrefq = str(data / 'wref_quant.nii.gz')
    ecc = str(data / 'wref_internal.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc_edit',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--align_ppm_dynamic', '1.8', '4.2',
         '--align_ppm_edit', '2.5', '4.2',
         '--leftshift', '2',
         '--overwrite',
         '--report',
         '--verbose',
         '--noaverage'])

    assert retcode == 0
    assert (tmp_path / 'diff.nii.gz').exists()
    assert (tmp_path / 'edit_0.nii.gz').exists()
    assert (tmp_path / 'edit_1.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()
    assert (tmp_path / 'options.txt').exists()
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()

    diff = mrs_io.read_FID(tmp_path / 'diff.nii.gz')
    on = mrs_io.read_FID(tmp_path / 'edit_0.nii.gz')
    off = mrs_io.read_FID(tmp_path / 'edit_1.nii.gz')

    assert diff.shape[-1] == 16
    assert on.shape[-1] == 16
    assert off.shape[-1] == 16
