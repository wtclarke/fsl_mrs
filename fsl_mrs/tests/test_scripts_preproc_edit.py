'''FSL-MRS test script

Test the edited svs preprocessing script

Copyright Will Clarke, University of Oxford, 2021'''

import subprocess
from pathlib import Path

testsPath = Path(__file__).parent
data = testsPath / 'testdata/fsl_mrs_preproc_edit'
t1 = str(testsPath / 'testdata/svs_segment/T1.anat/T1_biascorr.nii.gz')


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
         '--dynamic_align_ppm', '1.8', '4.2',
         '--edit_align_ppm', '2.5', '4.2',
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
