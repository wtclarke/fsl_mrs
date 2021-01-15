import subprocess
from pathlib import Path

testsPath = Path(__file__).parent
data = testsPath / 'testdata/fsl_mrs_preproc'
t1 = str(testsPath / 'testdata/svs_segment/T1.anat/T1_biascorr.nii.gz')


def test_preproc(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_raw.nii.gz')
    wrefq = str(data / 'quant_raw.nii.gz')
    ecc = str(data / 'ecc.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--hlsvd',
         '--leftshift', '1',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()
