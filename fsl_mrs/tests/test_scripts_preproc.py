import subprocess
from pathlib import Path
from glob import glob

testsPath = Path(__file__).parent
data = testsPath / 'testdata/fsl_mrs_preproc'


def test_preproc(tmp_path):

    allfiles_metab = list(data.glob('steam_metab_raw*.nii.gz'))
    allfiles_wrefc = list(data.glob('steam_wref_comb_raw*.nii.gz'))
    allfiles_wrefq = list(data.glob('steam_wref_quant_raw*.nii.gz'))
    allfiles_ecc = list(data.glob('steam_ecc_raw*.nii.gz'))

    retcode = subprocess.check_call(
            ['fsl_mrs_preproc',
             '--output', str(tmp_path),
             '--data'] + allfiles_metab +
            ['--reference'] + allfiles_wrefc +
            ['--quant'] + allfiles_wrefq +
            ['--ecc'] + allfiles_ecc +
            ['--hlsvd',
             '--leftshift', '1',
             '--overwrite',
             '--report',
             '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'mergedReports.html').exists()
