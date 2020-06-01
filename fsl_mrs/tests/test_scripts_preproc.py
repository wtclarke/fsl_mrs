import subprocess
import pytest
import os.path as op
from glob import glob

testsPath = op.dirname(__file__)
data = {'metab':op.join(testsPath,'testdata/fsl_mrs_preproc/steam_metab_raw*.nii.gz'),
        'wrefc':op.join(testsPath,'testdata/fsl_mrs_preproc/steam_wref_comb_raw*.nii.gz'),
        'wrefq':op.join(testsPath,'testdata/fsl_mrs_preproc/steam_wref_quant_raw*.nii.gz'),
        'ecc':op.join(testsPath,'testdata/fsl_mrs_preproc/steam_ecc_raw*.nii.gz')}

def test_preproc(tmp_path):

    allfiles_metab = glob(data['metab'])
    allfiles_wrefc = glob(data['wrefc'])
    allfiles_wrefq = glob(data['wrefq'])
    allfiles_ecc = glob(data['ecc'])

    retcode = subprocess.check_call(['fsl_mrs_preproc',
                        '--output', str(tmp_path),
                        '--data']+ allfiles_metab+
                        ['--reference',]+allfiles_wrefc+
                        ['--quant',]+allfiles_wrefq+
                        ['--ecc',]+allfiles_ecc+
                        ['--hlsvd',
                        '--leftshift','1',
                         '--overwrite',
                        '--report' ])
    print(retcode)
    assert retcode==0
    assert op.isfile(op.join(tmp_path,'mergedReports.html'))