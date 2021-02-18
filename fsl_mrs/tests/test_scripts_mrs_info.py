'''FSL-MRS test script

Test the mrs_info script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
from pathlib import Path

# Files
testsPath = Path(__file__).parent
processed = testsPath / 'testdata/fsl_mrs/metab.nii.gz'
unprocessed = testsPath / 'testdata/fsl_mrs_preproc/metab_raw.nii.gz'


def test_single_info(tmp_path):
    subprocess.check_call(['mrs_info', str(processed)])


def test_multi_info(tmp_path):
    subprocess.check_call(['mrs_info', str(processed), str(unprocessed)])
