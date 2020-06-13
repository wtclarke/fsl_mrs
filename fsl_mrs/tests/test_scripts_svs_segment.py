# Test the svs segmentation script

# Imports
import subprocess
from pathlib import Path

# Files
testsPath = Path(__file__).parent
anat = testsPath / 'testdata/svs_segment/T1.anat'
svs = testsPath / 'testdata/fsl_mrs/metab.nii'


def test_svs_segment(tmp_path):

    subprocess.check_call(['svs_segment',
                           '-a', anat,
                           '-o', tmp_path,
                           svs])
    assert (tmp_path / 'segmentation.json').exists()
    assert (tmp_path / 'mask.nii.gz').exists()
