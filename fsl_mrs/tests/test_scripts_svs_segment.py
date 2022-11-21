'''FSL-MRS test script

Test the svs segmentation script

Copyright Will Clarke, University of Oxford, 2021'''


# Imports
import subprocess
from pathlib import Path
import json
import numpy as np

# Files
testsPath = Path(__file__).parent
anat = testsPath / 'testdata/svs_segment/T1.anat'
svs = testsPath / 'testdata/fsl_mrs/metab.nii.gz'


def test_svs_segment(tmp_path):

    subprocess.check_call(['svs_segment',
                           '-a', anat,
                           '-o', tmp_path,
                           svs])
    assert (tmp_path / 'segmentation.json').exists()
    assert (tmp_path / 'mask.nii.gz').exists()

    with open(tmp_path / 'segmentation.json') as fp:
        vals = json.load(fp)
    sum_val = vals['CSF'] + vals['GM'] + vals['WM']

    assert np.isclose(sum_val, 1.0)
