# Test the mrsi segmentation script

# Imports
import subprocess
from pathlib import Path

# Files
testsPath = Path(__file__).parent
anat = testsPath / 'testdata/mrsi_segment/T1.anat'
mrsi = testsPath / 'testdata/fsl_mrsi/FID_metab.nii.gz'


def test_mrsi_segment(tmp_path):

    subprocess.check_call(['mrsi_segment',
                           '-a', anat,
                           '-o', tmp_path,
                           mrsi])
    assert (tmp_path / 'mrsi_seg_wm.nii.gz').exists()
    assert (tmp_path / 'mrsi_seg_gm.nii.gz').exists()
    assert (tmp_path / 'mrsi_seg_csf.nii.gz').exists()
