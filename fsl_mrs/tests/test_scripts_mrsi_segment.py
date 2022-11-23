'''FSL-MRS test script

Test the mrsi segmentation script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
from pathlib import Path
from fsl.data.image import Image
import numpy as np


# Files
testsPath = Path(__file__).parent
anat = testsPath / 'testdata/mrsi_segment/T1.anat'
mrsi = testsPath / 'testdata/fsl_mrsi/FID_Metab.nii.gz'


def test_mrsi_segment(tmp_path):

    subprocess.run(['mrsi_segment',
                    '-a', anat,
                    '-o', tmp_path,
                    mrsi])
    assert (tmp_path / 'mrsi_seg_wm.nii.gz').exists()
    assert (tmp_path / 'mrsi_seg_gm.nii.gz').exists()
    assert (tmp_path / 'mrsi_seg_csf.nii.gz').exists()

    assert not (tmp_path / 'tmp_sum.nii.gz').exists()

    wm = Image(tmp_path / 'mrsi_seg_wm.nii.gz')
    gm = Image(tmp_path / 'mrsi_seg_gm.nii.gz')
    csf = Image(tmp_path / 'mrsi_seg_csf.nii.gz')

    sum_img = wm[:] + gm[:] + csf[:]

    assert np.allclose(np.unique(sum_img.round(decimals=3)), [0.0, 1.0])


def test_mrsi_segment_no_norm(tmp_path):

    subprocess.run(['mrsi_segment',
                    '-a', anat,
                    '-o', tmp_path,
                    '--no_normalisation',
                    mrsi])
    assert (tmp_path / 'mrsi_seg_wm.nii.gz').exists()
    assert (tmp_path / 'mrsi_seg_gm.nii.gz').exists()
    assert (tmp_path / 'mrsi_seg_csf.nii.gz').exists()
