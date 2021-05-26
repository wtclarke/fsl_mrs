'''FSL-MRS test script

Test the tools script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
from pathlib import Path

# Files
testsPath = Path(__file__).parent

# Testing vis option
svs = testsPath / 'testdata/fsl_mrs/metab.nii.gz'
basis = testsPath / 'testdata/fsl_mrs/steam_basis'


def test_vis_svs(tmp_path):
    subprocess.check_call(['mrs_vis',
                           '--ppmlim', '0.2', '4.2',
                           '--save', str(tmp_path / 'svs.png'),
                           svs])

    assert (tmp_path / 'svs.png').exists()


def test_vis_basis(tmp_path):
    subprocess.check_call(['mrs_vis',
                           '--ppmlim', '0.2', '4.2',
                           '--save', str(tmp_path / 'basis.png'),
                           basis])

    assert (tmp_path / 'basis.png').exists()


# Testing info option
processed = testsPath / 'testdata/fsl_mrs/metab.nii.gz'
unprocessed = testsPath / 'testdata/fsl_mrs_preproc/metab_raw.nii.gz'


def test_single_info(tmp_path):
    subprocess.check_call(['mrs_info', str(processed)])


def test_multi_info(tmp_path):
    subprocess.check_call(['mrs_info', str(processed), str(unprocessed)])
