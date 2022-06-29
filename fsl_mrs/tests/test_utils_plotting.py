'''Test the plotting utilities in FSL-MRS

Copyright William Clarke, University of Oxford, 2022'''

from pathlib import Path
import filecmp

from matplotlib.figure import Figure

from fsl_mrs.utils import plotting
# Files
testsPath = Path(__file__).parent
t1_data = testsPath / 'testdata/svs_segment/T1.anat/T1_biascorr.nii.gz'
svs_data = testsPath / 'testdata/fsl_mrs/metab.nii.gz'

fig1 = testsPath / 'testdata/plotting/plot_world_orient.png'


def test_world_orientation_plot(tmp_path):
    fig = plotting.plot_world_orient(t1_data, svs_data)
    fig.savefig(tmp_path / 'plot_world_orient.png', bbox_inches='tight', facecolor='k')
    assert isinstance(fig, Figure)
    assert filecmp.cmp(fig1, str(tmp_path / 'plot_world_orient.png'))
