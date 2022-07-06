from pathlib import Path
import subprocess
import pandas as pd

testsPath = Path(__file__).parent

sim_results = [
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/stim',
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub1/stim',
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/ctrl',
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub1/ctrl']

fl_contrasts = testsPath / 'testdata/fmrs_tools/fl_contrasts.json'
design_file = testsPath / 'testdata/fmrs_tools/design.mat'
con_file = testsPath / 'testdata/fmrs_tools/design.con'


def test_fmrs_stats(tmp_path):

    with open(tmp_path / 'results_list', 'w') as fp:
        fp.writelines([str(x) + '\n' for x in sim_results])

    subprocess.check_call([
        'fmrs_stats',
        '--data', str(tmp_path / 'results_list'),
        '--output', str(tmp_path / 'out'),
        '--fl-contrasts', str(fl_contrasts),
        '--combine', 'NAA', 'NAAG',
        '--combine', 'Cr', 'PCr',
        '--combine', 'PCh', 'GPC',
        '--hl-design', str(design_file),
        '--hl-contrasts', str(con_file),
        '--hl-contrast-names', "STIM>CTRL", "CTRL>STIM",
        '--overwrite'])

    assert (tmp_path / 'out' / 'group_stats.csv').is_file()
    assert (tmp_path / 'out' / '0_stim').is_dir()
    assert (tmp_path / 'out' / '1_stim').is_dir()
    assert (tmp_path / 'out' / '2_ctrl').is_dir()
    assert (tmp_path / 'out' / '3_ctrl').is_dir()
    assert (tmp_path / 'out' / '0_stim' / 'free_parameters.csv').is_file()

    df = pd.read_csv(tmp_path / 'out' / 'group_stats.csv', index_col=0, header=[0, 1])
    assert ('COPE', "STIM>CTRL") in df.columns
    assert 'conc_NAA+NAAG_mean_activation' in df.index
