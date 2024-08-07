from pathlib import Path
import subprocess
import pandas as pd
import numpy as np

testsPath = Path(__file__).parent

sim_results = [
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/stim',
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub1/stim',
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/ctrl',
    testsPath / 'testdata/fmrs_tools/sim_fmrs/sub1/ctrl']

fl_contrasts = testsPath / 'testdata/fmrs_tools/fl_contrasts.json'
design_file_gm = testsPath / 'testdata/fmrs_tools/design_groupmean.mat'
con_file_gm = testsPath / 'testdata/fmrs_tools/design_groupmean.con'
design_file_paired = testsPath / 'testdata/fmrs_tools/design.mat'
con_file_paired = testsPath / 'testdata/fmrs_tools/design.con'

ftest_file = testsPath / 'testdata/fmrs_tools/design.fts'
ftests_file = testsPath / 'testdata/fmrs_tools/design2.fts'


def test_fmrs_stats_default(tmp_path):

    with open(tmp_path / 'results_list', 'w') as fp:
        fp.writelines([str(x) + '\n' for x in sim_results])

    subprocess.check_call([
        'fmrs_stats',
        '--data', str(tmp_path / 'results_list'),
        '--output', str(tmp_path / 'out'),
        '--overwrite'])

    assert (tmp_path / 'out' / 'group_stats.csv').is_file()
    assert (tmp_path / 'out' / '0_stim').is_dir()
    assert (tmp_path / 'out' / '1_stim').is_dir()
    assert (tmp_path / 'out' / '2_ctrl').is_dir()
    assert (tmp_path / 'out' / '3_ctrl').is_dir()
    assert (tmp_path / 'out' / '0_stim' / 'free_parameters.csv').is_file()

    df = pd.read_csv(tmp_path / 'out' / 'group_stats.csv', index_col=0, header=[0])
    assert 'COPE' in df.columns
    assert 'conc_NAA_beta3' in df.index


def test_fmrs_stats_first_level(tmp_path):

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
        '--overwrite'])

    assert (tmp_path / 'out' / 'group_stats.csv').is_file()
    assert (tmp_path / 'out' / '0_stim').is_dir()
    assert (tmp_path / 'out' / '1_stim').is_dir()
    assert (tmp_path / 'out' / '2_ctrl').is_dir()
    assert (tmp_path / 'out' / '3_ctrl').is_dir()
    assert (tmp_path / 'out' / '0_stim' / 'free_parameters.csv').is_file()

    df = pd.read_csv(tmp_path / 'out' / 'group_stats.csv', index_col=0, header=[0])
    assert 'COPE' in df.columns
    assert 'conc_NAA+NAAG_beta0' in df.index
    assert 'conc_NAA+NAAG_mean_activation' in df.index
    assert 'conc_Cr+PCr_mean_activation' in df.index
    assert 'conc_PCh+GPC_mean_activation' in df.index

    assert 'sigma_0_mean_activation' in df.index
    assert 'sigma_1_mean_activation' in df.index
    assert 'gamma_1_mean_activation' not in df.index
    assert 'gamma_1' in df.index


def test_fmrs_stats_first_level_shortcut(tmp_path):

    with open(tmp_path / 'results_list', 'w') as fp:
        fp.writelines([str(x) + '\n' for x in sim_results])

    subprocess.check_call([
        'fmrs_stats',
        '--data', str(tmp_path / 'results_list'),
        '--output', str(tmp_path / 'out'),
        '--mean-contrasts', 'beta0', 'beta1',
        '--mean-contrasts', 'beta2', 'beta3',
        '--combine', 'NAA', 'NAAG',
        '--combine', 'Cr', 'PCr',
        '--combine', 'PCh', 'GPC',
        '--overwrite'])

    assert (tmp_path / 'out' / 'group_stats.csv').is_file()
    assert (tmp_path / 'out' / '0_stim').is_dir()
    assert (tmp_path / 'out' / '1_stim').is_dir()
    assert (tmp_path / 'out' / '2_ctrl').is_dir()
    assert (tmp_path / 'out' / '3_ctrl').is_dir()
    assert (tmp_path / 'out' / '0_stim' / 'free_parameters.csv').is_file()

    df = pd.read_csv(tmp_path / 'out' / 'group_stats.csv', index_col=0, header=[0])
    assert 'COPE' in df.columns
    assert 'conc_NAA+NAAG_beta0' in df.index
    assert 'conc_NAA+NAAG_mean_0_beta0_beta1' in df.index
    assert 'sigma_1_mean_1_beta2_beta3' in df.index


def test_fmrs_stats_scaling(tmp_path):
    with open(tmp_path / 'results_list', 'w') as fp:
        fp.writelines([str(x) + '\n' for x in sim_results])

    subprocess.check_call([
        'fmrs_stats',
        '--data', str(tmp_path / 'results_list'),
        '--output', str(tmp_path / 'out_noscale'),
        '--fl-contrasts', str(fl_contrasts),
        '--combine', 'NAA', 'NAAG',
        '--combine', 'Cr', 'PCr',
        '--combine', 'PCh', 'GPC',
        '--overwrite'])

    subprocess.check_call([
        'fmrs_stats',
        '--data', str(tmp_path / 'results_list'),
        '--output', str(tmp_path / 'out_scaled'),
        '--fl-contrasts', str(fl_contrasts),
        '--reference-contrast', 'conc_Cr+PCr_beta3',
        '--combine', 'NAA', 'NAAG',
        '--combine', 'Cr', 'PCr',
        '--combine', 'PCh', 'GPC',
        '--overwrite'])

    df_no_scale = pd.read_csv(tmp_path / 'out_noscale' / 'group_stats.csv', index_col=0, header=[0])
    df_scaled = pd.read_csv(tmp_path / 'out_scaled' / 'group_stats.csv', index_col=0, header=[0])

    assert np.isclose(
        df_scaled.loc['conc_NAA_beta3', 'COPE'],
        df_no_scale.loc['conc_NAA_beta3', 'COPE'] / df_no_scale.loc['conc_Cr+PCr_beta3', 'COPE'],
        atol=1E-1)

    assert np.isclose(
        df_scaled.loc['sigma_0_beta3', 'COPE'],
        df_no_scale.loc['sigma_0_beta3', 'COPE'],
        atol=1E-2)


def test_fmrs_stats_group_mean(tmp_path):

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
        '--hl-design', str(design_file_gm),
        '--hl-contrasts', str(con_file_gm),
        '--hl-contrast-names', "positive", "negative",
        '--overwrite'])

    assert (tmp_path / 'out' / 'group_stats.csv').is_file()
    assert (tmp_path / 'out' / '0_stim').is_dir()
    assert (tmp_path / 'out' / '1_stim').is_dir()
    assert (tmp_path / 'out' / '2_ctrl').is_dir()
    assert (tmp_path / 'out' / '3_ctrl').is_dir()
    assert (tmp_path / 'out' / '0_stim' / 'free_parameters.csv').is_file()

    df = pd.read_csv(tmp_path / 'out' / 'group_stats.csv', index_col=0, header=[0, 1])
    assert ('COPE', "positive") in df.columns
    assert 'conc_NAA+NAAG_mean_activation' in df.index
    assert np.isclose(
        df.loc['conc_NAA+NAAG_mean_activation', ('z', 'positive')],
        -df.loc['conc_NAA+NAAG_mean_activation', ('z', 'negative')])


def test_fmrs_stats_pairedttest(tmp_path):

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
        '--hl-design', str(design_file_paired),
        '--hl-contrasts', str(con_file_paired),
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


def test_fmrs_stats_pairedttest_ftest(tmp_path):
    # Single f contrast
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
        '--hl-design', str(design_file_paired),
        '--hl-contrasts', str(con_file_paired),
        '--hl-contrast-names', "STIM>CTRL", "CTRL>STIM",
        '--hl-ftests', str(ftest_file),
        '--overwrite'])

    assert (tmp_path / 'out' / 'group_fstats.csv').is_file()


def test_fmrs_stats_pairedttest_ftests(tmp_path):
    # Two f contrasts
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
        '--hl-design', str(design_file_paired),
        '--hl-contrasts', str(con_file_paired),
        '--hl-contrast-names', "STIM>CTRL", "CTRL>STIM",
        '--hl-ftests', str(ftests_file),
        '--overwrite'])

    assert (tmp_path / 'out' / 'group_fstats.csv').is_file()
