"""Test the contrast manipulation functions used for fMRS second level group analysis

Test functions that appear in utils.fmrs_tools.contrasts module

Copyright Will Clarke, University of Oxford, 2022"""

from pathlib import Path

import numpy as np
import pandas as pd

import fsl_mrs.utils.fmrs_tools.contrasts as con

testsPath = Path(__file__).parent
sim_results = testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/ctrl'


def test__comb_variance():
    """Test simple covariance combination case"""

    cov = np.eye(2)
    cov[0, 1] = -0.25
    cov[1, 0] = -0.25

    cov_df = pd.DataFrame(cov, columns=['a', 'b'], index=['a', 'b'])

    comb_var = con._comb_variance(cov, [1, 1])
    comb_var_df = con._comb_variance(cov_df, [1, 1])

    true_var = 1 + 1 + 2 * -0.25

    assert np.isclose(comb_var, true_var)
    assert np.isclose(comb_var_df, true_var)


def test__comb_value():

    values = np.ones(5)
    df = pd.DataFrame(values, index=['a', 'b', 'c', 'd', 'e']).T

    new_df = con._comb_value(df, ['a', 'b'], 'a+b')
    assert np.isclose(new_df['a+b'], 2)

    new_df = con._comb_value(df, ['a', 'b', 'c'], 'a+b-2c', scale=[0.5, 0.5, -2])
    assert np.isclose(new_df['a+b-2c'], -1)


def test__combine_params():

    metabolites = ['A', 'B', 'C', 'D']
    params = [
        'conc_A_beta0', 'conc_A_beta1', 'conc_A_beta2',
        'conc_B_beta0', 'conc_B_beta1', 'conc_B_beta2',
        'conc_C_beta0', 'conc_C_beta1', 'conc_C_beta2',
        'conc_D_betaA', 'conc_D_betaB',
        'gamma_0', 'gamma_1', 'gamma_2',
        'eps_0', 'eps_1', 'eps_2',
        'sigma_0_beta0', 'sigma_0_beta1', 'sigma_0_beta2',
        'sigma_1_beta0', 'sigma_1_beta1',
        'sigma_2_beta1', 'sigma_2_beta2',
        'phi_0',
        'phi_1_betaA', 'phi_1_betaB']

    val_df = pd.DataFrame(np.ones(len(params)), index=params).T
    cov = 0.9E-3 * np.eye(len(params)) + 1E-4 * np.ones((len(params), len(params)))
    cov_df = pd.DataFrame(cov, columns=params, index=params)

    new_vals, new_cov, new_params = \
        con._combine_params(val_df, cov_df, [['A', 'B']], [], metabolites)

    assert all([x in new_vals.columns for x in new_params])
    assert all([x in new_cov.columns for x in new_params])
    assert all([x in new_vals.columns for x in ['conc_A+B_beta0', 'conc_A+B_beta1', 'conc_A+B_beta2']])
    assert np.isclose(new_vals['conc_A+B_beta0'], 2.0)
    assert np.isclose(new_cov.loc['conc_A+B_beta0', 'conc_A+B_beta0'], 2.2E-3)

    # Test contrasts
    contrasts = [
        con.Contrast('mean', ['beta0', 'beta1', 'beta2'], [1 / 3, 1 / 3, 1 / 3]),
        con.Contrast('0-1', ['beta0', 'beta1'], [1, -1]),
        con.Contrast('sum', ['betaA', 'betaB'], [1, 1])]

    new_vals, new_cov, new_params = \
        con._combine_params(val_df, cov_df, [], contrasts, metabolites)

    assert all([x in new_vals.columns for x in new_params])
    assert all([x in new_cov.columns for x in new_params])
    assert all([x in new_vals.columns for x in
                ['conc_A_mean', 'conc_B_mean', 'conc_C_mean', 'conc_A_0-1', 'conc_B_0-1', 'conc_C_0-1']])
    assert np.isclose(new_vals['conc_A_mean'], 1.0)
    assert np.isclose(new_cov.loc['conc_A_mean', 'conc_A_mean'], (3 / 9 + 6 / 90) * 1E-3)

    assert all([x in new_vals.columns for x in
                ['sigma_0_mean', 'sigma_0_0-1', 'sigma_1_0-1']])
    assert np.isclose(new_vals['sigma_0_mean'], 1.0)
    assert np.isclose(new_cov.loc['sigma_0_mean', 'sigma_0_mean'], (3 / 9 + 6 / 90) * 1E-3)

    assert 'phi_1_sum' in new_vals.columns
    assert np.isclose(new_vals['phi_1_sum'], 2.0)

    # Test contrast + metabs
    contrasts = [
        con.Contrast('mean', ['beta0', 'beta1', 'beta2'], [1 / 3, 1 / 3, 1 / 3])]

    new_vals, new_cov, new_params = \
        con._combine_params(val_df, cov_df, [['A', 'B']], contrasts, metabolites)

    assert all([x in new_vals.columns for x in new_params])
    assert all([x in new_cov.columns for x in new_params])
    assert all([x in new_vals.columns for x in
                ['conc_A_mean', 'conc_B_mean', 'conc_C_mean',
                 'conc_A+B_beta0', 'conc_A+B_beta1', 'conc_A+B_beta2',
                 'conc_A+B_mean']])
    assert np.isclose(new_vals['conc_A+B_mean'], 2.0)
    assert np.isclose(new_cov.loc['conc_A+B_mean', 'conc_A+B_mean'], (6 / 9 + 30 / 90) * 1E-3)


def test_create_contrasts(tmp_path):
    """Test the primary function in module.

    See that it can load, manipulate, and save outputs
    """
    contrasts = [
        con.Contrast('mean', ['beta0', 'beta1'], [0.5, 0.5])]
    metabolites_to_combine = [
        ['PCh', 'GPC'],
        ['Cr', 'PCr'],
        ['NAA', 'NAAG'],
        ['Glu', 'Gln']]

    new_vals, new_cov, summary_out, new_params = con.create_contrasts(
        sim_results,
        contrasts=contrasts,
        metabolites_to_combine=metabolites_to_combine,
        output_dir=tmp_path,
        full_load=False)

    assert all([x in new_vals.columns for x in new_params])
    assert all([x in new_cov.columns for x in new_params])
    assert all([x in summary_out.index for x in new_params])
    assert (tmp_path / 'dyn_cov.csv').is_file()
    assert (tmp_path / 'dyn_results.csv').is_file()
    assert (tmp_path / 'free_parameters.csv').is_file()
