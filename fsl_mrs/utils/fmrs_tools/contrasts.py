""" Module for combination of peaks and GLM betas

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
        Saad Jbabdi <saad@fmrib.ox.ac.uk>

Copyright (C) 2022 University of Oxford
# SHBASECOPYRIGHT
"""

from pathlib import Path
from dataclasses import dataclass
import re

import pandas as pd
import numpy as np

import fsl_mrs.dynamic.dyn_results as dres


class MismatchedBetasError():
    pass


@dataclass
class Contrast:
    name: str
    betas: list
    scale: list


def _comb_variance(cov, scale):
    """Calculate variance of linear combinations of scalar values

    :param cov: parameter covariance matrix
    :type cov: numpy.array or pandas dataframe
    :param scale: scale vector (n x 1) vector of parameter scalings
    :type scale: numpy.array
    :return: combined variance
    :rtype: float
    """

    if isinstance(cov, pd.DataFrame):
        return np.asarray(scale) @ cov.to_numpy() @ np.asarray(scale)
    else:
        return np.asarray(scale) @ cov @ np.asarray(scale)


def _comb_value(df, parameters, name, scale=None):
    """Linear combination of parameter values

    :param df: Results dataframe
    :type df: pandas.DataFrame
    :param parameters: List of parameter names to linearly combine
    :type parameters: List
    :param name: New parameter name
    :type name: Str
    :param scale: Optional scaling values for each beta, defaults to a summation of values
    :type scale: List of floats, optional
    :return: A dataframe with the combined value and name
    :rtype: pandas.DataFrame
    """
    if scale is None:
        scale = np.ones(len(parameters))

    return pd.DataFrame(
        df[parameters].multiply(scale).sum(axis=1),
        columns=[name])


def _combine_params(values, covariance, metabolite_comb, contrasts, metabolites):
    """Perform the parameter and variance combinations. Combine GLM betas (contrasts) and/or
    metabolite peaks.

    Three cases to identify groupings:
    1. metabolites to combine - this is the sum of two or more metabolites. Applied to all betas in turn
    2. Linear combination of betas - apply to all metabolites in turn
    3. The combination of 1 & 2 which should be performed in a single step to capture all covariances

    :param values: Main results dataframe with columns for each parameter
    :type values: pandas.DataFrame
    :param covariance: N_params x N_params covariance matrix
    :type covariance: pandas.DataFrame
    :param metabolite_comb: Metabolites to combine, as list of list. E.g. [['NAA', 'NAAG], ['PCr', 'Cr']]
    :type metabolite_comb: List
    :param contrasts: List of contrast objects defining which GLM betas to combine linearly
    :type contrasts: List
    :param metabolites: List of original metabolite names
    :type metabolites: List
    :return: An expanded dataframe of values, including the combined values
    :rtype: pandas.DataFrame
    :return: Expanded covariance matrix with only the on-diagonals for the combined values populated
    :rtype: pandas.DataFrame
    :return: List of new combined parameter names
    :rtype: List
    """

    all_params = values.columns.to_list()

    metab_beta_dict = {}
    for metab in metabolites:
        beta_re = re.compile(rf'^conc_{metab}_(.*)$',)
        betas = [beta_re.match(param)[1] for param in all_params if beta_re.match(param)]
        metab_beta_dict[metab] = betas
    metab_beta_dict

    new_params = []
    new_var = {}

    # 1. metabolites to combine - this is the sum of two or more metabolites. Applied to all betas in turn
    for metabs in metabolite_comb:
        for met in metabs:
            if not metab_beta_dict[met] == metab_beta_dict[metabs[0]]:
                raise MismatchedBetasError(
                    f'{met} has betas {metab_beta_dict[met]}, '
                    f'but {metabs[0]} has different betas: {metab_beta_dict[metabs[0]]}.')

        new_met_name = '+'.join(metabs)
        # print(new_met_name)
        # loop over betas
        for beta in metab_beta_dict[metabs[0]]:
            parameters = [f'conc_{met}_{beta}' for met in metabs]

            new_param_name = f'conc_{new_met_name}_{beta}'
            # Parameter value (sum)
            new_params.append(_comb_value(values, parameters, new_param_name))

            # Parameter variance
            n_met = len(parameters)
            new_var[new_param_name] = _comb_variance(covariance.loc[parameters, parameters], np.ones(n_met))

    # 2. Linear combination of betas - apply to all metabolites in turn
    for con in contrasts:
        # Loop over metabolites
        for met in metabolites:
            if not all([x in metab_beta_dict[met] for x in con.betas]):
                print(
                    f'Contrast {con.name} requires betas {con.betas}, '
                    f'but {met} has different betas: {metab_beta_dict[met]}. '
                    f'Skipping {met}.'
                )
                continue
            parameters = [f'conc_{met}_{beta}' for beta in con.betas]

            new_param_name = f'conc_{met}_{con.name}'
            # Parameter value (sum)
            new_params.append(_comb_value(values, parameters, new_param_name, scale=con.scale))

            # Parameter variance
            new_var[new_param_name] = _comb_variance(covariance.loc[parameters, parameters], con.scale)

    # 3. The combination of 1 & 2 which should be performed in a single step to capture all covariances
    for con in contrasts:
        for metabs in metabolite_comb:
            parameters = [f'conc_{met}_{beta}' for beta in con.betas for met in metabs]
            new_met_name = '+'.join(metabs)
            new_param_name = f'conc_{new_met_name}_{con.name}'
            # print(new_param_name)
            # print(parameters)
            scale = np.stack([np.array(con.scale) for _ in range(len(metabs))]).T.ravel()
            # print(scale)
            # Parameter value (sum)
            new_params.append(_comb_value(values, parameters, new_param_name, scale=scale))

            # print(value_df[parameters])
            # print(new_params[-1])
            # Parameter variance
            new_var[new_param_name] = _comb_variance(covariance.loc[parameters, parameters], scale)

    # Form outputs
    new_value_df = pd.concat([values, ] + new_params, axis=1)
    new_cov_df = covariance.copy()
    for var in new_var:
        new_cov_df.loc[var, var] = new_var[var]

    return new_value_df, new_cov_df, list(new_var.keys())


def create_contrasts(results, contrasts=[], metabolites_to_combine=[], output_dir=None, full_load=False):
    """Generate contrasts from dynamic fMRS fit.

    _extended_summary_

    :param results: FSL-MRS dynamic results object, or path to saved results
    :type results: fsl_mrs.dynamic.dyn_results.dynRes or str or pathlib.Path
    :param contrasts: Contrast objects which encode a contrast name, and linear combination of betas, defaults to []
    :type contrasts: list of .Contrast objects, optional
    :param metabolites_to_combine: Nested list of metabolite names, defaults to []
    :type metabolites_to_combine: list, optional
    :param output_dir: Output directory, defaults to None
    :type output_dir: str or pathlib.Path, optional
    :param full_load: Load the full results object from file, defaults to False
    :type full_load: Bool, optional
    :return values_out: Expanded free parameter results dataframe
    :rtype: pandas.Dataframe
    :return covariance_out: Expanded covariance matrix (off-diagonals of new parameters are Nan)
    :rtype: pandas.Dataframe
    :return summary_df: New free_parameters.csv with parameter means and std.
    :rtype: pandas.Dataframe
    :return new_params: List of parameter names added by function
    :rtype: list
    """
    # Load data
    if isinstance(results, (dres.dynRes_mcmc, dres.dynRes_newton)):
        value_df = results.dataframe_free
        cov_df = results.cov_free
        mapped_params = results.mapped_names
    elif isinstance(results, (str, Path)):
        if isinstance(results, str):
            results = Path(results)

        # Either load full results object (if data availible) or just the key dataframes
        if full_load:
            try:
                obj = dres.load_dyn_result(results)
                value_df = obj.dataframe_free
                cov_df = obj.cov_free
            except dres.ResultLoadError:
                value_df = pd.read_csv(results / 'dyn_results.csv', index_col=0)
                cov_df = pd.read_csv(results / 'dyn_cov.csv', index_col=0)
        else:
            value_df = pd.read_csv(results / 'dyn_results.csv', index_col=0)
            cov_df = pd.read_csv(results / 'dyn_cov.csv', index_col=0)

        mapped_params = pd.read_csv(results / 'mapped_parameters.csv', index_col=0, header=[0, 1]).index

    # Unambiously identify metabolites
    metab_re = re.compile(r'^conc_(.*)',)
    metabolites = np.array([metab_re.match(param)[1] for param in mapped_params if metab_re.match(param)])

    # Run the combination
    values_out, covariance_out, new_params = _combine_params(
        value_df,
        cov_df,
        metabolites_to_combine,
        contrasts,
        metabolites)

    # Form the summary df as well
    mean_free = values_out.mean()
    std_free = pd.Series(np.sqrt(np.diag(covariance_out)), index=covariance_out.index)
    summary_df = pd.concat((mean_free, std_free), axis=1, keys=['mean', 'sd'])

    # Optionally output to file and return key dataframes
    if output_dir:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        values_out.to_csv(output_dir / 'dyn_results.csv')
        covariance_out.to_csv(output_dir / 'dyn_cov.csv')
        summary_df.to_csv(output_dir / 'free_parameters.csv')

        return values_out, covariance_out, summary_df, new_params
    else:
        return values_out, covariance_out, summary_df, new_params
