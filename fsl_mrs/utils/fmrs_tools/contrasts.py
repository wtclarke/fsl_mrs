""" Module for combination of peaks and GLM betas

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
        Saad Jbabdi <saad@fmrib.ox.ac.uk>

Copyright (C) 2022 University of Oxford
# SHBASECOPYRIGHT
"""

from dataclasses import dataclass
import re

import pandas as pd
import numpy as np

from fsl_mrs.utils.fmrs_tools import utils


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
    2. Linear combination of betas - apply to all metabolites and other parameter groups (sigma, gamma, etc) in turn
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

    # First step: Identify all possible betas for the parameter combinations
    all_params = values.columns.to_list()

    # What betas exist for each metabolite?
    metab_beta_dict = {}
    for metab in metabolites:
        beta_re = re.compile(rf'^conc_{metab}_(.*)$',)
        betas = [beta_re.match(param)[1] for param in all_params if beta_re.match(param)]
        metab_beta_dict[metab] = betas

    # What betas exist for grouped parameter sets (sigma, gamma, eps)
    # a. unambiguously identify number of metabolite groups and if sigma exists
    any_sigma = [x.startswith('sigma') for x in all_params]
    contains_sigma = True if any(any_sigma) else False

    group_nums = []
    for param in all_params:
        matchobj = re.match(r'gamma_(\d+).*', param)
        if matchobj is not None:
            group_nums.append(int(matchobj[1]))
    group_nums = np.unique(group_nums)

    # b. Find betas
    group_param_used = ['gamma', 'eps']
    if contains_sigma:
        group_param_used += ['sigma']
    grouped_beta_dict = {}
    for num in group_nums:
        for gparam in group_param_used:
            beta_re = re.compile(rf'^{gparam}_{num}_(.*)$',)
            betas = [beta_re.match(param)[1] for param in all_params if beta_re.match(param)]
            grouped_beta_dict[f'{gparam}_{num}'] = betas

    # What betas exist for the other parameters?
    other_param_used = ['phi_0', 'phi_1', 'baseline']
    other_beta_dict = {}
    for param in other_param_used:
        if param == 'baseline':
            beta_re = re.compile(rf'^{param}_\d+_(.*)$',)
        else:
            beta_re = re.compile(rf'^{param}_(.*)$',)
        betas = [beta_re.match(param)[1] for param in all_params if beta_re.match(param)]
        other_beta_dict[param] = betas

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
        # A. Contrasts applied to metabolites: Loop over metabolites
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

        # B. Contrasts applied to grouped parameters: Loop over group numbers
        for num in group_nums:
            for gparam in group_param_used:
                curr_name = f'{gparam}_{num}'
                if not all([x in grouped_beta_dict[curr_name] for x in con.betas]):
                    if len(grouped_beta_dict[curr_name]) > 0:
                        print(
                            f'Contrast {con.name} requires betas {con.betas}, '
                            f'but {curr_name} has different betas: {grouped_beta_dict[curr_name]}. '
                            f'Skipping {curr_name}.'
                        )
                    continue
                parameters = [f'{curr_name}_{beta}' for beta in con.betas]

                new_param_name = f'{curr_name}_{con.name}'
                # Parameter value (sum)
                new_params.append(_comb_value(values, parameters, new_param_name, scale=con.scale))

                # Parameter variance
                new_var[new_param_name] = _comb_variance(covariance.loc[parameters, parameters], con.scale)

        # C. Contrasts applied to groupe parameters: Loop over group numbers
        for param in other_param_used:
            if not all([x in other_beta_dict[param] for x in con.betas]):
                if len(other_beta_dict[param]) > 0:
                    print(
                        f'Contrast {con.name} requires betas {con.betas}, '
                        f'but {param} has different betas: {other_beta_dict[param]}. '
                        f'Skipping {param}.'
                    )
                continue
            parameters = [f'{param}_{beta}' for beta in con.betas]

            new_param_name = f'{param}_{con.name}'
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

    Contrasts are (scaled) linear combinations of GLM betas. Applied at the first level.

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
    value_df, cov_df, metabolites = utils.load_dyn_res(results)

    # Run the combination
    values_out, covariance_out, new_params = _combine_params(
        value_df,
        cov_df,
        metabolites_to_combine,
        contrasts,
        metabolites)

    return utils.save_and_return_new_res(
        values_out,
        covariance_out,
        new_params,
        output_dir=output_dir)
