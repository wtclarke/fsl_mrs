"""Module for utility functions calculating fMRS stats.

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
        Saad Jbabdi <saad@fmrib.ox.ac.uk>

Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT
"""

from pathlib import Path
import re

import pandas as pd
import numpy as np

import fsl_mrs.dynamic.dyn_results as dres


def load_dyn_res(results, full_load=False, mapped_p=True):
    # Load data
    if isinstance(results, (dres.dynRes_mcmc, dres.dynRes_newton)):
        value_df = results.dataframe_free
        cov_df = results.cov_free
        mapped_params = results.mapped_names
    elif isinstance(results, (str, Path)):
        if isinstance(results, str):
            results = Path(results)

        # Either load full results object (if data available) or just the key dataframes
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

        if mapped_p:
            mapped_params = pd.read_csv(results / 'mapped_parameters.csv', index_col=0, header=[0, 1]).index
        else:
            mapped_params = []

    # Unambiguously identify metabolites
    metab_re = re.compile(r'^conc_(.*)',)
    metabolites = np.array([metab_re.match(param)[1] for param in mapped_params if metab_re.match(param)])
    return value_df, cov_df, metabolites


def save_and_return_new_res(values, covariance, new_params, output_dir=None):
    """Return and optionally save the modified dynamic results after combination/scaling

    :return values: Expanded free parameter results dataframe
    :rtype: pandas.Dataframe
    :param covariance: Expanded covariance matrix (off-diagonals of new parameters are Nan)
    :type covariance: pandas.Dataframe
    :param new_params: List of parameter names added
    :type new_params: list
    :param output_dir: Location to save outputs, defaults to None
    :type output_dir: pathlib.Path, optional
    :return values: Expanded free parameter results dataframe
    :rtype: pandas.Dataframe
    :return covariance: Expanded covariance matrix (off-diagonals of new parameters are Nan)
    :rtype: pandas.Dataframe
    :return summary_df: New free_parameters.csv with parameter means and std.
    :rtype: pandas.Dataframe
    :return new_params: List of parameter names added by function
    :rtype: list
    """
    # Form the summary df as well
    mean_free = values.mean()
    std_free = pd.Series(np.sqrt(np.diag(covariance)), index=covariance.index)
    summary_df = pd.concat((mean_free, std_free), axis=1, keys=['mean', 'sd'])

    # Optionally output to file and return key dataframes
    if output_dir:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        values.to_csv(output_dir / 'dyn_results.csv')
        covariance.to_csv(output_dir / 'dyn_cov.csv')
        summary_df.to_csv(output_dir / 'free_parameters.csv')

        return values, covariance, summary_df, new_params
    else:
        return values, covariance, summary_df, new_params
