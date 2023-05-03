"""Module for applying metabolite scalings (internal and external) to fMRS results

    Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
            Saad Jbabdi <saad@fmrib.ox.ac.uk>

    Copyright (C) 2023 University of Oxford
    # SHBASECOPYRIGHT
    """
import numpy as np

from fsl_mrs.utils.fmrs_tools import utils


def fmrs_internal_reference(results, reference_contrast, output_dir=None, samples=1E5):
    """_summary_

    Parameters are scaled to a named contrast (typically a concentration parameter).

    Calculation of the scaled covariance matrix is done using Gaussian sampling.
    The analytical calculation of ratios to correlated parameters remains difficult.
    Number of samples set to default of 1E5, but can be increased.

    :param results: _description_
    :type results: _type_
    :param reference_contrast: _description_
    :type reference_contrast: _type_
    :param output_dir: _description_
    :type output_dir: _type_, optional
    :param samples: _description_, defaults to 1E5
    :type samples: _type_, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    value_df, cov_df, _ = utils.load_dyn_res(results, mapped_p=False)
    all_contrasts = list(value_df.columns)

    if reference_contrast not in all_contrasts:
        raise ValueError(f'{reference_contrast} not in list of contrasts.')

    # Suggest that scaling to a non-concentration parameter is odd
    if 'conc' not in reference_contrast:
        import warnings
        warnings.warn('Scaling to a contrast not associated with concentrations is unusual.')

    # Calculate means
    new_value_df = value_df.copy()
    conc_columns = [x for x in new_value_df.columns if 'conc' in x]
    new_value_df[conc_columns] = new_value_df[conc_columns].divide(
        value_df.loc[:, reference_contrast],
        axis=0)

    # Calculate covariance
    denom_index = all_contrasts.index(reference_contrast)
    conc_index = [all_contrasts.index(x) for x in conc_columns]
    rng = np.random.default_rng()
    # Use 1E5 samples, possibly needs scaling for different size covariance matrices/noise levels
    samples = rng.multivariate_normal(
        value_df.mean().to_numpy(),
        np.nan_to_num(cov_df.to_numpy()),
        int(samples)).T
    samples[conc_index, :] /= samples[denom_index, :]
    cov_df.loc[:, :] = np.cov(samples)

    return utils.save_and_return_new_res(new_value_df, cov_df, [], output_dir)
