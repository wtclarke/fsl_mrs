'''FSL-MRS test script

Test the core dynamic MRS class

Copyright Will Clarke, University of Oxford, 2021'''

import pytest
import numpy as np
import pandas as pd
import fsl_mrs.utils.synthetic as syn
from fsl_mrs.core import MRS, basis
import fsl_mrs.utils.dynamic as dyn


# Fixture returning two MRS objects (with basis) linked by a concentration scaling of 2x.
@pytest.fixture
def fixed_ratio_mrs():
    FID_basis1 = syn.syntheticFID(chemicalshift=[1, ], amplitude=[1], noisecovariance=[[0]], damping=[3])
    FID_basis2 = syn.syntheticFID(chemicalshift=[3, ], amplitude=[1], noisecovariance=[[0]], damping=[3])
    FID_basis1[1]['fwhm'] = 3 * np.pi
    FID_basis2[1]['fwhm'] = 3 * np.pi
    b = basis.Basis(
        np.stack((FID_basis1[0][0], FID_basis2[0][0]), axis=1),
        ['Met1', 'Met2'],
        [FID_basis1[1], FID_basis2[1]])

    FID1 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[1, 1], noisecovariance=[[0.01]])
    FID2 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[2, 2], noisecovariance=[[0.01]])

    mrs1 = MRS(FID=FID1[0][0], header=FID1[1], basis=b)
    mrs2 = MRS(FID=FID2[0][0], header=FID2[1], basis=b)

    mrs1.check_FID(repair=True)
    mrs1.check_Basis(repair=True)
    mrs2.check_FID(repair=True)
    mrs2.check_Basis(repair=True)

    mrs_list = [mrs1, mrs2]
    return mrs_list


def test_dynRes(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)
    res = dyn_obj.fit()

    res_obj = res['result']

    import plotly
    fig = res_obj.plot_spectra(init=True, fit_to_init=True)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)

    import matplotlib.pyplot
    fig = res_obj.plot_mapped(fit_to_init=True)
    assert isinstance(fig, matplotlib.pyplot.Figure)

    assert isinstance(res_obj.data_frame, pd.DataFrame)
    assert isinstance(res_obj.x, np.ndarray)
    assert res_obj.x.shape[0] == res_obj.data_frame.shape[1]

    assert isinstance(res_obj.mapped_parameters, np.ndarray)
    assert res_obj.mapped_parameters.shape == (1, len(mrs_list), len(dyn_obj.mapped_names))

    assert isinstance(res_obj.mapped_parameters_init, np.ndarray)
    assert res_obj.mapped_parameters_init.shape == (len(mrs_list), len(dyn_obj.mapped_names))

    assert isinstance(res_obj.free_parameters_init, np.ndarray)
    assert res_obj.free_parameters_init.shape == (len(dyn_obj.free_names),)

    assert isinstance(res_obj.init_dataframe, pd.DataFrame)
    assert res_obj.init_dataframe.shape == (1, len(dyn_obj.free_names),)

    assert isinstance(res_obj.mapped_parameters_fitted_init, np.ndarray)
    assert res_obj.mapped_parameters_fitted_init.shape == (len(mrs_list), len(dyn_obj.mapped_names))

    assert res_obj.mapped_names == dyn_obj.mapped_names
    assert res_obj.free_names == dyn_obj.free_names


def test_dynRes_newton(fixed_ratio_mrs):
    """Test newton optimiser specific components"""
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)
    res = dyn_obj.fit()

    res_obj = res['result']
    assert isinstance(res_obj.cov, pd.DataFrame)
    assert res_obj.cov.shape == (10, 10)

    assert isinstance(res_obj.corr, pd.DataFrame)
    assert res_obj.corr.shape == (10, 10)

    assert isinstance(res_obj.std, pd.Series)
    assert res_obj.std.shape == (10,)

    assert np.allclose(res_obj.std, np.sqrt(np.diagonal(res_obj.cov)))


def test_dynRes_mcmc(fixed_ratio_mrs):
    """Test mcmc optimiser specific components"""
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)

    res = dyn_obj.fit(method='MH', mh_jumps=20)

    res_obj = res['result']
    assert isinstance(res_obj.cov, pd.DataFrame)
    assert res_obj.cov.shape == (10, 10)

    assert isinstance(res_obj.corr, pd.DataFrame)
    assert res_obj.corr.shape == (10, 10)

    assert isinstance(res_obj.std, pd.Series)
    assert res_obj.std.shape == (10,)

    assert np.allclose(res_obj.std, np.sqrt(np.diagonal(res_obj.cov)))
