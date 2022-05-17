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
    res_obj = dyn_obj.fit()

    resinit = dyn_obj.initialise()

    import plotly
    fig = res_obj.plot_spectra(init=True, fit_to_init=True)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)

    import matplotlib.pyplot
    fig = res_obj.plot_mapped(fit_to_init=True)
    assert isinstance(fig, matplotlib.pyplot.Figure)

    assert isinstance(res_obj.dataframe_free, pd.DataFrame)
    assert isinstance(res_obj.x, np.ndarray)
    assert res_obj.x.shape[0] == res_obj.dataframe_free.shape[1]

    assert isinstance(res_obj._init_x, pd.DataFrame)
    assert np.allclose(dyn_obj.vm.mapped_to_free(resinit['x']), res_obj.init_free_parameters)

    assert isinstance(res_obj.mapped_parameters_array, np.ndarray)
    assert res_obj.mapped_parameters_array.shape == (1, len(mrs_list), len(dyn_obj.mapped_names))

    assert isinstance(res_obj.init_mapped_parameters_array, np.ndarray)
    assert res_obj.init_mapped_parameters_array.shape == (len(mrs_list), len(dyn_obj.mapped_names))

    assert isinstance(res_obj.init_free_parameters, np.ndarray)
    assert res_obj.init_free_parameters.shape == (len(dyn_obj.free_names),)

    assert isinstance(res_obj.init_free_dataframe, pd.Series)
    assert res_obj.init_free_dataframe.shape == (len(dyn_obj.free_names),)

    assert isinstance(res_obj.init_mapped_parameters_fitted_array, np.ndarray)
    assert res_obj.init_mapped_parameters_fitted_array.shape == (len(mrs_list), len(dyn_obj.mapped_names))

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
    res_obj = dyn_obj.fit()

    assert isinstance(res_obj.cov_free, pd.DataFrame)
    assert res_obj.cov_free.shape == (10, 10)

    assert isinstance(res_obj.corr_free, pd.DataFrame)
    assert res_obj.corr_free.shape == (10, 10)

    assert isinstance(res_obj.std_free, pd.Series)
    assert res_obj.std_free.shape == (10,)

    assert np.allclose(res_obj.std_free, np.sqrt(np.diagonal(res_obj.cov_free)))

    assert isinstance(res_obj.std_mapped, pd.DataFrame)
    assert res_obj.std_mapped.shape == (2, 8)

    assert isinstance(res_obj.dataframe_mapped, pd.DataFrame)
    assert res_obj.dataframe_mapped.shape == (2, 8)

    assert isinstance(res_obj.reslist, list)
    assert len(res_obj.reslist) == 2


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

    res_obj = dyn_obj.fit(method='MH', mh_jumps=20)

    assert isinstance(res_obj.cov_free, pd.DataFrame)
    assert res_obj.cov_free.shape == (10, 10)

    assert isinstance(res_obj.corr_free, pd.DataFrame)
    assert res_obj.corr_free.shape == (10, 10)

    assert isinstance(res_obj.std_free, pd.Series)
    assert res_obj.std_free.shape == (10,)

    assert np.allclose(res_obj.std_free, np.sqrt(np.diagonal(res_obj.cov_free)))

    assert isinstance(res_obj.std_mapped, pd.DataFrame)
    assert res_obj.std_mapped.shape == (2, 8)

    assert isinstance(res_obj.dataframe_mapped, pd.DataFrame)
    assert res_obj.dataframe_mapped.shape == (2, 8)


def test_load_save(fixed_ratio_mrs, tmp_path):
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

    res.save(tmp_path / 'res_save_test')
    res_loaded = dyn.load_dyn_result(tmp_path / 'res_save_test', dyn_obj)

    from pandas._testing import assert_frame_equal
    assert_frame_equal(res._data, res_loaded._data)
    assert_frame_equal(res._init_x, res_loaded._init_x)

    assert (tmp_path / 'res_save_test' / 'free_parameters.csv').is_file()
    assert (tmp_path / 'res_save_test' / 'mapped_parameters.csv').is_file()

    res.save(tmp_path / 'res_save_test2', save_dyn_obj=True)
    res_loaded2 = dyn.load_dyn_result(tmp_path / 'res_save_test2')

    assert_frame_equal(res._data, res_loaded2._data)
    assert_frame_equal(res._init_x, res_loaded2._init_x)
