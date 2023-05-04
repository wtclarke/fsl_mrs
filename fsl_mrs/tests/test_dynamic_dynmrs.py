'''FSL-MRS test script

Test the core dynamic MRS class

Copyright Will Clarke, University of Oxford, 2021'''

import pytest
import numpy as np

import fsl_mrs.utils.synthetic as syn
from fsl_mrs.core import MRS, basis
import fsl_mrs.dynamic as dyn
from fsl_mrs.dynamic.dynmrs import dynMRSArgumentError
from fsl_mrs.dynamic.variable_mapping import ConfigFileError


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


def test_fixtures(fixed_ratio_mrs):
    assert fixed_ratio_mrs[0].names == ['Met1', 'Met2']
    assert fixed_ratio_mrs[1].names == ['Met1', 'Met2']


def test_dynMRS_setup(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)

    # Test properties
    assert dyn_obj.metabolite_names == ['Met1', 'Met2']
    assert dyn_obj.free_names == [
        'conc_Met1_c_0',
        'conc_Met1_c_g',
        'conc_Met2_c_0',
        'conc_Met2_c_g',
        'gamma_0',
        'eps_0',
        'Phi_0_0',
        'Phi_1_0',
        'baseline_0',
        'baseline_1']
    assert dyn_obj.mapped_names == [
        'conc_Met1',
        'conc_Met2',
        'gamma_0',
        'eps_0',
        'Phi_0_0',
        'Phi_1_0',
        'baseline_0',
        'baseline_1']

    assert isinstance(dyn_obj.vm, dyn.VariableMapping)


def test_process_mrs_list(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)

    assert dyn_obj.mrs_list[0].scaling['FID'] == 1.0
    assert dyn_obj.mrs_list[0].scaling['basis'] == 1.0

    dyn_obj_scaled = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=True)

    assert dyn_obj_scaled.mrs_list[0].scaling['FID'] > 1.0
    assert dyn_obj_scaled.mrs_list[0].scaling['basis'] > 1.0
    assert dyn_obj_scaled.mrs_list[0].scaling['FID'] == dyn_obj_scaled.mrs_list[1].scaling['FID']


# Test Utility methods
def test_get_constants(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)

    consts = dyn_obj._get_constants(
        mrs_list[0],
        (0.2, 4.2),
        0,
        [0, 0])

    assert len(consts) == 8
    assert np.allclose(consts[0], mrs_list[0].frequencyAxis)
    assert np.allclose(consts[1], mrs_list[0].timeAxis)
    assert np.allclose(consts[2], mrs_list[0].basis)
    assert consts[3].shape == (2048, 2)
    assert consts[4] == [0, 0]
    assert consts[5] == 1
    assert consts[6] == mrs_list[0].ppmlim_to_range((0.2, 4.2), True)[0]
    assert consts[7] == mrs_list[0].ppmlim_to_range((0.2, 4.2), True)[1]


def test_dynMRS_fit(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)
    init = dyn_obj.initialise(indiv_init=None)
    res = dyn_obj.fit(init=init)

    concs = res.dataframe_free.filter(like='conc').to_numpy()
    assert np.allclose(concs, [1, 1, 1, 1], atol=0.1)


def test_dynMRS_fit_mcmc(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)
    init = dyn_obj.initialise(indiv_init=None)
    res = dyn_obj.fit(method='MH', mh_jumps=50, init=init)

    concs = res.dataframe_free.filter(like='conc').mean(axis=0).to_numpy()
    assert np.allclose(concs, [1, 1, 1, 1], atol=0.1)


def test_dynMRS_mean_fit_init(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)

    # check mean fit gives same results as fit of the mean mrs lsit
    from fsl_mrs.utils.preproc.combine import combine_FIDs
    from fsl_mrs.utils import fitting
    from copy import deepcopy
    mean_fid = combine_FIDs([mrs.FID for mrs in dyn_obj.mrs_list], 'mean')
    mean_mrs = deepcopy(dyn_obj.mrs_list[0])
    mean_mrs.FID = mean_fid

    meanres = fitting.fit_FSLModel(mean_mrs, method='Newton', **dyn_obj._fit_args)
    assert np.allclose(meanres.params, dyn_obj.fit_mean_spectrum())

    # Check the init produces the same results via both interfaces
    init1 = dyn_obj.initialise(indiv_init=meanres.params)
    init2 = dyn_obj.initialise(indiv_init='mean')
    assert np.allclose(np.hstack(np.hstack(init1['x'])), np.hstack(np.hstack(init2['x'])))


def test_save_load(tmp_path, fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    dyn_obj = dyn.dynMRS(
        mrs_list,
        [0, 1],
        'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
        model='lorentzian',
        baseline_order=0,
        metab_groups=[0, 0],
        rescale=False)

    dyn_obj.save(tmp_path / 'test_save')
    dyn_obj_load = dyn.dynMRS.load(tmp_path / 'test_save', mrs_list=mrs_list)

    assert dyn_obj.metabolite_names == dyn_obj_load.metabolite_names
    assert dyn_obj.free_names == dyn_obj_load.free_names
    assert dyn_obj.mapped_names == dyn_obj_load.mapped_names
    assert dyn_obj.vm.fcns.keys() == dyn_obj_load.vm.fcns.keys()
    assert np.allclose(dyn_obj.time_var, dyn_obj_load.time_var)

    dyn_obj.save(tmp_path / 'test_save2', save_mrs_list=True)
    dyn_obj_load2 = dyn.dynMRS.load(tmp_path / 'test_save2')

    assert dyn_obj.metabolite_names == dyn_obj_load2.metabolite_names
    assert dyn_obj.free_names == dyn_obj_load2.free_names
    assert dyn_obj.mapped_names == dyn_obj_load2.mapped_names
    assert dyn_obj.vm.fcns.keys() == dyn_obj_load2.vm.fcns.keys()
    assert np.allclose(dyn_obj.time_var, dyn_obj_load2.time_var)


def test_errors(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    # Wrong number of MRS objects in list
    with pytest.raises(
            dynMRSArgumentError,
            match=r'Number of time steps \(currently \d+\) must match mrs_list length \(\d+\)'):
        _ = dyn.dynMRS(
            mrs_list[0:1],
            [0, 1],
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)

    # Wrong length time variables
    with pytest.raises(
            dynMRSArgumentError,
            match=r'Number of time steps \(currently \d+\) must match mrs_list length \(\d+\)'):
        _ = dyn.dynMRS(
            mrs_list,
            [0, ],
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)

    # Wonky time variable definition
    with pytest.raises(
            dynMRSArgumentError,
            match=r'All values in time_var dict must have the same first dimension shape.'):
        _ = dyn.dynMRS(
            mrs_list,
            {'param1': [0, 1], 'param2': [0, ]},
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)

    # Wrong length time variables (dict format)
    with pytest.raises(
            dynMRSArgumentError,
            match=r'Number of time steps \(currently \d+\) must match mrs_list length \(\d+\)'):
        _ = dyn.dynMRS(
            mrs_list,
            {'param1': [0, ], 'param2': [0, ]},
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)


# This should be moved to the (non-existant) Variable mapping test function
def test_vm_errors(fixed_ratio_mrs):
    mrs_list = fixed_ratio_mrs

    # Bad config: bounds name
    with pytest.raises(
            ConfigFileError,
            match=r'Not all bounds are used, remove or rename. Extra bounds:.+'):
        _ = dyn.dynMRS(
            mrs_list,
            [0, 1],
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model_badbounds.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)

    # Bad config: mode
    with pytest.raises(
            ConfigFileError,
            match=r'Unknown parameter mode \([a-z]+\) in configuration.*'):
        _ = dyn.dynMRS(
            mrs_list,
            [0, 1],
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model_badmode.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)

    # Bad config: no function
    with pytest.raises(
            ConfigFileError,
            match=r'\w+ for type \w+ \(parameter: .+\) not found in config file'):
        _ = dyn.dynMRS(
            mrs_list,
            [0, 1],
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model_badfunc.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)

    # Bad config: no grad function
    with pytest.raises(
            ConfigFileError,
            match=r'Could not find gradient function \w+ for parameter .+ / function \w+'):
        _ = dyn.dynMRS(
            mrs_list,
            [0, 1],
            'fsl_mrs/tests/testdata/dynamic/simple_linear_model_badgrad.py',
            model='lorentzian',
            baseline_order=0,
            metab_groups=[0, 0],
            rescale=False)
