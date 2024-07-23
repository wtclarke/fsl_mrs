'''FSL-MRS test script

Test the dynamic VariableMapping class

Copyright Will Clarke, University of Oxford, 2023'''

import pytest
import numpy as np
from pathlib import Path

from fsl_mrs import models
from fsl_mrs.dynamic import variable_mapping as varmap

testsPath = Path(__file__).parent
config_dmi = testsPath / 'testdata' / 'dynamic' / 'dmi_model.py'
config_dmi_sigma = testsPath / 'testdata' / 'dynamic' / 'dmi_model_bounds.py'
config_dmi_dummy_init = testsPath / 'testdata' / 'dynamic' / 'dmi_model_no_init_match.py'


@pytest.fixture
def vm_obj_inputs():
    metab_names = ['Glc', 'Glx', 'water']
    numGroups = 1
    baseline_order = 0
    varNames, varSizes = models.FSLModel_vars(
        'lorentzian',
        len(metab_names),
        numGroups,
        baseline_order + 1)
    return varNames, varSizes, metab_names, numGroups


@pytest.fixture
def vm_obj(vm_obj_inputs):
    return varmap.VariableMapping(
        param_names=vm_obj_inputs[0],
        param_sizes=vm_obj_inputs[1],
        metabolite_names=vm_obj_inputs[2],
        metabolite_groups=vm_obj_inputs[3],
        time_variable=np.arange(5),
        config_file=str(config_dmi))


def test_VariableMapping(vm_obj_inputs):
    tvar = np.arange(5)

    with pytest.raises(varmap.ConfigFileError):
        varmap.VariableMapping(
            param_names=vm_obj_inputs[0],
            param_sizes=vm_obj_inputs[1],
            metabolite_names=vm_obj_inputs[2],
            metabolite_groups=vm_obj_inputs[3],
            time_variable=tvar,
            config_file=str(config_dmi_sigma))

    with pytest.raises(
            varmap.ConfigFileError,
            match='Custom init function model_dummy_init does not have matching dynamic function model_dummy.'):
        varmap.VariableMapping(
            param_names=vm_obj_inputs[0],
            param_sizes=vm_obj_inputs[1],
            metabolite_names=vm_obj_inputs[2],
            metabolite_groups=vm_obj_inputs[3],
            time_variable=tvar,
            config_file=str(config_dmi_dummy_init))

    vm_obj = varmap.VariableMapping(
        param_names=vm_obj_inputs[0],
        param_sizes=vm_obj_inputs[1],
        metabolite_names=vm_obj_inputs[2],
        metabolite_groups=vm_obj_inputs[3],
        time_variable=tvar,
        config_file=str(config_dmi))

    assert np.allclose(vm_obj.time_variable, tvar)
    assert vm_obj.ntimes == tvar.size

    # Test mapped parameter properties and functions
    assert vm_obj.mapped_categories == ['conc', 'gamma', 'eps', 'Phi_0', 'Phi_1', 'baseline']
    assert vm_obj.mapped_names == [
        'conc_Glc', 'conc_Glx', 'conc_water', 'gamma_0', 'eps_0', 'Phi_0_0', 'Phi_1_0', 'baseline_0', 'baseline_1']
    assert vm_obj.nmapped == sum(vm_obj_inputs[1])
    assert len(vm_obj.mapped_parameters) == sum(vm_obj_inputs[1])
    assert isinstance(vm_obj.mapped_parameters[0], varmap.VariableMapping._MappedParameter)
    assert vm_obj.mapped_parameters[0].function_name == 'model_exp_range'
    assert vm_obj.mapped_parameters[0].init_function() == 'model_exp_range_init'
    assert vm_obj.mapped_parameters[0].grad_function() == 'model_exp_range_grad'

    # Test free parameter properties and functions
    assert vm_obj.nfree == 12
    assert vm_obj.free_names == [
        'conc_water',
        'conc_Glc_c_rate', 'conc_Glc_c_min', 'conc_Glc_c_max',
        'conc_Glx_c_amp', 'conc_Glx_c_slope',
        'gamma_0', 'eps_0', 'Phi_0_0', 'Phi_1_0', 'baseline_0', 'baseline_1']
    assert vm_obj.free_types == [
        'fixed',
        'dynamic', 'dynamic', 'dynamic',
        'dynamic', 'dynamic',
        'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed']
    assert vm_obj.free_category == [
        'conc', 'conc', 'conc', 'conc', 'conc', 'conc', 'gamma', 'eps', 'Phi_0', 'Phi_1', 'baseline', 'baseline']
    assert vm_obj.free_met_or_group == ['water', 'Glc', 'Glc', 'Glc', 'Glx', 'Glx', 0, 0, 0, 0, 0, 1]
    assert len(vm_obj.free_to_mapped_assoc) == 12
    assert len(vm_obj.free_to_mapped_assoc[-1]) == 5
    assert vm_obj.free_to_mapped_assoc[0] == [
        'conc_water_t0', 'conc_water_t1', 'conc_water_t2', 'conc_water_t3', 'conc_water_t4']
    assert vm_obj.free_functions == [
        None,
        'model_exp_range', 'model_exp_range', 'model_exp_range',
        'model_lin', 'model_lin',
        None, None, None, None, None, None]


def test_bounds(vm_obj):
    bounds = vm_obj.Bounds
    assert len(bounds) == 12
    for param in ['conc_water', 'conc_Glc_c_rate', 'conc_Glx_c_amp', 'gamma_0']:
        assert (bounds[vm_obj.free_names.index(param)] == (0, None)).all()


def test_free_to_mapped(vm_obj):
    params = np.asarray([
        1,
        1, 0, 1,
        1, 1,
        1, 1, 1, 1, 1, 1])

    assert vm_obj.free_to_mapped(params).shape == (vm_obj.ntimes, vm_obj.nmapped)
    assert np.allclose(vm_obj.free_to_mapped(params)[:, 1], np.arange(1, 6))


def test_get_fcns(vm_obj):
    assert callable(vm_obj.get_gradient_fcn(vm_obj.mapped_parameters[0]))
    assert vm_obj.get_gradient_fcn(vm_obj.mapped_parameters[0]).__name__ == 'model_exp_range_grad'
    assert callable(vm_obj.get_gradient_fcn(vm_obj.mapped_parameters[0]))
    assert vm_obj.get_gradient_fcn(vm_obj.mapped_parameters[-1]).__name__ == '<lambda>'

    assert callable(vm_obj.get_init_fcn(vm_obj.mapped_parameters[0]))
    assert vm_obj.get_init_fcn(vm_obj.mapped_parameters[0]).__name__ == 'model_exp_range_init'
    assert callable(vm_obj.get_init_fcn(vm_obj.mapped_parameters[1]))
    assert vm_obj.get_init_fcn(vm_obj.mapped_parameters[1]).__name__ == 'default_init'

# TO DO test mapped_to_free
