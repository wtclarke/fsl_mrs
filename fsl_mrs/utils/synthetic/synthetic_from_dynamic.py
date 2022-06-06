# synthetic.py - Create synthetic data basis sets
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

from fsl_mrs.core import MRS
from fsl_mrs import dynamic
from fsl_mrs.utils.models import getModelFunctions, FSLModel_vars
from fsl_mrs.utils.synthetic.synthetic_from_basis import prep_mrs_for_synthetic, synthetic_from_fwd_model
import numpy as np


def synthetic_spectra_from_model(config_file,
                                 time_var,
                                 basis_file,
                                 ignore=None,
                                 metab_groups=None,
                                 ind_scaling=None,
                                 concentrations=None,
                                 baseline_order=0,
                                 baseline_ppm=None,
                                 defined_vals=None,
                                 param_noise=None,
                                 param_rel_noise=None,
                                 coilamps=[1.0],
                                 coilphase=[0.0],
                                 noisecovariance=[[0.1]],
                                 bandwidth=4000,
                                 points=2048):
    """ Create synthetic dynamic data from FSL-MRS basis file and dynamic configuration file.
        Model parameters may be specified using the defined_vals argument. Otheriwse values are randomly set between
            the model defined bounds (or -1 and 1 if bounds not set).
    Args:
            config_file (str): path to dynamic configuration file.
            time_var (list of floats): Dynamic 'time' axis.
            basisFile (str): path to directory containg basis spectra json files
            ignore (list of str, optional): Ignore metabolites in basis set.
            metab_groups (list of str, optional): Group metabolites to apply different model parameters to groups.
            ind_scaling (list of str, optional): Independently scale basis spectra in the basis set.
            concentrations (list or dict or None, optional ): If None, standard concentrations will be used.
                If list of same length as basis spectra, then these will be used.
                Pass dict to overide standard values for specific metabolites.
                Key should be metabolite name.
            baseline_order (int, optional): Order of baseline to simulate.
            baseline_ppm (tuple, optional): Specify ppm range over which baseline is calculated.
            defined_vals (dict, optional): Model parameters can be specified by adding a key of the same name.
                If set to one of 'conc', 'gamma', 'sigma', 'eps', 'baseline' then standard values will be used.
            param_noise (dict, optional): Add fixed Gaussian noise to 'conc', 'gamma', 'sigma', 'eps', 'baseline' at
                each timepoint. Specified as (mean, SD) for each key.
            param_rel_noise (dict, optional): Add relative Gaussian noise to 'conc', 'gamma', 'sigma', 'eps', 'baseline'
                at each timepoint. Specified as (mean, SD) for each key.
            coilamps (list of floats, optional): If multiple coils, specify magnitude scaling.
            coilphase (list of floats, optional): If multiple coils, specify phase.
            noisecovariance (list of floats, optional): N coils x N coils array of noise variance/covariance.
            bandwidth (float,optional): Bandwidth of output spectrum in Hz
            points (int,optional): Number of points in output spectrum.

    Returns:
        mrs_list: Numpy array of synthetic MRS objects at each timepoint.
        vm: fsl_mrs.dynamic.VariableMapping object.
        syn_free_params: Value of model free parameters used in the simulation.
    """
    empty_mrs, concentrations, mg = prep_mrs_for_synthetic(basis_file,
                                                           points,
                                                           bandwidth,
                                                           ignore,
                                                           ind_scaling,
                                                           concentrations,
                                                           metab_groups)

    model = 'voigt'
    _, _, forward, x2p, p2x = getModelFunctions(model)
    varNames, sizes = FSLModel_vars(model,
                                    n_basis=empty_mrs.numBasis,
                                    n_groups=max(mg) + 1,
                                    b_order=baseline_order)

    vm = dynamic.VariableMapping(param_names=varNames,
                                 param_sizes=sizes,
                                 metabolite_names=empty_mrs.names,
                                 metabolite_groups=mg,
                                 time_variable=time_var,
                                 config_file=config_file)

    std_vals = {'Phi_0': 0,
                'Phi_1': 0,
                'eps': 0,
                'gamma': 10,
                'sigma': 10,
                'baseline': 0,
                'conc': concentrations}

    def_vals_int = {}
    if defined_vals is not None:
        for key in defined_vals:
            if isinstance(defined_vals[key], str) \
                    and defined_vals[key] in std_vals:
                def_vals_int[key] = std_vals[defined_vals[key]]
            else:
                def_vals_int[key] = defined_vals[key]
    rng = np.random.default_rng()

    syn_free_params = []
    for param in vm._free_params:
        if param.name in def_vals_int:
            syn_free_params.append(def_vals_int[param.name])
        elif param.mapped_category in std_vals:
            if param.mapped_category == 'conc':
                for idx, name in enumerate(empty_mrs.names):
                    if f'_{name}_' in param.name:
                        syn_free_params.append(std_vals[param.mapped_category][idx])
            else:
                syn_free_params.append(std_vals[param.mapped_category])
        else:
            if vm.defined_bounds is not None \
                    and param in vm.defined_bounds:
                current_bounds = list(vm.defined_bounds[param])
                if current_bounds[0] is None:
                    current_bounds[0] = -1
                if current_bounds[1] is None:
                    current_bounds[1] = 1
            else:
                current_bounds = [-1, 1]
            syn_free_params.append(rng.uniform(current_bounds[0], current_bounds[1]))

    syn_free_params = np.asarray(syn_free_params)

    mapped = vm.free_to_mapped(syn_free_params)

    # Amount of noise to add to each parameter in each timepoint
    mapped_noise = np.empty(mapped.shape, dtype=object)
    for idx, mp_obj in enumerate(vm.mapped_parameters):
        if param_noise is None \
                or mp_obj.category not in param_noise:
            fixed_noise = 0
        else:
            fixed_noise = rng.normal(loc=param_noise[mp_obj.category][0],
                                     scale=param_noise[mp_obj.category][1])
        if param_rel_noise is None \
                or mp_obj.category not in param_rel_noise:
            rel_noise = 0 * np.asarray(mapped[:, idx])
        else:
            rel_noise = rng.normal(loc=param_rel_noise[mp_obj.category][0],
                                   scale=param_rel_noise[mp_obj.category][1]) \
                * np.asarray(mapped[:, idx])
        mapped_noise[:, idx] = fixed_noise + rel_noise

    mrs_list = []
    for mm, mn in zip(mapped, mapped_noise):
        x = mm + mn

        con, gamma, sigma, eps, phi0, phi1, b = x2p(x,
                                                    empty_mrs.numBasis,
                                                    max(mg) + 1)
        con[con < 0] = 0
        gamma[gamma < 0] = 0
        sigma[sigma < 0] = 0
        x = p2x(con, gamma, sigma, eps, phi0, phi1, b)

        syn_fid = synthetic_from_fwd_model(forward,
                                           x,
                                           empty_mrs,
                                           baseline_order,
                                           mg,
                                           ppmlim=baseline_ppm,
                                           coilamps=coilamps,
                                           coilphase=coilphase,
                                           noisecovariance=noisecovariance)
        if syn_fid.ndim > 1:
            coils_mrs = []
            for fid in syn_fid.T:
                mrs_out = MRS(FID=fid,
                              cf=empty_mrs.centralFrequency,
                              bw=bandwidth,
                              nucleus='1H',
                              basis=empty_mrs._basis)
                # Sort out basis scaling
                mrs_out._indept_scale = empty_mrs._indept_scale
                mrs_out._scaling_factor = empty_mrs._scaling_factor
                mrs_out.ignore = empty_mrs.ignore
                coils_mrs.append(mrs_out)
            mrs_list.append(coils_mrs)
        else:
            mrs_out = MRS(FID=syn_fid,
                          cf=empty_mrs.centralFrequency,
                          bw=bandwidth,
                          nucleus='1H',
                          basis=empty_mrs._basis)
            # Sort out basis scaling
            mrs_out._indept_scale = empty_mrs._indept_scale
            mrs_out._scaling_factor = empty_mrs._scaling_factor
            mrs_out.ignore = empty_mrs.ignore
            mrs_list.append(mrs_out)

    return mrs_list, vm, syn_free_params
