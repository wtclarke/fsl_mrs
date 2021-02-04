# synthetic.py - Create synthetic data basis sets
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

from fsl_mrs.core import MRS
from fsl_mrs.utils import dynamic
from fsl_mrs.utils.misc import parse_metab_groups
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
                If the value is a single value it will be coppied for all relavent values e.g. each metabolite group.
                If the value is the same length as the parameter group then then each value will be assigned.
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
        vm: fsl_mrs.utils.dynamic.VariableMapping object.
        syn_free_params: Value of model free parameters used in the simulation.
    """
    empty_mrs, concentrations = prep_mrs_for_synthetic(basis_file,
                                                       points,
                                                       bandwidth,
                                                       ignore,
                                                       ind_scaling,
                                                       concentrations)

    mg = parse_metab_groups(empty_mrs, metab_groups)

    model = 'voigt'
    _, _, forward, x2p, p2x = getModelFunctions(model)
    varNames, sizes = FSLModel_vars(model,
                                    n_basis=empty_mrs.numBasis,
                                    n_groups=max(mg) + 1,
                                    b_order=baseline_order)

    vm = dynamic.VariableMapping(param_names=varNames,
                                 param_sizes=sizes,
                                 time_variable=time_var,
                                 config_file=config_file)

    std_vals = {'Phi_0': 0,
                'Phi_1': 0,
                'eps': 0,
                'gamma': 10,
                'sigma': 10,
                'baseline': [0, 0] * (baseline_order + 1),
                'conc': concentrations}

    for key in defined_vals:
        if isinstance(defined_vals[key], str) \
                and defined_vals[key] in std_vals:
            defined_vals[key] = std_vals[defined_vals[key]]

    rng = np.random.default_rng()

    syn_free_params = []
    for index, param in enumerate(vm.mapped_names):
        beh = vm.Parameters[param]
        if beh == 'fixed':
            if param in defined_vals:
                if hasattr(defined_vals[param], "__len__") \
                        and len(defined_vals[param]) == vm.mapped_sizes[index]:
                    syn_free_params.extend(defined_vals[param])
                elif hasattr(defined_vals[param], "__len__"):
                    raise ValueError('Must be the same length as sizes.')
                else:
                    syn_free_params.extend([defined_vals[param], ] * vm.mapped_sizes[index])
            elif param in std_vals:
                if hasattr(std_vals[param], "__len__") \
                        and len(std_vals[param]) == vm.mapped_sizes[index]:
                    syn_free_params.extend(std_vals[param])
                elif hasattr(std_vals[param], "__len__"):
                    raise ValueError('Must be the same length as sizes.')
                else:
                    syn_free_params.extend([std_vals[param], ] * vm.mapped_sizes[index])

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
                syn_free_params.extend(rng.uniform(current_bounds[0], current_bounds[1], size=vm.mapped_sizes[index]))
        elif beh == 'variable':
            pass

        elif 'dynamic' in beh:
            dyn_name = vm.Parameters[param]['params']
            for x in range(vm.mapped_sizes[index]):
                for y in dyn_name:
                    if y in defined_vals:
                        if hasattr(defined_vals[y], "__len__") \
                                and len(defined_vals[y]) == vm.mapped_sizes[index]:
                            syn_free_params.append(defined_vals[y][x])
                        elif hasattr(defined_vals[y], "__len__"):
                            raise ValueError('Must be the same length as sizes.')
                        else:
                            syn_free_params.append(defined_vals[y])
                    elif y in std_vals:
                        if hasattr(std_vals[y], "__len__") \
                                and len(defined_vals[y]) == vm.mapped_sizes[index]:
                            syn_free_params.append(std_vals[y][x])
                        elif hasattr(std_vals[y], "__len__"):
                            raise ValueError('Must be the same length as sizes.')
                        else:
                            syn_free_params.append(std_vals[y])

                    else:
                        if vm.defined_bounds is not None \
                                and y in vm.defined_bounds:
                            current_bounds = list(vm.defined_bounds[y])
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
    for idx, _ in enumerate(mapped_noise):
        for jdx, k in enumerate(varNames):
            if param_noise is None \
                    or k not in param_noise:
                fixed_noise = np.zeros(sizes[jdx])
            else:
                fixed_noise = rng.normal(loc=param_noise[k][0],
                                         scale=param_noise[k][1],
                                         size=sizes[jdx])
            if param_rel_noise is None \
                    or k not in param_rel_noise:
                rel_noise = np.zeros(sizes[jdx])
            else:
                rel_noise = rng.normal(loc=param_rel_noise[k][0],
                                       scale=param_rel_noise[k][1],
                                       size=sizes[jdx]) \
                    * np.asarray(mapped[idx, jdx])
            mapped_noise[idx, jdx] = fixed_noise + rel_noise

    mrs_list = []
    for mm, mn in zip(mapped, mapped_noise):
        x = np.concatenate(mm) + np.concatenate(mn)

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
                              basis=empty_mrs.basis,
                              names=empty_mrs.names,
                              basis_hdr={'centralFrequency': empty_mrs.centralFrequency,
                                         'bandwidth': bandwidth})
                coils_mrs.append(mrs_out)
            mrs_list.append(coils_mrs)
        else:
            mrs_out = MRS(FID=syn_fid,
                          cf=empty_mrs.centralFrequency,
                          bw=bandwidth,
                          nucleus='1H',
                          basis=empty_mrs.basis,
                          names=empty_mrs.names,
                          basis_hdr={'centralFrequency': empty_mrs.centralFrequency,
                                     'bandwidth': bandwidth})

            mrs_list.append(mrs_out)

    return mrs_list, vm, syn_free_params
