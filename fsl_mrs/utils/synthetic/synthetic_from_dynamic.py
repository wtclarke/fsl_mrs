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
                                 baseline_order=0,
                                 concentrations=None,
                                 defined_vals=None,
                                 param_noise=None,
                                 bandwidth=4000,
                                 points=2048,
                                 baseline_ppm=None,
                                 coilamps=[1.0],
                                 coilphase=[0.0],
                                 noisecovariance=[[0.1]]):

    empty_mrs, concentrations = prep_mrs_for_synthetic(basis_file,
                                                       points,
                                                       bandwidth,
                                                       ignore,
                                                       ind_scaling,
                                                       concentrations)

    mg = parse_metab_groups(empty_mrs, metab_groups)

    model = 'voigt'
    _, _, forward, _, p2x = getModelFunctions(model)
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
                'baseline': (0, 0),
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
                        and len(defined_vals[param]) == vm.mapped_sizes[index]:
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
                mapped_noise[idx, jdx] = np.zeros(sizes[jdx])
            else:
                mapped_noise[idx, jdx] = rng.normal(loc=param_noise[k][0],
                                                    scale=param_noise[k][1],
                                                    size=sizes[jdx])

    mrs_list = []
    for mm, mn in zip(mapped, mapped_noise):
        x = np.concatenate(mm) + np.concatenate(mn)
        syn_fid = synthetic_from_fwd_model(forward,
                                           x,
                                           empty_mrs,
                                           baseline_order,
                                           mg,
                                           ppmlim=baseline_ppm,
                                           coilamps=coilamps,
                                           coilphase=coilphase,
                                           noisecovariance=noisecovariance)

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
