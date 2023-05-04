# variable_mapping.py - Class responsible for variable mapping
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

from dataclasses import dataclass
from typing import Union
from functools import partial

import numpy as np
from scipy.optimize import minimize


class ConfigFileError(Exception):
    pass


class VariableMapping(object):
    def __init__(self,
                 param_names,
                 param_sizes,
                 metabolite_names,
                 metabolite_groups,
                 time_variable,
                 config_file):
        """
        Variable Mapping Class Constructor

        Mapping between free and mapped:
        Mapped = TxN matrix
        Mapped[i,j] = float or 1D-array of floats with size param_sizes[j]



        Parameters
        ----------
        param_names  : list
        param_sizes  : list
        time_variable : array-like or dict
        config_file  : string
        """

        if isinstance(time_variable, dict):
            self._time_variable = {}
            t_size = []
            for key in time_variable:
                t_element = np.asarray(time_variable[key])
                t_size.append(t_element.shape[0])
                self._time_variable.update({key: t_element})
            t_size = np.asarray(t_size)
            if np.all(np.isclose(t_size, t_size[0])):
                self._ntimes = t_size[0]
            else:
                raise ValueError('All values in time_variable dict must have the same first dimension shape.')
        else:
            self._time_variable = np.asarray(time_variable)
            self._ntimes = self._time_variable.shape[0]

        self._metabolites = metabolite_names
        self._n_groups = metabolite_groups

        self._mapped_categories = param_names  # The catergories, e.g. 'conc', 'eps' etc.
        self._mapped_sizes      = param_sizes  # The number of parameters in each category
        # Calculate the mapped names
        self._mapped_names = []
        for mc, sizes in zip(self._mapped_categories, self._mapped_sizes):
            # Special case for the concentrations - need to link to metabolites
            if mc == 'conc':
                self._mapped_names.extend([f'conc_{met}' for met in self._metabolites])
            else:
                self._mapped_names.extend([f'{mc}_{idx}' for idx in range(sizes)])

        # Process the config file
        from runpy import run_path
        settings = run_path(config_file)
        self._Parameters = settings['Parameters']
        for name in self._mapped_categories:
            if name not in self._Parameters:
                self._Parameters[name] = 'fixed'
        self.fcns = {}
        for key in settings:
            if callable(settings[key]):
                self.fcns[key] = settings[key]
        if 'Bounds' in settings:
            self.defined_bounds = settings['Bounds']
        else:
            self.defined_bounds = None

        # Form view of free and mapped parameter relations, and store
        self._free_params = self._calculate_free_params()
        self._mapped_params = self._calculate_mapped_parameters()

        # Check bounds by calling (and discarding) bound property
        _ = self.Bounds
        # Check functions and gradient functions exist
        for mp_obj in self._mapped_params:
            if mp_obj.param_type == 'dynamic':
                if mp_obj.function_name not in self.fcns:
                    raise ConfigFileError(
                        f'{mp_obj.function_name} for type {mp_obj.category}'
                        f' (parameter: {mp_obj.name}) not found in config file.')
                _ = self.get_gradient_fcn(mp_obj)

    def __str__(self):
        OUT  = '-----------------------\n'
        OUT += 'Variable Mapping Object\n'
        OUT += '-----------------------\n'
        OUT += f'Number of Mapped param groups  = {len(self.mapped_categories)}\n'
        OUT += f'Number of Mapped params        = {sum(self._mapped_sizes)}\n'
        OUT += f'Number of Free params          = {self.nfree}\n'
        OUT += f'Number of params if all indep  = {sum(self._mapped_sizes)*self.ntimes}\n'

        OUT += 'Dynamic functions\n'
        for param_name in self.mapped_categories:
            beh = self._Parameters[param_name]
            OUT += f'{param_name} \t  {beh}\n'

        return OUT

    def __repr__(self) -> str:
        return str(self)

    @property
    def time_variable(self):
        """Time (dynamic) variable.

        :return: List of time variable values
        :rtype: List
        """
        return self._time_variable

    @property
    def ntimes(self):
        """Number of time steps

        :return: Integer number of time steps
        :rtype: int
        """
        return self._ntimes

    # Properties for mapped parameters
    @property
    def mapped_categories(self):
        """List of mapped parameter categories,e.g. "conc"

        :return: List of categories as strings
        :rtype: List
        """
        return self._mapped_categories

    @property
    def mapped_names(self):
        """Mapped parameter names (excluding time point repetitions)

        :return: List of constructed name strings
        :rtype: List
        """
        return self._mapped_names

    @property
    def nmapped(self):
        """Number of mapped parameters

        :return: Integer number of mapped parameters
        :rtype: int
        """
        return len(self._mapped_names)

    @dataclass
    class _MappedParameter:
        """Class to keep track of infromation about each mapped parameter."""
        name: str
        category: str
        param_type: str
        free_indices: tuple
        function_name: str = None

        def grad_function(self):
            """Return gradient function name"""
            return self.function_name + '_grad'

    @property
    def mapped_parameters(self):
        """Return view of all mapped parameter objects

        :return: List of _MappedParameter objects
        :rtype: List
        """
        return self._mapped_params

    def _calculate_mapped_parameters(self):
        """Create an array of _MappedParameter objects to keep track of Mapped parameter information

        :return: List of _MappedParameter objects
        :rtype: List
        """
        def gen_mp_obj(name):
            """Nested function to create an mapped parameter object from a name."""
            free_pars = []
            for t in range(self.ntimes):
                t_name = name + f'_t{t}'
                free_pars.extend([idx for idx, vals in enumerate(self.free_to_mapped_assoc) if t_name in vals])
            free_pars = np.unique(free_pars)

            # Warning
            # These next lines might go badly wrong at some point if free parameters are ever shared
            function = self.free_functions[free_pars[0]]
            ptype = self.free_types[free_pars[0]]
            category = self.free_category[free_pars[0]]
            # End of warning

            return self._MappedParameter(name, category, ptype, free_pars, function)

        mapped_params = []
        for name in self.mapped_names:
            mapped_params.append(gen_mp_obj(name))
        return mapped_params

    # Properties for free parameters
    @dataclass
    class _FreeParameter:
        """Class to keep track of infromation about each free parameter."""
        name: str
        mapped_category: str
        param_type: str
        mapped_param: str
        met_or_group: Union[str, int] = None
        function_name: str = None

    @property
    def free_names(self):
        """
        list of names for free params

        Returns
        -------
        list of strings
        """
        return [x.name for x in self._free_params]

    @property
    def free_types(self):
        """
        list of free param types

        Returns
        -------
        list of strings
        """
        return [x.param_type for x in self._free_params]

    @property
    def free_category(self):
        """
        list of free param types

        Returns
        -------
        list of strings
        """
        return [x.mapped_category for x in self._free_params]

    @property
    def free_met_or_group(self):
        """
        list of free param types

        Returns
        -------
        list of strings
        """
        return [x.met_or_group for x in self._free_params]

    @property
    def free_to_mapped_assoc(self):
        """
        list of mapped parameters that free parameters are associated with.

        Returns
        -------
        list of strings
        """
        return [x.mapped_param for x in self._free_params]

    @property
    def free_functions(self):
        """
        list of free param dynamic functions. None if fixed or variable

        Returns
        -------
        list of strings
        """
        return [x.function_name if x.param_type == 'dynamic' else None for x in self._free_params]

    def _calculate_free_params(self):
        """
        Calculate list of free parameters, including names,
        types and associated dynamic functions

        Returns
        -------
        list of _FreeParameters objects
        """

        def process_param(ptype, cat, group_or_mets):
            """Nested function to Generate _FreeParameter object

            :param ptype: Parameter type: fixed, variable or dynamic
            :type ptype: str
            :param cat: Parameter category e.g. conc, eps etc.
            :type cat: str
            :param group_or_mets: List of metabolites or groups
            :type group_or_mets: list
            """
            def gen_mapped(stem):
                return [stem + f'_t{t}' for t in range(self.ntimes)]

            if (ptype == 'fixed'):
                return [self._FreeParameter(f'{cat}_{x}', cat, 'fixed', gen_mapped(f'{cat}_{x}'), x)
                        for x in group_or_mets]
            elif (ptype == 'variable'):
                return [self._FreeParameter(f'{cat}_{x}_t{t}', cat, 'variable', f'{cat}_{x}_t{t}', x)
                        for t in range(self.ntimes) for x in group_or_mets]
            elif 'dynamic' in ptype:
                dyn_name = ptype['params']
                return [self._FreeParameter(f'{cat}_{x}_{y}', cat, 'dynamic',
                                            gen_mapped(f'{cat}_{x}'), x, ptype['dynamic'])
                        for x in group_or_mets for y in dyn_name]
            else:
                raise ConfigFileError(
                    f"Unknown parameter mode ({ptype}) in configuration "
                    "- should be one of 'fixed', 'variable', {'dynamic'}")

        fparams = []
        for index, param in enumerate(self.mapped_categories):

            # Concentration parameters can be separated per-metabolite
            if param == 'conc':
                if isinstance(self._Parameters[param], dict)\
                        and any([key in self._metabolites for key in self._Parameters[param]]):
                    other_metabs = [m for m in self._metabolites if m not in self._Parameters[param].keys()]
                    for key in self._Parameters[param]:
                        if key in self._metabolites:
                            fparams.extend(process_param(self._Parameters[param][key], param, [key, ]))
                        elif key == 'other':
                            fparams.extend(process_param(self._Parameters[param]['other'], param, other_metabs))
                        else:
                            raise ConfigFileError(
                                f'Key in nested dynamic "conc" definition must be a known metabolite name or "other",'
                                f' not {key}.'
                                f' Known metabolites: {self._metabolites}.')
                else:
                    fparams.extend(process_param(self._Parameters[param], param, self._metabolites))

            # Shift and linewidth parameters can be separated per metabolite group
            elif param in ['eps', 'sigma', 'gamma']:
                if isinstance(self._Parameters[param], dict)\
                        and any([key.isdigit() for key in self._Parameters[param]]):
                    other_groups = [g for g in range(self._mapped_sizes[index])
                                    if str(g) not in self._Parameters[param].keys()]
                    for key in self._Parameters[param]:
                        if key.isdigit() and int(key) < self._mapped_sizes[index]:
                            fparams.extend(process_param(self._Parameters[param][key], param, [int(key), ]))
                        elif key == 'other':
                            fparams.extend(process_param(self._Parameters[param][key], param, other_groups))
                        else:
                            raise ConfigFileError(
                                f'Key in nested {param} must be a group index '
                                f'(below {self._mapped_sizes[index]}) or "other", not {key}.')
                else:
                    group_list = list(range(self._mapped_sizes[index]))
                    fparams.extend(process_param(self._Parameters[param], param, group_list))

            # Other parameter categories are fixed
            else:
                group_list = list(range(self._mapped_sizes[index]))
                fparams.extend(process_param(self._Parameters[param], param, group_list))
        return fparams

    @property
    def nfree(self):
        """
        Number of free parameters

        Returns
        -------
        int
        """
        return len(self.free_names)

    @property
    def Bounds(self):
        """
        List of constraints on free parameters to be used in optimization

        Returns
        -------
        list
        """

        if self.defined_bounds is None:
            return [(None, None)] * self.nfree

        if not isinstance(self.defined_bounds, dict):
            raise TypeError('defined_bounds should either be a dict or None')

        # Keep track of which bounds are used for error checking at end
        used_bounds = []
        # Form list of bounds
        b = []  
        for free_p, free_t in zip(self.free_names, self.free_types):
            # First look for an exact match, i.e. a bound on the precise parameter
            if free_p in self.defined_bounds:
                b.append(self.defined_bounds[free_p])
                used_bounds.append(free_p)
            else:
                # If a dynamic function look for a bound specific to the dynamic function
                if free_t == 'dynamic':
                    mod_p_name = '_'.join(free_p.split('_')[2:])
                    if mod_p_name in self.defined_bounds:
                        b.append(self.defined_bounds[mod_p_name])
                        used_bounds.append(mod_p_name)
                    else:
                        b.append((None, None))
                # If fixed or variable look for a bound on the generic form
                elif free_t in ['fixed', 'variable']:
                    mod_p_name = free_p.split('_')[0]
                    if mod_p_name in self.defined_bounds:
                        b.append(self.defined_bounds[mod_p_name])
                        used_bounds.append(mod_p_name)
                    else:
                        b.append((None, None))

        # Check for unused bounds and raise error
        used_bounds = set(np.unique(used_bounds))
        all_bounds = self.defined_bounds.keys()
        if len(all_bounds - used_bounds) > 0:
            bounds_str = ', '.join(list(all_bounds - used_bounds))
            raise ConfigFileError(f'Not all bounds are used, remove or rename. Extra bounds: {bounds_str}.')

        return np.asarray(b)

    def free_to_mapped(self, p):
        """
        Convert free into mapped params over time
        fixed params get copied over time domain
        variable params are indep over time
        dynamic params are mapped using dyn model

        Parameters
        ----------
        p : 1D array
        Returns
        -------
        2D array (time X params)

        """
        # Check input
        if (p.size != self.nfree):
            raise ValueError(
                'Input free params does not have expected number of entries.'
                f' Found {p.size}, expected {self.nfree}')

        # Mapped params is time X nparams (each param is an array of params)
        mapped_params = np.zeros((self.ntimes, self.nmapped))

        for index, mp_obj in enumerate(self._mapped_params):
            if mp_obj.param_type == 'fixed':
                mapped_params[:, index] = p[mp_obj.free_indices] * np.ones(self.ntimes)
            elif mp_obj.param_type == 'variable':
                mapped_params[:, index] = p[mp_obj.free_indices]
            elif mp_obj.param_type == 'dynamic':
                # Generate time courses
                func_name = mp_obj.function_name
                params = p[mp_obj.free_indices]
                mapped_params[:, index] = self.fcns[func_name](params, self.time_variable)
            else:
                raise ConfigFileError(
                    f"Unknown parameter mode ({mp_obj.param_type}) in configuration "
                    "- should be one of 'fixed', 'variable', {'dynamic'}")

        return mapped_params

    def print_free(self, x):
        """
        Print free params and their names
        """
        print(dict(zip(self.free_names, x)))

    def check_bounds(self, x, tol=1e-10):
        """
        Check that bounds apply and return corrected x
        """
        if self.Bounds is None:
            return x

        for i, b in enumerate(self.Bounds):
            LB = b[0] if b[0] is not None else -np.inf
            UB = b[1] if b[1] is not None else np.inf
            if (x[i] < LB):
                x[i] = LB + tol
            if (x[i] > UB):
                x[i] = UB - tol
        return x

    # This function may 'invert' the dynamic mapping
    # if the input params are from a single timepoint it assumes constant
    def mapped_to_free(self, p):
        """
        Convert mapped params to free (e.g. to initialise the free params)
        fixed and variable params are simply copied
        dynamic params are converted by inverting the dyn model with Scipy optimize

        Parameters
        ----------
        p : 2D array (time X params)

        Returns
        -------
        1D array
        """
        # Check input
        if isinstance(p, list):
            p = np.asarray(p)

        if (p.shape != (self.ntimes, self.nmapped)):
            raise ValueError('Input mapped params does not have expected number of entries.'
                             f' Found {p.shape}, expected {(self.ntimes,self.nmapped)}')

        free_params = np.zeros(self.nfree)
        for index, mp_obj in enumerate(self._mapped_params):
            if mp_obj.param_type == 'fixed':
                free_params[mp_obj.free_indices] = np.median(np.stack(p[:, index]), axis=0)
            elif mp_obj.param_type == 'variable':
                free_params[mp_obj.free_indices] = p[:, index]
            elif mp_obj.param_type == 'dynamic':
                time_var  = self.time_variable
                func      = partial(self.fcns[mp_obj.function_name], t=time_var)
                gradfunc  = partial(self.get_gradient_fcn(mp_obj), t=time_var)

                def loss(x):
                    pred = func(x)
                    return np.mean((p[:, index] - pred)**2)

                def loss_grad(x):
                    jac_out = []
                    S = func(x)
                    for ds in gradfunc(x):
                        jac_out.append(np.sum((2 * S * ds)
                                              - (2 * p[:, index] * ds)))
                    return np.asarray(jac_out)

                bounds = self.Bounds[mp_obj.free_indices].tolist()
                vals = minimize(loss,
                                np.zeros(len(mp_obj.free_indices)),
                                jac=loss_grad,
                                method='TNC',
                                bounds=bounds).x

                free_params[mp_obj.free_indices] = vals

            else:
                raise ConfigFileError(
                    f"Unknown parameter mode ({mp_obj.param_type}) in configuration "
                    "- should be one of 'fixed', 'variable', {'dynamic'}")

        return free_params

    def get_gradient_fcn(self, mp_param_obj):
        """
        Get the gradient function for a given parameter
        Returns:
        function
        """
        if (mp_param_obj.param_type == 'fixed') or (mp_param_obj.param_type  == 'variable'):
            return lambda x, t: np.ones((1, self.ntimes))
        elif mp_param_obj.param_type == 'dynamic':
            grad_name = mp_param_obj.grad_function()
            if grad_name not in self.fcns:
                raise ConfigFileError(
                    f"Could not find gradient function {grad_name} for parameter {mp_param_obj.name}"
                    f" / function {mp_param_obj.function_name}")
            return self.fcns[grad_name]
        else:
            raise ConfigFileError(
                f"Unknown parameter mode ({mp_param_obj.param_type}) in configuration "
                "- should be one of 'fixed', 'variable', {'dynamic'}")

    def mapped_to_dict(self, params):
        """
        Conveniently store mapped params into a dict
        This makes it easier to access individuals parameters

        Parameters:
            params (array)
        Returns:
            dict
        """
        mapped_params = {c: [] for c in self.mapped_categories}
        for p, mp_obj in zip(params.T, self._mapped_params):
            mapped_params[mp_obj.category].append(p)
        mapped_params = {c: np.stack(mapped_params[c]) for c in self.mapped_categories}

        return mapped_params
