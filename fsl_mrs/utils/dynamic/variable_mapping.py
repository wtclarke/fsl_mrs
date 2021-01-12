# variable_mapping.py - Class responsible for variable mapping
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from scipy.optimize import minimize
from functools import partial


class VariableMapping(object):
    def __init__(self,
                 param_names,
                 param_sizes,
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
        time_variale : array-like
        config_file  : string
        """

        self.time_variable  = np.asarray(time_variable)
        self.ntimes         = self.time_variable.shape[0]

        self.mapped_names   = param_names
        self.mapped_nparams = len(self.mapped_names)
        self.mapped_sizes   = param_sizes

        from runpy import run_path
        settings = run_path(config_file)

        self.Parameters     = settings['Parameters']
        for name in self.mapped_names:
            if name not in self.Parameters:
                self.Parameters[name] = 'fixed'
        self.fcns = {}
        for key in settings:
            if callable(settings[key]):
                self.fcns[key] = settings[key]
        if 'Bounds' in settings:
            self.defined_bounds = settings['Bounds']
            self.Bounds = self.create_constraints(settings['Bounds'])
        else:
            self.defined_bounds = None
            self.Bounds         = self.create_constraints(None)
        self.nfree          = self.calc_nfree()

    def __str__(self):
        OUT  = '-----------------------\n'
        OUT += 'Variable Mapping Object\n'
        OUT += '-----------------------\n'
        OUT += f'Number of Mapped param groups  = {len(self.mapped_names)}\n'
        OUT += f'Number of Mapped params        = {sum(self.mapped_sizes)}\n'
        OUT += f'Number of Free params          = {self.nfree}\n'
        OUT += f'Number of params if all indep  = {sum(self.mapped_sizes)*self.ntimes}\n'

        OUT += 'Dynamic functions\n'
        for param_name in self.mapped_names:
            beh = self.Parameters[param_name]
            OUT += f'{param_name} \t  {beh}\n'

        return OUT

    def __repr__(self) -> str:
        return str(self)



    def calc_nfree(self):
        """
        Calculate number of free parameters based on mapped behaviour

        Returns
        -------
        int
        """
        N = 0
        for index, param in enumerate(self.mapped_names):
            beh = self.Parameters[param]
            if (beh == 'fixed'):
                N += self.mapped_sizes[index]
            elif (beh == 'variable'):
                N += self.ntimes * self.mapped_sizes[index]
            else:
                if 'dynamic' in beh:
                    N += len(beh['params']) * self.mapped_sizes[index]
        return N

    def create_constraints(self, bounds):
        """
        Create list of constraints to be used in optimization

        Parameters:
        -----------
        bounds : dict   {param:bounds}

        Returns
        -------
        list
        """

        if bounds is None:
            return [(None, None)] * self.calc_nfree()

        if not isinstance(bounds, dict):
            raise(Exception('Input should either be a dict or None'))

        b = []  # list of bounds
        for index, name in enumerate(self.mapped_names):
            psize = self.mapped_sizes[index]

            if (self.Parameters[name] == 'fixed'):
                # check if there are bound on this param
                if name in bounds:
                    for s in range(psize):
                        b.append(bounds[name])
                else:
                    for s in range(psize):
                        b.append((None, None))

            elif (self.Parameters[name] == 'variable'):
                for t in range(self.ntimes):
                    for s in range(psize):
                        if name in bounds:
                            b.append(bounds[name])
                        else:
                            b.append((None, None))
            else:
                if 'dynamic' in self.Parameters[name]:
                    pnames = self.Parameters[name]['params']
                    for s in range(psize):
                        for p in pnames:
                            if p in bounds:
                                b.append(bounds[p])
                            else:
                                b.append((None, None))

        return b

    def mapped_from_list(self, p):
        """
        Converts list of params into Mapped by repeating over time

        Parameters
        ----------
        p : list

        Returns
        -------
        2D array
        """
        if isinstance(p, list):
            p = np.asarray(p)
        if (p.ndim == 1):
            p = np.repeat(p[None, :], self.ntimes, 0)
        return p

    def create_free_names(self):
        """
        create list of names for free params

        Returns
        -------
        list of strings
        """
        names = []
        for index, param in enumerate(self.mapped_names):
            beh = self.Parameters[param]
            if (beh == 'fixed'):
                name = [f'{param}_{x}' for x in range(self.mapped_sizes[index])]
                names.extend(name)
            elif (beh == 'variable'):
                name = [f'{param}_{x}_t{t}' for t in range(self.ntimes) for x in range(self.mapped_sizes[index])]
                names.extend(name)
            else:
                if 'dynamic' in beh:
                    dyn_name = self.Parameters[param]['params']
                    name = [f'{param}_{y}_{x}' for x in range(self.mapped_sizes[index]) for y in dyn_name]
                    names.extend(name)

        return names

    def free_to_mapped(self, p, copy_only=False):
        """
        Convert free into mapped params over time
        fixed params get copied over time domain
        variable params are indep over time
        dynamic params are mapped using dyn model

        Parameters
        ----------
        p : 1D array
        copy_only : bool (copy params - don't use dynamic models)
        Returns
        -------
        2D array (time X params)

        """
        # Check input
        if (p.size != self.nfree):
            raise(Exception('Input free params does not have expected number of entries.'
                            f' Found {p.size}, expected {self.nfree}'))

        # Mapped params is time X nparams (each param is an array of params)
        mapped_params = np.empty((self.ntimes, self.mapped_nparams), dtype=object)

        counter = 0
        for index, name in enumerate(self.mapped_names):
            nmapped   = self.mapped_sizes[index]

            if (self.Parameters[name] == 'fixed'):  # repeat param over time
                for t in range(self.ntimes):
                    mapped_params[t, index] = p[counter:counter + nmapped]
                counter += nmapped

            elif (self.Parameters[name] == 'variable'):  # copy one param for each time point
                for t in range(self.ntimes):
                    mapped_params[t, index] = p[counter :counter + nmapped]
                    counter += nmapped

            else:
                if 'dynamic' in self.Parameters[name]:
                    # Generate time courses
                    func_name = self.Parameters[name]['dynamic']
                    nfree     = len(self.Parameters[name]['params'])

                    if not copy_only:
                        mapped = np.zeros((self.ntimes, nmapped))
                        for i in range(nmapped):
                            params      = p[counter:counter + nfree]
                            mapped[:, i] = self.fcns[func_name](params, self.time_variable)
                            counter += nfree
                        for t in range(self.ntimes):
                            mapped_params[t, index] = mapped[t, :]
                    else:
                        mapped = np.empty((self.ntimes, nmapped),dtype=object)
                        for i in range(nmapped):
                            params      = p[counter:counter + nfree]
                            for t in range(self.ntimes):
                                mapped[t, i] = params
                            counter += nfree
                        for t in range(self.ntimes):
                            mapped_params[t, index] = mapped[t, :]
                else:
                    raise(Exception("Unknown Parameter type - should be one of 'fixed', 'variable', {'dynamic'}"))

        return mapped_params

    def print_free(self, x):
        """
        Print free params and their names
        """
        print(dict(zip(self.create_free_names(), x)))

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
        p = self.mapped_from_list(p)
        if (p.shape != (self.ntimes, self.mapped_nparams)):
            raise(Exception(f'Input mapped params does not have expected number of entries.'
                            f' Found {p.shape}, expected {(self.ntimes,self.mapped_nparams)}'))

        free_params = np.empty(self.nfree)
        counter = 0
        for index, name in enumerate(self.mapped_names):
            psize = self.mapped_sizes[index]
            if (self.Parameters[name] == 'fixed'):
                free_params[counter:counter + psize] = p[0, index]
                counter += psize
            elif (self.Parameters[name] == 'variable'):
                for t in range(self.ntimes):
                    free_params[counter:counter + psize] = p[t, index]
                    counter += psize
            else:
                if 'dynamic' in self.Parameters[name]:
                    func_name = self.Parameters[name]['dynamic']
                    time_var  = self.time_variable
                    func      = partial(self.fcns[func_name], t=time_var)
                    nfree     = len(self.Parameters[name]['params'])

                    pp = np.stack(p[:, index][:], axis=0)
                    for ppp in range(pp.shape[1]):
                        def loss(x):
                            pred = func(x)
                            return np.mean((pp[:, ppp] - pred)**2)
                        bounds = self.Bounds[counter:counter + nfree]
                        vals = minimize(loss,
                                        np.zeros(len(self.Parameters[name]['params'])),
                                        method='TNC', bounds=bounds).x
                        free_params[counter:counter + nfree] = vals
                        counter += nfree

                else:
                    raise(Exception("Unknown Parameter type - should be one of 'fixed', 'variable', {'dynamic'}"))

        return free_params

    def get_gradient_fcn(self,param_name):
        """
        Get the gradient function for a given parameter
        Returns:
        function
        """
        if (self.Parameters[param_name] == 'fixed') or (self.Parameters[param_name] == 'variable'):
            return lambda x, t: 1
        else:
            if 'dynamic' in self.Parameters[param_name]:
                func_name = self.Parameters[param_name]['dynamic']
                grad_name = func_name + '_grad'
                if grad_name not in self.fcns:
                    raise (Exception(f"Could not find gradient for parameter {param_name}"))
                return self.fcns[grad_name]
            else:
                raise (Exception("Unknown Parameter type - should be one of 'fixed', 'variable', {'dynamic'}"))
