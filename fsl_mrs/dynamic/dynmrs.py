# dynmrs.py - Class responsible for dynMRS fitting
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford

# SHBASECOPYRIGHT
import time
import re
from shutil import copyfile

import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import pickle
import json

from fsl_mrs.utils import fitting
from fsl_mrs.utils import baseline as bline
from fsl_mrs import models
from . import variable_mapping as varmap
from . import dyn_results
from fsl_mrs.utils.results import FitRes
from fsl_mrs.utils.stats import mh, dist
from fsl_mrs.utils.misc import rescale_FID

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fsl_mrs.core.mrs import MRS

conc_index_re = re.compile(r'^(conc_.*?_)(\d+)$')


class dynMRSError(Exception):
    pass


class dynMRSArgumentError(dynMRSError):
    pass


class dynMRSLoadError(dynMRSError):
    pass


class dynMRS(object):
    """Dynamic MRS class"""

    def __init__(
            self,
            mrs_list,
            time_var,
            config_file,
            model='voigt',
            ppmlim=None,
            baseline: str = 'poly,2',
            baseline_order: int | None = None,
            metab_groups=None,
            rescale=True):
        """Create a dynMRS class object

        :param mrs_list: List of MRS objects, one per time_var
        :type mrs_list: List
        :param time_var: List containing the dynamic variable, or dict for multiple lists
        :type time_var: List or dict
        :param config_file: Path to the python model configuration file
        :type config_file: str
        :param model: 'voigt' or 'lorentzian', defaults to 'voigt'
        :type model: str, optional
        :param ppmlim: Chemical shift fitting limits, defaults to nucleus standard (via None) e.g. (.2, 4.2) for 1H.
        :type ppmlim: tuple, optional
        :param baseline: Baseline fitting option, defaults to poly, 2
        :type baseline: str, optional
        :param metab_groups: Metabolite group list, defaults to None
        :type metab_groups: list, optional
        :param rescale: Apply basis and FID rescaling, defaults to True
        :type rescale: bool, optional
        """

        if isinstance(time_var, dict):
            self.time_var = {}
            t_size = []
            for key in time_var:
                t_element = np.asarray(time_var[key])
                t_size.append(t_element.shape[0])
                self.time_var.update({key: t_element})
            t_size = np.asarray(t_size)
            if np.all(np.isclose(t_size, t_size[0])):
                self._t_steps = t_size[0]
            else:
                raise dynMRSArgumentError('All values in time_var dict must have the same first dimension shape.')
        else:
            self.time_var = np.asarray(time_var)
            self._t_steps = self.time_var.shape[0]

        # Check suitability of arguments
        if self._t_steps != len(mrs_list):
            raise dynMRSArgumentError(
                f'Number of time steps (currently {self._t_steps}) must match mrs_list length ({len(mrs_list)})')

        if ppmlim is None:
            ppmlim = mrs_list[0].default_ppm_range

        self.mrs_list = mrs_list
        if rescale:
            self._process_mrs_list()

        if metab_groups is None:
            metab_groups = [0] * len(self.metabolite_names)

        self._fit_args = {'model': model,
                          'metab_groups': metab_groups,
                          'baseline': baseline,
                          'baseline_order': baseline_order,
                          'ppmlim': ppmlim}

        self.data = self._prepare_data(ppmlim)
        self.forward = self._get_forward()
        self.gradient = self._get_gradient()

        metab_names, numBasis, numGroups = self.mrs_list[0].names, self.mrs_list[0].numBasis, max(metab_groups) + 1

        varNames, varSizes = models.FSLModel_vars(
            model,
            numBasis,
            n_groups=numGroups,
            n_baseline=self._baseline_object(self.mrs_list[0]).n_basis)
        self._vm = varmap.VariableMapping(
            param_names=varNames,
            param_sizes=varSizes,
            metabolite_names=metab_names,
            metabolite_groups=numGroups,
            time_variable=self.time_var,
            config_file=config_file)

        self.mapped_penalty = self._gen_penalty()

        # For save function
        self._config_file = Path(config_file)
        self._kwargs = {
            'model': model,
            'ppmlim': ppmlim,
            'baseline': baseline,
            'baseline_order': baseline_order,
            'metab_groups': metab_groups,
            'rescale': rescale}

    def save(self, save_dir, save_mrs_list=False):

        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save out the components needed to reconstruct this object
        # 1 - mrs_list as a pickle file
        if save_mrs_list:
            with open(save_dir / 'mrs_list.pkl', 'wb') as fp:
                pickle.dump(self.mrs_list, fp)

        # 2 - Save time_var and kw_args in json file
        with open(save_dir / 'args.json', 'w') as fp:
            json_dict = {
                'time_var': self.time_var.tolist(),
                'config_file': self._config_file.name,
                'kwargs': self._kwargs}
            json.dump(json_dict, fp)

        # 3 - Save a copy of the config file
        copyfile(self._config_file, save_dir / self._config_file.name)

    @classmethod
    def load(cls, load_dir, mrs_list=None):
        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)

        # 1) mrs_list
        if mrs_list:
            mrs_list_from_pkl = False
        elif mrs_list is None\
                and (load_dir / 'mrs_list.pkl').is_file():
            with open(load_dir / 'mrs_list.pkl', 'rb') as pickle_file:
                mrs_list = pickle.load(pickle_file)
            mrs_list_from_pkl = True
        else:
            raise dynMRSLoadError(f'mrs_list must be supplied as argument no mrs_list.pkl found in {str(load_dir)}.')

        # 2) Other arguments
        with open(load_dir / 'args.json', 'r') as json_file:
            args = json.load(json_file)

        time_var = args['time_var']
        config_file = args['config_file']
        if not (load_dir / config_file).is_file():
            raise dynMRSLoadError(f'config_file {str(load_dir / config_file)} not found!')

        kwargs = args['kwargs']

        if mrs_list_from_pkl and kwargs['rescale']:
            # Don't rescale twice!
            kwargs['rescale'] = False

        return cls(mrs_list, time_var, load_dir / config_file, **kwargs)

    def _process_mrs_list(self):
        """Apply single scaling to the mrs_list
        """
        scales = []
        for mrs in self.mrs_list:
            scales.append(rescale_FID(mrs.FID, scale=100.0)[1])

        scale = np.mean(scales)
        for mrs in self.mrs_list:
            mrs.fid_scaling = scale
            mrs.basis_scaling_target = 100.0

    @property
    def metabolite_names(self):
        return self.mrs_list[0].names

    @property
    def free_names(self):
        return self.vm.free_names
        # metabolites = self.metabolite_names

        # metab_fn = []
        # for fn in freenames:
        #     match = conc_index_re.match(fn)
        #     if match:
        #         mod_str = match[1] + metabolites[int(match[2])]
        #         metab_fn.append(mod_str)
        #     else:
        #         metab_fn.append(fn)

        # return metab_fn

    @property
    def mapped_names(self):
        return self.vm.mapped_names
        # full_mapped_names = []
        # for nn, ns in zip(self.vm.mapped_names, self.vm.mapped_sizes):
        #     if nn == 'conc':
        #         full_mapped_names.extend([f'{nn}_{nbasis}' for nbasis in self.metabolite_names])
        #     else:
        #         full_mapped_names.extend([f'{nn}_{idx:02.0f}' for idx in range(ns)])
        # return full_mapped_names

    @property
    def vm(self):
        return self._vm

    @property
    def time_index(self):
        """Returns a simple 1D index of time points.

        Usefull for plotting etc. when there is a ND time variable
        """
        return np.arange(len(self.time_var)).tolist()

    @property
    def fitargs(self) -> dict:
        """Returns the fit_arg dict

        :return: Fitting arguments: 'model', 'metab_groups', 'baseline', 'ppmlim'
        :rtype: dict
        """
        return self._fit_args

    def fit(self,
            method='quasi-newton',
            mh_jumps=600,
            init=None,
            x0=None,
            verbose=False,
            output_opt_sol=False,
            minimize_options={}):
        """Fit the dynamic model

        :param method: 'Quasi-newton', 'Newton' or 'MH', defaults to 'Quasi-newton'
        :type method: str, optional
        :param mh_jumps: Number of MH jumps, defaults to 600
        :type mh_jumps: int, optional
        :param init: Initialisation based on independent fitting approach, defaults to None
        :type init: dict, optional
        :param x0: Initialisation based on free parameters, defaults to None
        :type x0: np.array, optional
        :param verbose: Verbosity flag, defaults to False
        :type verbose: bool, optional
        :param output_opt_sol: Output the Scipy solution object (for debugging), defaults to False
        :type output_opt_sol: bool, optional
        :param minimize_options: Dict containing algorithm specific options.
            This dict is passed to the options kwarg of scipy minimize.
        :type minimize_options: dict, optional
        :return: Tuple containing dedicated results object, and optimisation output ([Quasi]-Newton only)
        :rtype: tuple
        """
        if verbose:
            print('Start fitting')
            start_time = time.time()

        if init is None \
                and x0 is None:
            init = self.initialise(verbose=verbose)

        if x0 is None:
            x0 = self.vm.mapped_to_free(init['x'])
        else:
            init = {'x': self.vm.free_to_mapped(x0)}

        # MCMC or Newton
        if method.lower() == 'newton':
            if 'maxfun' not in minimize_options:
                minimize_options.update(
                    {'maxfun': 100 * len(x0)})
            sol = minimize(
                method='TNC',
                fun=self.dyn_loss,
                x0=x0,
                jac=self.dyn_loss_grad,
                bounds=self.vm.Bounds,
                options=minimize_options)
            if sol.status != 0:
                print(
                    f'The TNC optimisation might have failed (status = {sol.status}), '
                    'please check solver output message.')
                print(sol)
            elif verbose:
                print(sol)
            x = sol.x
        elif method.lower() == 'quasi-newton':
            if 'maxcor' not in minimize_options:
                minimize_options.update(
                    {'maxcor': 100})
            sol = minimize(
                method='L-BFGS-B',
                fun=self.dyn_loss,
                x0=x0,
                jac=self.dyn_loss_grad,
                bounds=self.vm.Bounds,
                options=minimize_options)
            if sol.status != 0:
                print(
                    f'The L-BFGS optimisation might have failed (status = {sol.status}), '
                    'please check solver output message.')
                print(sol)
            elif verbose:
                print(sol)
            x = sol.x
        elif method.lower() == 'mh':
            self.prior_means = np.zeros_like(self.vm.nfree)
            self.prior_stds = np.ones_like(self.vm.nfree) * 1E3
            mcmc = mh.MH(self.dyn_loglik, self.dyn_logpr, burnin=100, njumps=mh_jumps, sampleevery=5)
            LB, UB = mcmc.bounds_from_list(self.vm.nfree, self.vm.Bounds.tolist())
            x = mcmc.fit(x0, LB=LB, UB=UB, verbose=verbose)
            sol = None
        else:
            raise ValueError(f'Unrecognised method {method}, must be one of "Quasi-newton", "Newton", or "MH".')
        end_fit_time = time.time()
        if verbose:
            print(f"...completed in {end_fit_time - start_time} seconds.")
        # Results
        if verbose:
            print('Collect results')
        # Create dedicated dynamic fit results
        if method.lower() in ('newton', 'quasi-newton'):
            results = dyn_results.dynRes_newton(sol.x, self, init)
        elif method.lower() == 'mh':
            results = dyn_results.dynRes_mcmc(x, self, init)
        else:
            raise ValueError(f'Unrecognised method {method}, must be one of "Quasi-newton", "Newton", or "MH".')

        if verbose:
            print(f"...completed in {time.time() - end_fit_time} seconds.")

        if output_opt_sol:
            return results, sol
        else:
            return results

    def fit_mean_spectrum(self):
        """Return the parameters from the fit of the mean spectra stored in mrs_list."""
        from fsl_mrs.utils.preproc.combine import combine_FIDs
        from copy import deepcopy

        mean_fid = combine_FIDs([mrs.FID for mrs in self.mrs_list], 'mean')
        mean_mrs = deepcopy(self.mrs_list[0])
        mean_mrs.FID = mean_fid
        return fitting.fit_FSLModel(mean_mrs, method='Newton', **self._fit_args).params

    def initialise(self, indiv_init='mean', verbose=False):
        """Initialise the dynamic fitting using seperate fits of each spectrum.

        :param indiv_init: Optional initialisation of individual fits.
            Can be a numpy array of mapped parameters, 'mean' (fits the mean spectrum), or None (independent).
            Defaults to 'mean'.
        :param verbose: Print information during fitting, defaults to False
        :type verbose: bool, optional
        :return: Dict containing free parameters and individual FitRes objects
        :rtype: dict
        """
        if verbose:
            start_time = time.time()

        if isinstance(indiv_init, str) and indiv_init == 'mean':
            indiv_init = self.fit_mean_spectrum()

        # Get init from fitting to individual time points
        init = np.zeros((self._t_steps, self.vm.nmapped))
        resList = []
        for t, mrs in enumerate(self.mrs_list):
            if verbose:
                print(f'Initialising {t + 1}/{len(self.mrs_list)}', end='\r')
            res = fitting.fit_FSLModel(mrs, method='Newton', x0=indiv_init, **self._fit_args)
            resList.append(res)
            init[t, :] = res.params
        # Conveniently store mapped params
        mapped_params = self.vm.mapped_to_dict(init)

        if verbose:
            print(f'Init done in {time.time() - start_time} seconds.')
        return {'x': init, 'mapped_params': mapped_params, 'resList': resList}

    # Utility methods
    def _baseline_object(self, mrs: "MRS") -> bline.Baseline:
        """Returns an instance of baseline class object
        corresponding to a single time point

        :param mrs: A single mrs object (time point)
        :type mrs: MRS
        :return: Single instance of baseline
        :rtype: bline.Baseline
        """
        return bline.Baseline(
            mrs,
            self._fit_args['ppmlim'],
            self._fit_args['baseline'],
            self._fit_args['baseline_order'])

    def _get_constants(self, mrs: "MRS") -> tuple:
        """collect constants for forward model per mrs object"""
        first, last = mrs.ppmlim_to_range(self._fit_args['ppmlim'])
        return (
            mrs.frequencyAxis,
            mrs.timeAxis,
            mrs.basis,
            self._baseline_object(mrs).regressor,
            self._fit_args['metab_groups'],
            max(self._fit_args['metab_groups']) + 1,
            first, last)

    def _prepare_data(self, ppmlim):
        """FID to Spec and slice for fitting"""
        first, last = self.mrs_list[0].ppmlim_to_range(ppmlim)
        data = [mrs.get_spec().copy()[first:last] for mrs in self.mrs_list]
        return data

    def _get_forward(self):
        """Get forward model"""
        forward = models.getModelForward(self._fit_args['model'])

        def raiser(const):
            first, last = const[-2:]
            return lambda x: forward(x, *const[:-2])[first:last]

        return [raiser(self._get_constants(mrs)) for mrs in self.mrs_list]

    def _get_gradient(self):
        """Get gradient"""
        gradient = models.getModelJac(self._fit_args['model'])

        def raiser(const):
            return lambda x: gradient(x, *const)

        return [raiser(self._get_constants(mrs)) for mrs in self.mrs_list]

    # Penalty functions
    def _gen_penalty(self):
        mapped_penalty = []

        def zero_func(*args):
            return 0
        _, _, _, x2p, _ = models.getModelFunctions(self._fit_args['model'])

        for time_index in range(self.vm.ntimes):
            bobj = self._baseline_object(self.mrs_list[time_index])

            mapped_penalty.append(
                bobj.prepare_penalised_fit_functions(
                    zero_func,
                    zero_func,
                    lambda x: x2p(
                        x,
                        self.mrs_list[time_index].numBasis,
                        max(self._fit_args['metab_groups']) + 1)[-1]
                ))
        return mapped_penalty

    # Loss functions
    def loss(self, x, i):
        """Calc loss function"""
        loss_real = .5 * np.mean(np.real(self.forward[i](x) - self.data[i]) ** 2)
        loss_imag = .5 * np.mean(np.imag(self.forward[i](x) - self.data[i]) ** 2)
        return loss_real + loss_imag

    def loss_grad(self, x, i):
        """Calc gradient of loss function"""
        g = self.gradient[i](x)
        e = self.forward[i](x) - self.data[i]
        grad_real = np.mean(np.real(g) * np.real(e[:, None]), axis=0)
        grad_imag = np.mean(np.imag(g) * np.imag(e[:, None]), axis=0)
        return grad_real + grad_imag

    def dyn_loss(self, x):
        """Add loss functions across data list"""
        ret = 0
        mapped = self.vm.free_to_mapped(x)
        for time_index in range(self.vm.ntimes):
            p = np.hstack(mapped[time_index, :])
            ret += self.loss(p, time_index)
            ret += self.mapped_penalty[time_index][0](p)
        ret /= self.vm.ntimes
        return ret

    def dyn_loss_grad(self, x):
        """Add gradients across data list"""
        mapped = self.vm.free_to_mapped(x)
        dfdx = 0
        for time_index in range(self.vm.ntimes):
            # dfdmapped
            p = mapped[time_index, :]
            dfdp = self.loss_grad(p, time_index)
            dfdp += self.mapped_penalty[time_index][1](p)
            # dmappeddfree
            dpdx = []
            for mp in self.vm.mapped_parameters:
                grad_fcn = self.vm.get_gradient_fcn(mp)
                fp_index = mp.free_indices

                gg = np.zeros(self.vm.nfree)
                gg[fp_index] = grad_fcn(x[fp_index], self.vm.time_variable)[:, time_index]

                dpdx.append(gg)

            dpdx = np.asarray(dpdx)
            dfdx += np.matmul(dfdp, dpdx)
        dfdx /= self.vm.ntimes
        return dfdx

    def dyn_loglik(self, x):
        """neg log likelihood for MCMC"""
        ll = 0.0
        mapped = self.vm.free_to_mapped(x)
        n_over_2 = len(self.data[0]) / 2
        for time_index in range(self.vm.ntimes):
            p = np.hstack(mapped[time_index, :])
            pred = self.forward[time_index](p)
            ll += np.log(np.linalg.norm(pred - self.data[time_index])) * n_over_2
        return ll

    def dyn_logpr(self, p):
        """neg log prior for MCMC"""
        return np.sum(dist.gauss_logpdf(p, loc=self.prior_means, scale=self.prior_stds))

    # Results functions
    def full_fwd(self, x):
        '''Return flattened vector of the full estimated model'''
        fwd = np.zeros((self.vm.ntimes, self.data[0].shape[0]), dtype=np.complex64)
        mapped = self.vm.free_to_mapped(x)
        for time_index in range(self.vm.ntimes):
            p = np.hstack(mapped[time_index, :])
            fwd[time_index, :] = self.forward[time_index](p)
        return fwd.flatten()

    def form_FitRes(self, x, method):
        """Create list of FitRes object"""
        if method.lower() == 'mh':
            mapped = []
            for xx in x:
                tmp = self.vm.free_to_mapped(xx)
                tmp_tmp = []
                for tt in tmp:
                    tmp_tmp.append(np.hstack(tt))
                mapped.append(np.asarray(tmp_tmp))
            mapped = np.asarray(mapped)
            mapped = np.moveaxis(mapped, 0, 1)
        else:
            tmp = self.vm.free_to_mapped(x)
            tmp_tmp = []
            for tt in tmp:
                tmp_tmp.append(np.hstack(tt))
            mapped = np.asarray(tmp_tmp)
        dynresList = []
        for t in range(self.vm.ntimes):
            mrs = self.mrs_list[t]
            results = FitRes(mrs,
                             mapped[t],
                             self._fit_args['model'],
                             method,
                             self._fit_args['metab_groups'],
                             self._baseline_object(mrs),
                             self._fit_args['ppmlim'])
            dynresList.append(results)
        return dynresList
