# dynmrs.py - Class responsible for dynMRS fitting
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford

# SHBASECOPYRIGHT

import numpy as np
from scipy.optimize import minimize
import time
import re

from fsl_mrs.utils import models, fitting
from . import variable_mapping as varmap
from . import dyn_results
from fsl_mrs.utils.results import FitRes
from fsl_mrs.utils.stats import mh, dist
from fsl_mrs.utils.misc import rescale_FID

conc_index_re = re.compile(r'^(conc_.*?_)(\d+)$')


class dynMRS(object):
    """Dynamic MRS class"""

    def __init__(
            self,
            mrs_list,
            time_var,
            config_file,
            model='voigt',
            ppmlim=(.2, 4.2),
            baseline_order=2,
            metab_groups=None,
            rescale=True):
        """Create a dynMRS class object

        :param mrs_list: List of MRS objects, one per time_var
        :type mrs_list: List
        :param time_var: List containing the dynamic variable
        :type time_var: List
        :param config_file: Path to the python model configuration file
        :type config_file: str
        :param model: 'voigt' or 'lorentzian', defaults to 'voigt'
        :type model: str, optional
        :param ppmlim: Chemical shift fitting limits, defaults to (.2, 4.2)
        :type ppmlim: tuple, optional
        :param baseline_order: Plynomial baseline fitting order, defaults to 2
        :type baseline_order: int, optional
        :param metab_groups: Metabolite group list, defaults to None
        :type metab_groups: list, optional
        :param rescale: Apply basis and FID rescaling, defaults to True
        :type rescale: bool, optional
        """

        self.time_var = time_var
        self.mrs_list = mrs_list
        if rescale:
            self._process_mrs_list()

        if metab_groups is None:
            metab_groups = [0] * len(self.metabolite_names)

        self._fit_args = {'model': model,
                          'metab_groups': metab_groups,
                          'baseline_order': baseline_order,
                          'ppmlim': ppmlim}

        self.data = self._prepare_data(ppmlim)
        self.forward = self._get_forward()
        self.gradient = self._get_gradient()

        numBasis, numGroups = self.mrs_list[0].numBasis, max(metab_groups) + 1
        varNames, varSizes = models.FSLModel_vars(model, numBasis, numGroups, baseline_order)
        self._vm = varmap.VariableMapping(
            param_names=varNames,
            param_sizes=varSizes,
            time_variable=self.time_var,
            config_file=config_file)

    def _process_mrs_list(self):
        """Apply single scaling to the mrs_list
        """
        scales = []
        for mrs in self.mrs_list:
            scales.append(rescale_FID(mrs.FID, scale=100.0)[1])

        for mrs in self.mrs_list:
            mrs.fid_scaling = np.mean(scales)
            mrs.basis_scaling_target = 100.0

    @property
    def metabolite_names(self):
        return self.mrs_list[0].names

    @property
    def free_names(self):
        freenames = self.vm.create_free_names()
        metabolites = self.metabolite_names

        metab_fn = []
        for fn in freenames:
            match = conc_index_re.match(fn)
            if match:
                mod_str = match[1] + metabolites[int(match[2])]
                metab_fn.append(mod_str)
            else:
                metab_fn.append(fn)

        return metab_fn

    @property
    def mapped_names(self):
        full_mapped_names = []
        for nn, ns in zip(self.vm.mapped_names, self.vm.mapped_sizes):
            if nn == 'conc':
                full_mapped_names.extend([f'{nn}_{nbasis}' for nbasis in self.metabolite_names])
            else:
                full_mapped_names.extend([f'{nn}_{idx:02.0f}' for idx in range(ns)])
        return full_mapped_names

    @property
    def vm(self):
        return self._vm

    def fit(self,
            method='Newton',
            mh_jumps=600,
            init=None,
            verbose=False):
        """Fit the dynamic model

        :param method: 'Newton' or 'MH', defaults to 'Newton'
        :type method: str, optional
        :param mh_jumps: Number of MH jumps, defaults to 600
        :type mh_jumps: int, optional
        :param init: Initilisation (x0), defaults to None
        :type init: dict, optional
        :param verbose: Verbosity flag, defaults to False
        :type verbose: bool, optional
        :return: Tuple containing dedicated results object, list of fitRes objects, optimisation output (Newton only)
        :rtype: tuple
        """
        if verbose:
            print('Start fitting')
            start_time = time.time()

        if init is None:
            init = self.initialise(verbose=verbose)
        x0 = self.vm.mapped_to_free(init['x'])

        # MCMC or Newton
        if method.lower() == 'newton':
            sol = minimize(fun=self.dyn_loss, x0=x0, jac=self.dyn_loss_grad, method='TNC', bounds=self.vm.Bounds)
            x = sol.x
        elif method.lower() == 'mh':
            self.prior_means = np.zeros_like(self.vm.nfree)
            self.prior_stds = np.ones_like(self.vm.nfree) * 1E3
            mcmc = mh.MH(self.dyn_loglik, self.dyn_logpr, burnin=100, njumps=mh_jumps, sampleevery=5)
            LB, UB = mcmc.bounds_from_list(self.vm.nfree, self.vm.Bounds)
            x = mcmc.fit(x0, LB=LB, UB=UB, verbose=verbose)
            sol = None
        else:
            raise (Exception(f'Unrecognised method {method}'))
        end_fit_time = time.time()
        if verbose:
            print(f"...completed in {end_fit_time-start_time} seconds.")
        # Results
        if verbose:
            print('Collect results')
        # 1. Create dedicated dynamic fit results
        if method.lower() == 'newton':
            results = dyn_results.dynRes_newton(sol.x, self, init)
        elif method.lower() == 'mh':
            results = dyn_results.dynRes_mcmc(x, self, init)
        else:
            raise (Exception(f'Unrecognised method {method}'))
        # 2. Create a list of fitRes objects for each timepoint for fsl_mrs like capabilities
        res_list = self._form_FitRes(
            x,
            self._fit_args['model'],
            method,
            self._fit_args['ppmlim'],
            self._fit_args['baseline_order'])

        if verbose:
            print(f"...completed in {time.time()-end_fit_time} seconds.")

        return {'result': results, 'resList': res_list, 'optimisation_sol': sol}

    def initialise(self, verbose=False):
        """Initialise the dynamic fitting using seperate fits of each spectrum.

        :param model: Spectral model 'lorentzian' or 'voigt', defaults to 'voigt'
        :type model: str, optional
        :param metab_groups: List of metabolite groupings, defaults to None
        :type metab_groups: List of ints, optional
        :param ppmlim: Ppm range over which to fit, defaults to (.2, 4.2)
        :type ppmlim: tuple, optional
        :param baseline_order: Polynomial baseline order, defaults to 2, -1 disables.
        :type baseline_order: int, optional
        :param verbose: Print information during fitting, defaults to False
        :type verbose: bool, optional
        :return: Dict containing free parameters and individual FitRes objects
        :rtype: dict
        """
        if verbose:
            start_time = time.time()

        varNames = models.FSLModel_vars(self._fit_args['model'])
        numMetabs = self.mrs_list[0].numBasis
        numGroups = max(self._fit_args['metab_groups']) + 1
        if self._fit_args['model'] == 'lorentzian':
            x2p = models.FSLModel_x2param
        else:
            x2p = models.FSLModel_x2param_Voigt
        # Get init from fitting to individual time points
        init = np.empty((len(self.time_var), len(varNames)), dtype=object)
        resList = []
        for t, mrs in enumerate(self.mrs_list):
            if verbose:
                print(f'Initialising {t + 1}/{len(self.mrs_list)}', end='\r')
            res = fitting.fit_FSLModel(mrs, method='Newton', **self._fit_args)
            resList.append(res)
            params = x2p(res.params, numMetabs, numGroups)
            for i, p in enumerate(params):
                init[t, i] = p
        # Conveniently store mapped params
        mapped_params = self.vm.mapped_to_dict(init)

        if verbose:
            print(f'Init done in {time.time()-start_time} seconds.')
        return {'x': init, 'mapped_params': mapped_params, 'resList': resList}

    # Utility methods
    def _get_constants(self, mrs, ppmlim, baseline_order, metab_groups):
        """collect constants for forward model"""
        first, last = mrs.ppmlim_to_range(ppmlim)  # data range
        freq, time, basis = mrs.frequencyAxis, mrs.timeAxis, mrs.basis
        base_poly = fitting.prepare_baseline_regressor(mrs, baseline_order, ppmlim)
        freq, time, basis = mrs.frequencyAxis, mrs.timeAxis, mrs.basis
        g = max(metab_groups) + 1
        return (freq, time, basis, base_poly, metab_groups, g, first, last)

    def _prepare_data(self, ppmlim):
        """FID to Spec and slice for fitting"""
        first, last = self.mrs_list[0].ppmlim_to_range(ppmlim)
        data = [mrs.get_spec().copy()[first:last] for mrs in self.mrs_list]
        return data

    def _get_forward(self):
        """Get forward model"""
        fwd_lambdas = []
        for mrs in self.mrs_list:
            forward = models.getModelForward(self._fit_args['model'])

            constants = self._get_constants(
                mrs,
                self._fit_args['ppmlim'],
                self._fit_args['baseline_order'],
                self._fit_args['metab_groups'])

            def raiser(const):
                first, last = const[-2:]
                return lambda x: forward(x, *const[:-2])[first:last]
            fwd_lambdas.append(raiser(constants))

        return fwd_lambdas

    def _get_gradient(self):
        """Get gradient"""
        gradient = models.getModelJac(self._fit_args['model'])
        fwd_grads = []
        for mrs in self.mrs_list:
            constants = self._get_constants(
                mrs,
                self._fit_args['ppmlim'],
                self._fit_args['baseline_order'],
                self._fit_args['metab_groups'])

            def raiser(const):
                return lambda x: gradient(x, *const)

            fwd_grads.append(raiser(constants))

        return fwd_grads

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
        for time_index in range(len(self.vm.time_variable)):
            p = np.hstack(mapped[time_index, :])
            ret += self.loss(p, time_index)
        return ret

    def dyn_loss_grad(self, x):
        """Add gradients across data list"""
        mapped = self.vm.free_to_mapped(x)
        LUT = self.vm.free_to_mapped(np.arange(self.vm.nfree), copy_only=True)
        dfdx = 0
        for time_index, _ in enumerate(self.vm.time_variable):
            # dfdmapped
            p = np.hstack(mapped[time_index, :])
            dfdp = self.loss_grad(p, time_index)
            # dmappeddfree
            dpdx = []
            for param_index, param in enumerate(self.vm.mapped_names):
                grad_fcn = self.vm.get_gradient_fcn(param)
                nparams = self.vm.mapped_sizes[param_index]
                xindex = LUT[time_index, param_index]

                for ip in range(nparams):
                    gg = np.zeros(self.vm.nfree)
                    if isinstance(xindex[ip], (int, np.int64)):
                        gg[xindex[ip]] = grad_fcn(x[xindex[ip]], self.vm.time_variable)
                    else:
                        gg[xindex[ip]] = grad_fcn(x[xindex[ip]], self.vm.time_variable)[:, time_index]
                    dpdx.append(gg)

            dpdx = np.asarray(dpdx)
            dfdx += np.matmul(dfdp, dpdx)
        return dfdx

    def dyn_loglik(self, x):
        """neg log likelihood for MCMC"""
        ll = 0.0
        mapped = self.vm.free_to_mapped(x)
        n_over_2 = len(self.data[0]) / 2
        for time_index in range(len(self.vm.time_variable)):
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

    def _form_FitRes(self, x, model, method, ppmlim, baseline_order):
        """Create list of FitRes object"""
        _, _, _, base_poly, metab_groups, _, _, _ = self._get_constants(
            self.mrs_list[0],
            self._fit_args['ppmlim'],
            self._fit_args['baseline_order'],
            self._fit_args['metab_groups'])
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
            results = FitRes(model,
                             method,
                             mrs.names,
                             metab_groups,
                             baseline_order,
                             base_poly,
                             ppmlim)
            results.loadResults(mrs, mapped[t])
            dynresList.append(results)
        return dynresList
