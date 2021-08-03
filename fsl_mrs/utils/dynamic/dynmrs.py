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

from fsl_mrs.utils import models, fitting
from . import variable_mapping as varmap
from fsl_mrs.utils.results import FitRes
from fsl_mrs.utils.stats import mh, dist
from fsl_mrs.utils.misc import calculate_lap_cov


class dynMRS(object):
    """Dynamic MRS class"""

    def __init__(self, mrs_list, time_var):
        """
        mrs_list : list of MRS objects
        time_var : array-like
        """
        self.mrs_list = mrs_list
        self.time_var = time_var
        self.data = None
        self.constants = None
        self.forward = None
        self.gradient = None
        self.vm = None

    def fit(self,
            config_file,
            method='Newton',
            mh_jumps=600,
            model='voigt',
            ppmlim=(.2, 4.2),
            baseline_order=2,
            metab_groups=None,
            init=None,
            verbose=False):
        """
        Fit dynamic MRS model

        Parameters
        ----------
        config_file    : string
        method         : string  ('Newton' or 'MH')
        model          : string  ('voigt' or 'lorentzian')
        mh_jumps       : int
        ppmlim         : tuple
        baseline_order : int
        metab_groups   : array-like
        init           : dynMRSres object
        verbose        : bool

        Returns
        -------
        dynMRSres object
        """
        if verbose:
            print('Start fitting')
            start_time = time.time()
        if metab_groups is None:
            metab_groups = [0] * len(self.mrs_list[0].names)
        self.data = self.prepare_data(ppmlim)
        self.constants = self.get_constants(model, ppmlim, baseline_order, metab_groups)
        self.forward = self.get_forward(model)
        self.gradient = self.get_gradient(model)

        numBasis, numGroups = self.mrs_list[0].numBasis, max(metab_groups) + 1
        varNames, varSizes = models.FSLModel_vars(model, numBasis, numGroups, baseline_order)
        self.vm = self.create_vm(model, config_file, varNames, varSizes)

        bounds = self.vm.Bounds
        if init is None:
            init = self.initialise(model, metab_groups, ppmlim, baseline_order, verbose)
        x0 = self.vm.mapped_to_free(init['x'])

        # MCMC or Newton
        if method.lower() == 'newton':
            sol = minimize(fun=self.dyn_loss, x0=x0, jac=self.dyn_loss_grad, method='TNC', bounds=bounds)
            # breakpoint()
            # calculate covariance
            data = np.asarray(self.data).flatten()
            x_cov = calculate_lap_cov(sol.x, self.full_fwd, data)
            x = sol.x
            x_out = x
            x_all = x
        elif method.lower() == 'mh':
            self.prior_means = np.zeros_like(self.vm.nfree)
            self.prior_stds = np.ones_like(self.vm.nfree) * 1E3
            mcmc = mh.MH(self.dyn_loglik, self.dyn_logpr, burnin=100, njumps=mh_jumps, sampleevery=5)
            LB, UB = mcmc.bounds_from_list(self.vm.nfree, self.vm.Bounds)
            x = mcmc.fit(x0, LB=LB, UB=UB, verbose=verbose)
            x_out = np.mean(x, axis=0)
            x_all = x
            x_cov = np.cov(x.T)
            sol = None
        else:
            raise (Exception(f'Unrecognised method {method}'))
        res_list = self.collect_results(x, model, method, ppmlim, baseline_order)

        if verbose:
            print(f"Fitting completed in {time.time()-start_time} seconds.")
        return {'x': x_out, 'cov': x_cov, 'samples': x_all, 'resList': res_list, 'OptimizeResult': sol}

    def get_constants(self, model, ppmlim, baseline_order, metab_groups):
        """collect constants for forward model"""
        mrs = self.mrs_list[0]
        first, last = mrs.ppmlim_to_range(ppmlim)  # data range
        freq, time, basis = mrs.frequencyAxis, mrs.timeAxis, mrs.basis
        base_poly = fitting.prepare_baseline_regressor(mrs, baseline_order, ppmlim)
        freq, time, basis = mrs.frequencyAxis, mrs.timeAxis, mrs.basis
        g = max(metab_groups) + 1
        return (freq, time, basis, base_poly, metab_groups, g, first, last)

    def initialise(self, model='voigt', metab_groups=None, ppmlim=(.2, 4.2), baseline_order=2, verbose=False):
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
        FitArgs = {'model': model,
                   'metab_groups': metab_groups,
                   'ppmlim': ppmlim,
                   'method': 'Newton',
                   'baseline_order': baseline_order}
        varNames = models.FSLModel_vars(model)
        numMetabs = self.mrs_list[0].numBasis
        numGroups = max(metab_groups) + 1
        if FitArgs['model'] == 'lorentzian':
            x2p = models.FSLModel_x2param
        else:
            x2p = models.FSLModel_x2param_Voigt
        # Get init from fitting to individual time points
        init = np.empty((len(self.time_var), len(varNames)), dtype=object)
        resList = []
        for t, mrs in enumerate(self.mrs_list):
            if verbose:
                print(f'Initialising {t + 1}/{len(self.mrs_list)}', end='\r')
            res = fitting.fit_FSLModel(mrs, **FitArgs)
            resList.append(res)
            params = x2p(res.params, numMetabs, numGroups)
            for i, p in enumerate(params):
                init[t, i] = p
        if verbose:
            print(f'Init done in {time.time()-start_time} seconds.')
        return {'x': init, 'resList': resList}

    def create_vm(self, model, config_file, varNames, varSizes):
        """Create Variable Mapping object"""
        vm = varmap.VariableMapping(param_names=varNames,
                                    param_sizes=varSizes,
                                    time_variable=self.time_var,
                                    config_file=config_file)
        return vm

    def prepare_data(self, ppmlim):
        """FID to Spec and slice for fitting"""
        first, last = self.mrs_list[0].ppmlim_to_range(ppmlim)
        data = [mrs.get_spec().copy()[first:last] for mrs in self.mrs_list]
        return data

    def get_forward(self, model):
        """Get forward model"""
        forward = models.getModelForward(model)
        first, last = self.constants[-2:]
        return lambda x: forward(x, *self.constants[:-2])[first:last]

    def get_gradient(self, model):
        """Get gradient"""
        gradient = models.getModelJac(model)
        return lambda x: gradient(x, *self.constants)

    def loss(self, x, i):
        """Calc loss function"""
        loss_real = .5 * np.mean(np.real(self.forward(x) - self.data[i]) ** 2)
        loss_imag = .5 * np.mean(np.imag(self.forward(x) - self.data[i]) ** 2)
        return loss_real + loss_imag

    def loss_grad(self, x, i):
        """Calc gradient of loss function"""
        g = self.gradient(x)
        e = self.forward(x) - self.data[i]
        grad_real = np.mean(np.real(g) * np.real(e[:, None]), axis=0)
        grad_imag = np.mean(np.imag(g) * np.imag(e[:, None]), axis=0)
        return grad_real + grad_imag

    def full_fwd(self, x):
        '''Return flattened vector of the full estimated model'''
        fwd = np.zeros((self.vm.ntimes, self.data[0].shape[0]), dtype=np.complex64)
        mapped = self.vm.free_to_mapped(x)
        for time_index in range(self.vm.ntimes):
            p = np.hstack(mapped[time_index, :])
            fwd[time_index, :] = self.forward(p)
        return fwd.flatten()

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
        for time_index, time_var in enumerate(self.vm.time_variable):
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
                    gg[xindex[ip]] = grad_fcn(x[xindex[ip]], time_var)
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
            pred = self.forward(p)
            ll += np.log(np.linalg.norm(pred - self.data[time_index])) * n_over_2
        return ll

    def dyn_logpr(self, p):
        """neg log prior for MCMC"""
        return np.sum(dist.gauss_logpdf(p, loc=self.prior_means, scale=self.prior_stds))

    # collect results
    def collect_results(self, x, model, method, ppmlim, baseline_order):
        """Create list of FitRes object"""
        _, _, _, base_poly, metab_groups, _, _, _ = self.constants
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
