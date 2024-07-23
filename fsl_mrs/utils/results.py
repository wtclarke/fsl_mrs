# results.py - Collate results from MRS fits
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

from copy import deepcopy
import json

import pandas as pd
import numpy as np

import fsl_mrs.models as models
import fsl_mrs.utils.quantify as quant
import fsl_mrs.utils.qc as qc
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID, calculate_lap_cov


class FitRes(object):
    """
       Collects fitting results
    """

    def __init__(
            self,
            mrs,
            results,
            model,
            method,
            metab_groups,
            baseline_obj,
            ppmlim,
            runqc=True):

        # Store options from
        self.model = model
        self.method = method
        self.ppmlim = ppmlim
        self._baseline_obj = baseline_obj

        self.fill_names(mrs.names, nbaseline=baseline_obj.n_basis, metab_groups=metab_groups)

        # Init properties
        self.concScalings = {'internal': None, 'internalRef': None, 'molarity': None, 'molality': None, 'info': None}
        self._combined_crlb = np.asarray([])

        # Populate data frame
        if results.ndim == 1:
            self.fitResults = pd.DataFrame(data=results[np.newaxis, :], columns=self.params_names)
        else:
            self.fitResults = pd.DataFrame(data=results, columns=self.params_names)

        # Store prediction, baseline, residual, so that the mrs object doesn't have to be stored
        self._pred = self.predictedFID(mrs, mode='Full')
        self._baseline = self.predictedFID(mrs, mode='Baseline')
        self._residuals = mrs.FID - self.pred
        first, last = mrs.ppmlim_to_range(self.ppmlim)
        self._mse = np.mean(np.abs(FIDToSpec(self._residuals)[first:last])**2)

        # Calculate single point crlb and cov
        _, _, forward, _, _ = models.getModelFunctions(self.model)
        jac = models.getModelJac(self.model)
        data = mrs.get_spec(ppmlim=self.ppmlim)

        def forward_lim(p):
            return forward(
                p,
                mrs.frequencyAxis,
                mrs.timeAxis,
                mrs.basis,
                self.base_poly,
                self.metab_groups,
                self.g)[first:last]

        def jac_lim(p):
            return jac(
                p,
                mrs.frequencyAxis,
                mrs.timeAxis,
                mrs.basis,
                self.base_poly,
                self.metab_groups,
                self.g,
                first, last)

        # Calculate uncertainties using covariance derived from Fisher information
        # Tested in fsl_mrs/tests/mc_validation/uncertainty_validation.ipynb
        # Empirical factor of 2 found, likely to arise from complex data/residuals
        self._cov = calculate_lap_cov(self.params, forward_lim, data, jac_lim(self.params).T)
        self._cov /= 2  # Apply factor 2 correction

        # Calculate mcmc metrics
        if self.method == 'MH':
            self.mcmc_cov = self.fitResults.cov().to_numpy()
            self.mcmc_cor = self.fitResults.corr().to_numpy()
            self.mcmc_var = self.fitResults.var().to_numpy()
            self.mcmc_samples = self.fitResults.to_numpy()

        self.hzperppm = mrs.centralFrequency / 1E6
        self.bandwidth = mrs.bandwidth

        # Calculate QC metrics
        if runqc:
            self.FWHM, self.SNR = qc.calcQC(mrs, self, ppmlim=self.ppmlim)

        # Run relative concentration scaling to tCr in 'default' 1H MRS case.
        # Create combined metab at same time to avoid later errors.
        if (('Cr' in self.metabs) and ('PCr' in self.metabs)):
            self.combine([['Cr', 'PCr']])
            self.calculateConcScaling(mrs)

    @property
    def numMetabs(self):
        """Return number of metabolites in model"""
        return len(self.metabs)

    @property
    def params(self):
        """Return mean fit parameters as numpy array"""
        return self.fitResults.loc[:, self.params_names].mean().to_numpy()

    @property
    def params_inc_combine(self):
        """Return mean fit parameters (including combined metabolites) as numpy array"""
        return self.fitResults.mean().to_numpy()

    @property
    def pred(self):
        """Return predicted FID"""
        return self._pred

    @property
    def pred_spec(self):
        """Returns predicted spectrum"""
        return FIDToSpec(self._pred)

    @property
    def baseline(self):
        """Returns predicted baseline"""
        return self._baseline

    @property
    def residuals(self):
        """Returns fit residual"""
        return self._residuals

    @property
    def mse(self):
        """Returns mse of fit"""
        return self._mse

    @property
    def cov(self):
        """Returns covariance matrix"""
        return self._cov

    @property
    def corr(self):
        """Returns correlation matrix"""
        std = np.sqrt(np.diagonal(self.cov))
        return self.cov / (std[:, np.newaxis] * std[np.newaxis, :])

    @property
    def crlb(self):
        """Returns crlb (variance) vector"""
        original_crlb = np.diagonal(self.cov)
        return np.concatenate([original_crlb, self._combined_crlb])

    @property
    def perc_SD(self):
        """Returns the percentage standard deviation"""
        with np.errstate(divide='ignore', invalid='ignore'):
            perc_SD = np.sqrt(self.crlb) / self.params * 100
            perc_SD[perc_SD > 999] = 999   # Like LCModel :)
            perc_SD[np.isnan(perc_SD)] = 999
        return perc_SD

    @property
    def baseline_mode(self):
        return self._baseline_obj.mode

    @property
    def base_poly(self):
        return self._baseline_obj.regressor

    @property
    def n_baseline_bases(self):
        return self._baseline_obj.n_basis

    class QuantificationError(Exception):
        pass

    def calculateConcScaling(self,
                             mrs,
                             quant_info=None,
                             internal_reference=['Cr', 'PCr'],
                             verbose=False):
        """Run calculation of internal and (if possible) water concentration scaling.

        :param mrs: MRS object
        :type mrs: fsl_mrs.core.mrs.MRS
        :param quant_info: Quantification information for water scaling, defaults to None
        :type quant_info: fsl_mrs.utils.quantify.QuantificationInfo, optional
        :param internal_reference: Internal referencing metabolite, defaults to ['Cr', 'PCr'] i.e. tCr
        :type internal_reference: list, optional
        :param verbose: Enable for verbose output, defaults to False
        :type verbose: bool, optional
        """

        self.intrefstr = '+'.join(internal_reference)
        self.referenceMetab = internal_reference

        internalRefScaling = quant.quantifyInternal(internal_reference, self.getConc(), self.metabs)

        if mrs.H2O is not None and quant_info is not None:
            molalityScaling, molarityScaling, ref_info = quant.quantifyWater(mrs,
                                                                             self,
                                                                             quant_info,
                                                                             verbose=verbose)
            if ref_info['metab_ref'].integral == 0.0:
                raise self.QuantificationError(
                    f'Metabolite reference {quant_info.ref_metab} has not been fit (conc=0). '
                    'Please choose another or refine fit first.')
            elif ref_info['water_ref'].integral == 0.0:
                raise self.QuantificationError(
                    'Water reference has zero integral. Please check water reference data.')

            self.concScalings = {
                'internal': internalRefScaling,
                'internalRef': self.intrefstr,
                'molarity': molarityScaling,
                'molality': molalityScaling,
                'quant_info': quant_info,
                'ref_info': ref_info}
        else:
            self.concScalings = {
                'internal': internalRefScaling,
                'internalRef': self.intrefstr,
                'molarity': None,
                'molality': None,
                'quant_info': None,
                'ref_info': None}

    def combine(self, combineList):
        """Combine two or more basis into single result"""
        # Create extra entries in the fitResults DF , add to param_names and recalculate
        for toComb in combineList:
            newstr = '+'.join(toComb)
            if newstr in self.metabs:
                continue
            ds = pd.Series(np.zeros(self.fitResults.shape[0]), index=self.fitResults.index)
            jac = np.zeros(self.cov.shape[0])
            for metab in toComb:
                if metab not in self.metabs:
                    raise ValueError(f'Metabolites to combine must be in res.metabs. {metab} not found.')
                ds = ds.add(self.fitResults[metab])
                index = self.metabs.index(metab)
                jac[index] = 1.0

            self.fitResults[newstr] = pd.Series(ds, index=self.fitResults.index)
            self.params_names_inc_comb.append(newstr)
            self.metabs.append(newstr)
            new_crlb = np.atleast_1d(jac @ self.cov @ jac)
            self._combined_crlb = np.concatenate([self._combined_crlb, new_crlb])

        if self.method == 'MH':
            self.mcmc_cov = self.fitResults.cov().to_numpy()
            self.mcmc_cor = self.fitResults.corr().to_numpy()
            self.mcmc_var = self.fitResults.var().to_numpy()

    def predictedSpec(self, mrs, mode='Full', noBaseline=False, ppmlim=None, shift=True):
        if ppmlim is None:
            ppmlim = self.ppmlim

        if mode.lower() == 'full':
            out = models.getFittedModel(
                self.model,
                self.params,
                self.base_poly,
                self.metab_groups,
                mrs,
                noBaseline=noBaseline)
        elif mode.lower() == 'baseline':
            out = models.getFittedModel(self.model, self.params, self.base_poly,
                                        self.metab_groups, mrs, baselineOnly=True)
        elif mode in self.metabs:
            out = models.getFittedModel(
                self.model,
                self.params,
                self.base_poly,
                self.metab_groups,
                mrs,
                basisSelect=mode,
                baselineOnly=False,
                noBaseline=noBaseline)
        else:
            raise ValueError('Unknown mode, must be one of: Full, baseline or a metabolite name.')

        first, last = mrs.ppmlim_to_range(ppmlim=ppmlim, shift=shift)

        return out[first:last]

    def predictedFID(self, mrs, mode='Full', noBaseline=False, no_phase=False):
        """Return the predicted FID generated from the fitted model.

        Mode may be set to return full fit, baseline or ore or more metabolite

        :param mrs: MRS object containing data and basis
        :type mrs: fsl_mrs.core.mrs.MRS
        :param mode: 'Full', 'Baseline', metabolite name or list of names, defaults to 'Full'
        :type mode: str, optional
        :param noBaseline: Remove baseline, defaults to False
        :type noBaseline: bool, optional
        :param no_phase: Remove fitted phase, defaults to False
        :type no_phase: bool, optional
        :return: predicted FID
        :rtype: numpy.array
        """
        if isinstance(mode, (list, tuple))\
                and all([x in self.metabs for x in mode]):
            out = np.sum(
                [models.getFittedModel(
                    self.model,
                    self.params,
                    self.base_poly,
                    self.metab_groups,
                    mrs,
                    basisSelect=x,
                    baselineOnly=False,
                    noBaseline=noBaseline,
                    no_phase=no_phase) for x in mode],
                axis=0)
        elif mode in self.metabs:
            out = models.getFittedModel(
                self.model,
                self.params,
                self.base_poly,
                self.metab_groups,
                mrs,
                basisSelect=mode,
                baselineOnly=False,
                noBaseline=noBaseline,
                no_phase=no_phase)
        elif mode.lower() == 'full':
            out = models.getFittedModel(
                self.model,
                self.params,
                self.base_poly,
                self.metab_groups,
                mrs,
                noBaseline=noBaseline,
                no_phase=no_phase)
        elif mode.lower() == 'baseline':
            out = models.getFittedModel(
                self.model,
                self.params,
                self.base_poly,
                self.metab_groups,
                mrs,
                baselineOnly=True,
                no_phase=no_phase)
        else:
            raise ValueError(
                'Unknown mode, must be one of: Full, baseline, a metabolite name or list of names.')
        return SpecToFID(out)

    def __str__(self):
        out = "----- Fitting Results ----\n"
        out += " names          = {}\n".format(self.params_names)
        out += " params         = {}\n".format(self.params)
        out += " CRLB           = {}\n".format(self.crlb)
        out += " MSE            = {}\n".format(self.mse)
        # out += " cov            = {}\n".format(self.cov)
        out += " phi0 (deg)     = {}\n".format(self.getPhaseParams()[0])
        out += " phi1 (deg/ppm) = {}\n".format(self.getPhaseParams(phi1='deg_per_ppm')[1])
        out += " gamma (Hz)     = {}\n".format(self.getLineShapeParams())
        out += " eps (ppm)      = {}\n".format(self.getShiftParams())
        out += " b_norm         = {}\n".format(self.getBaselineParams())

        return out

    def fill_names(self, names, nbaseline=0, metab_groups=None):
        """
        mrs            : MRS Object
        nbaseline : int
        metab_groups   : list (by default assumes single metab group)
        """
        self.metabs = deepcopy(names)
        self.original_metabs = deepcopy(names)

        self.params_names = []
        self.params_names.extend(names)

        if metab_groups is None:
            g = 1
        else:
            g = max(metab_groups) + 1

        self.g = g
        self.metab_groups = metab_groups

        for i in range(g):
            self.params_names.append(f'gamma_{i}')
        if self.model.lower() in ['voigt', 'free_shift']:
            for i in range(g):
                self.params_names.append(f'sigma_{i}')

        if self.model.lower() == 'free_shift':
            for i in range(len(names)):
                self.params_names.append(f"eps_{i}")
        else:
            for i in range(g):
                self.params_names.append(f"eps_{i}")

        self.params_names.extend(['Phi0', 'Phi1'])

        for i in range(0, nbaseline):
            self.params_names.append(f"B_real_{i}")
            self.params_names.append(f"B_imag_{i}")

        self.params_names_inc_comb = deepcopy(self.params_names)

    def to_file(self, filename, what='concentrations'):
        """
        Save results to csv file

        Parameters:
        -----------
        filename : str
        what     : one of 'summary', 'concentrations, 'qc', 'parameters', 'concentrations-mh','parameters-mh'
        """

        if what == 'summary':
            df = pd.DataFrame()
            df['Metab'] = self.metabs

            if self.concScalings['internal'] is not None:
                concstr = f'/{self.concScalings["internalRef"]}'
                df[concstr] = self.getConc(scaling='internal')
                df[concstr + ' CRLB'] = self.getUncertainties(type='internal')
            else:
                df['Raw conc'] = self.getConc()
                df['Raw CRLB'] = self.getUncertainties(type='raw')
            if self.concScalings['molality'] is not None:
                df['mMol/kg'] = self.getConc(scaling='molality')
                df['mMol/kg CRLB'] = self.getUncertainties(type='molality')
            if self.concScalings['molarity'] is not None:
                df['mM'] = self.getConc(scaling='molarity')
                df['mM CRLB'] = self.getUncertainties(type='molarity')
            df['%CRLB'] = self.getUncertainties()

            SNR = self.getQCParams()[0]
            SNR.index = SNR.index.str.replace('SNR_', '')
            SNR.index.name = 'Metab'
            SNR = SNR.reset_index(name='SNR')
            df = df.merge(SNR, how='outer')

            FWHM = self.getQCParams()[1]
            FWHM.index = FWHM.index.str.replace('fwhm_', '')
            FWHM.index.name = 'Metab'
            FWHM = FWHM.reset_index(name='FWHM')
            df = df.merge(FWHM, how='outer')

        elif what == 'concentrations':
            scaling_type = ['raw']
            if self.concScalings['internal'] is not None:
                scaling_type.append('internal')
            if self.concScalings['molality'] is not None:
                scaling_type.append('molality')
            if self.concScalings['molarity'] is not None:
                scaling_type.append('molarity')

            std = [self.getUncertainties(type=st) for st in scaling_type]
            mean = [self.getConc(scaling=st, function='mean').T for st in scaling_type]

            d = {}
            d['mean'] = {key: value for key, value in zip(scaling_type, mean)}
            d['std'] = {key: value for key, value in zip(scaling_type, std)}

            dict_of_df = {k: pd.DataFrame(v) for k, v in d.items()}
            df = pd.concat(dict_of_df, axis=1)
            df.insert(0, 'Metabs', self.metabs)

        elif what == 'concentrations-mh':
            scaling_type = ['raw']
            if self.concScalings['internal'] is not None:
                scaling_type.append('internal')
            if self.concScalings['molality'] is not None:
                scaling_type.append('molality')
            if self.concScalings['molarity'] is not None:
                scaling_type.append('molarity')

            all_df = []
            for st in scaling_type:
                df = self.getConc(scaling=st, function=None).T
                df.insert(0, 'scaling', st)
                df.index.name = 'metabolite'
                all_df.append(df)

            df = pd.concat(all_df)
            df.reset_index(inplace=True)

        elif what == 'qc':

            SNR = self.getQCParams()[0]
            SNR.index = SNR.index.str.replace('SNR_', '')
            SNR.index.name = 'Metab'
            SNR = SNR.reset_index(name='SNR')

            FWHM = self.getQCParams()[1]
            FWHM.index = FWHM.index.str.replace('fwhm_', '')
            FWHM.index.name = 'Metab'
            FWHM = FWHM.reset_index(name='FWHM')

            df = SNR.merge(FWHM, how='outer')

        elif what == 'parameters':
            if self.method == 'MH':
                mean = self.fitResults.T.mean(axis=1)
                std = self.fitResults.T.std(axis=1)
                mean = mean[~mean.index.str.contains('\\+')].to_numpy()
                std = std[~std.index.str.contains('\\+')].to_numpy()
            else:
                mean = self.params
                std = np.sqrt(np.diagonal(self.cov))
            df = pd.DataFrame()
            df['parameter'] = self.params_names
            df['mean'] = mean
            df['std'] = std

        elif what == 'parameters-mh':
            df = self.fitResults.T
            df.index.name = 'parameter'
            df.reset_index(inplace=True)

        df.to_csv(filename, index=False, header=True)

    def metabs_in_groups(self):
        """Return list of metabolites in each metabolite group

        :return: Returns nested list with metabolite strings in each metabolite_group
        :rtype: list
        """
        group_list = []
        for g in range(self.g):
            metabs = []
            for i, m in enumerate(self.original_metabs):
                if self.metab_groups[i] == g:
                    metabs.append(m)
            group_list.append(metabs)
        return group_list

    def metabs_in_group(self, group):
        """Get list of metabolites in specific group

        :param group: Metabolite group index
        :type group: int
        :return: List of emtabolites in specified group
        :rtype: List
        """
        if group >= self.g:
            raise ValueError(f'Group must be in the range 0 to {self.g - 1}.')

        return self.metabs_in_groups()[group]

    def metab_in_group_json(self, save_path=None):
        """Generate and (optionally) save a json representation
        of the metabolites in each group

        :param save_path: Optional path to save json to, defaults to None
        :type save_path: str, optional
        :return: Returns json formmatted string of metabolite groups
        :rtype: str or Pathlib.Path
        """
        dict_repr = {idx: ml for idx, ml in enumerate(self.metabs_in_groups())}
        if save_path:
            with open(save_path, 'w') as jf:
                json.dump(dict_repr, jf)
            return json.dumps(dict_repr)
        else:
            return json.dumps(dict_repr)

    def fit_parameters_json(self, save_path):
        """Save a list of parameter names

        :param save_path: Path to save json file containing parameter names
        :type save_path: str or Pathlib.Path
        """
        param_dict = {'parameters': self.params_names,
                      'parameters_inc_comb': self.params_names_inc_comb,
                      'metabolites': self.original_metabs,
                      'metabolites_inc_comb': self.metabs}

        with open(save_path, 'w') as jf:
            json.dump(param_dict, jf)

    # Functions to return physically meaningful units from the fitting results
    def getConc(self, scaling='raw', metab=None, function='mean'):
        if function is None:
            def dfFunc(m):
                return self.fitResults[m]
        elif metab is None:
            def dfFunc(m):
                return self.fitResults[m].apply(function).to_numpy()
        else:
            def dfFunc(m):
                return self.fitResults[m].apply(function)

        # Extract concentrations from parameters.
        if metab is not None:
            if isinstance(metab, list):
                for mm in metab:
                    if mm not in self.metabs:
                        raise ValueError(f'{mm} is not a recognised metabolite.')
            else:
                if metab not in self.metabs:
                    raise ValueError(f'{metab} is not a recognised metabolite.')
            rawConc = dfFunc(metab)
        else:
            rawConc = dfFunc(self.metabs)

        if scaling == 'raw':
            return rawConc
        elif scaling == 'internal':
            if self.concScalings['internal'] is None:
                raise ValueError('Internal concetration scaling not calculated, run calculateConcScaling method.')
            return rawConc * self.concScalings['internal']

        elif scaling == 'molality':
            if self.concScalings['molality'] is None:
                raise ValueError('Molality concetration scaling not calculated, run calculateConcScaling method.')
            return rawConc * self.concScalings['molality']

        elif scaling == 'molarity':
            if self.concScalings['molarity'] is None:
                raise ValueError('Molarity concetration scaling not calculated, run calculateConcScaling method.')
            return rawConc * self.concScalings['molarity']
        else:
            raise ValueError(f'Unrecognised scaling value {scaling}.')

    def getPhaseParams(self, phi0='degrees', phi1='seconds', function='mean'):
        """Return the two phase parameters in specified units"""
        if function is None:
            p0 = self.fitResults['Phi0'].to_numpy()
            p1 = self.fitResults['Phi1'].to_numpy()
        else:
            p0 = self.fitResults['Phi0'].apply(function)
            p1 = self.fitResults['Phi1'].apply(function)

        if phi0.lower() == 'degrees':
            p0 *= 180.0 / np.pi
        elif (phi0.lower() == 'radians') or (phi0.lower() == 'raw'):
            pass
        else:
            raise ValueError('phi0 must be degrees or radians')

        if phi1.lower() == 'seconds':
            p1 *= -1.0 / (2 * np.pi)
        elif phi1.lower() == 'deg_per_ppm':
            p1 *= -180.0 / np.pi * self.hzperppm
        elif phi1.lower() == 'deg_per_hz':
            p1 *= -180.0 / np.pi * 1.0
        elif phi1.lower() == 'raw':
            pass
        else:
            raise ValueError('phi1 must be seconds, deg_per_ppm, deg_per_hz or raw.')

        return p0, p1

    def getShiftParams(self, units='ppm', function='mean'):
        """ Return shift parameters (eps) in specified units - default = ppm."""
        if self.model == 'free_shift':
            iter_range = range(len(self.original_metabs))
        else:
            iter_range = range(self.g)
        if function is None:
            shiftParams = np.zeros([self.fitResults.shape[0], len(iter_range)])
            for g in iter_range:
                shiftParams[:, g] = self.fitResults[f'eps_{g}'].to_numpy()
        else:
            shiftParams = []
            for g in iter_range:
                shiftParams.append(self.fitResults[f'eps_{g}'].apply(function))
            shiftParams = np.asarray(shiftParams)

        if units.lower() == 'ppm':
            shiftParams *= 1 / (2 * np.pi * self.hzperppm)
        elif units.lower() == 'hz':
            shiftParams *= 1 / (2 * np.pi)
        elif units.lower() == 'raw':
            pass
        else:
            raise ValueError('Units must be Hz, ppm or raw.')

        return shiftParams

    def getLineShapeParams(self, units='Hz', function='mean'):
        """Return line broadening parameters (gamma and sigma) in specified units.

        Note/warning: Does not incorporate any implicit linewidth already in the basis set

        :param units: Output units, defaults to 'Hz'. Can be 'ppm' or 'raw'.
        :type units: str, optional
        :param function: Point statistic for MCMC approach, defaults to 'mean'
        :type function: str, optional
        :return: Tuple containing combined, lorentzian, and gaussian broadening terms.
            Nested tuples used for multiple metabolite groups.
            Combined is undefined for units=raw.
        :rtype: tuple
        """
        if self.model == 'lorentzian':
            if function is None:
                gamma = np.zeros([self.fitResults.shape[0], self.g])
                for g in range(self.g):
                    gamma[:, g] = self.fitResults[f'gamma_{g}'].to_numpy()
            else:
                gamma = []
                for g in range(self.g):
                    gamma.append(self.fitResults[f'gamma_{g}'].apply(function))
                gamma = np.asarray(gamma)

            if units.lower() == 'hz':
                gamma *= 1 / (np.pi)
            elif units.lower() == 'ppm':
                gamma *= 1 / (np.pi * self.hzperppm)
            elif units.lower() == 'raw':
                pass
            else:
                raise ValueError('Units must be Hz, ppm or raw.')
            combined = gamma
            return combined, gamma
        elif self.model in ['voigt', 'free_shift']:
            if function is None:
                gamma = np.zeros([self.fitResults.shape[0], self.g])
                sigma = np.zeros([self.fitResults.shape[0], self.g])
                for g in range(self.g):
                    gamma[:, g] = self.fitResults[f'gamma_{g}'].to_numpy()
                    sigma[:, g] = self.fitResults[f'sigma_{g}'].to_numpy()
            else:
                gamma = []
                sigma = []
                for g in range(self.g):
                    gamma.append(self.fitResults[f'gamma_{g}'].apply(function))
                    sigma.append(self.fitResults[f'sigma_{g}'].apply(function))
                gamma = np.asarray(gamma)
                sigma = np.asarray(sigma)

            if units.lower() == 'hz':
                gamma *= 1 / (np.pi)
                # Equation for image space FWHM of Gaussian
                # https://www.cmrr.umn.edu/stimulate/frame/fwhm/node1.html Eq 7
                with np.errstate(divide='ignore'):
                    sigma = 2.335 / (2 * np.pi * (np.sqrt(0.5) / sigma))
            elif units.lower() == 'ppm':
                gamma *= 1 / (np.pi * self.hzperppm)
                # Equation for image space FWHM of Gaussian
                # https://www.cmrr.umn.edu/stimulate/frame/fwhm/node1.html Eq 7
                with np.errstate(divide='ignore'):
                    sigma = 2.335 / (2 * np.pi * (np.sqrt(0.5) / sigma))
                sigma /= self.hzperppm
            elif units.lower() == 'raw':
                pass
            else:
                raise ValueError('Units must be Hz, ppm or raw.')

            if units.lower() == 'raw':
                # Combining the values in raw units doesn't make sense
                combined = None
            else:
                # For ppm or Hz
                # https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
                combined = 0.5346 * gamma + np.sqrt(0.2166 * gamma**2 + sigma**2)
            return combined, gamma, sigma

    def getBaselineParams(self, complex=True, normalise=True):
        """ Return normalised complex baseline parameters."""
        bParams = []
        for b in range(self.n_baseline_bases):
            breal = self.fitResults[f'B_real_{b}'].mean()
            bimag = self.fitResults[f'B_imag_{b}'].mean()
            if complex:
                bParams.append(breal + 1j * bimag)
            else:
                bParams.extend([breal, bimag])

        bParams = np.asarray(bParams)
        if normalise:
            with np.errstate(divide='ignore', invalid='ignore'):
                return bParams / np.abs(bParams[0])
        else:
            return bParams

    def getQCParams(self, metab=None):
        """Returns peak wise SNR and FWHM (in Hz)"""
        if metab is None:
            return self.SNR.peaks.mean(), self.FWHM.mean()
        else:
            return self.SNR.peaks['SNR_' + metab].mean(), self.FWHM['fwhm_' + metab].mean()

    def getUncertainties(self, type='percentage', metab=None):
        """ Return the uncertainties (SD) on concentrations.
        Can either be in raw, molarity or molality or percentage uncertainties.

        """
        abs_std = []
        if metab is None:
            metab = self.metabs
        elif isinstance(metab, str):
            metab = [metab, ]
        for m in metab:
            if self.method == 'Newton':
                index = self.params_names_inc_comb.index(m)
                abs_std.append(np.sqrt(self.crlb[index]))
            elif self.method == 'MH':
                abs_std.append(self.fitResults[m].std())
        abs_std = np.asarray(abs_std)
        if type.lower() == 'raw':
            return abs_std
        elif type.lower() == 'molarity':
            return abs_std * self.concScalings['molarity']
        elif type.lower() == 'molality':
            return abs_std * self.concScalings['molality']
        elif type.lower() == 'internal':
            internal_ref = self.concScalings['internalRef']
            if self.method == 'Newton':
                internalRefIndex = self.params_names_inc_comb.index(internal_ref)
                internalRefSD = np.sqrt(self.crlb[internalRefIndex])
            elif self.method == 'MH':
                internalRefSD = self.fitResults[internal_ref].std()
            abs_std = np.sqrt(abs_std**2 + internalRefSD**2)
            return abs_std * self.concScalings['internal']
        elif type.lower() == 'percentage':
            vals = self.fitResults[metab].mean().to_numpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                perc_SD = abs_std / vals * 100
            perc_SD[perc_SD > 999] = 999   # Like LCModel :)
            perc_SD[np.isnan(perc_SD)] = 999
            return perc_SD
        else:
            raise ValueError('type must either be "absolute" or "percentage".')

    def plot(self, mrs, **kwargs):
        """Utility method to plot the results of a fit"""
        from fsl_mrs.utils.plotting import plot_fit
        return plot_fit(mrs, self, **kwargs)
