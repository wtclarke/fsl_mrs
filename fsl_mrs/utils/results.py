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

import fsl_mrs.utils.models as models
import fsl_mrs.utils.quantify as quant
import fsl_mrs.utils.qc as qc
from fsl_mrs.utils.misc import FIDToSpec, SpecToFID, calculate_lap_cov


class FitRes(object):
    """
       Collects fitting results
    """

    def __init__(self, model, method, names, metab_groups, baseline_order, B, ppmlim):
        """ Short class initilisation """
        # Initilise some basic parameters - includes fitting options
        # Populate parameter names
        self.model = model
        self.fill_names(names, baseline_order=baseline_order, metab_groups=metab_groups)
        self.method = method
        self.ppmlim = ppmlim
        self.baseline_order = baseline_order
        self.base_poly = B

        self.concScalings = {'internal': None, 'internalRef': None, 'molarity': None, 'molality': None, 'info': None}

    def loadResults(self, mrs, fitResults):
        "Load fitting results and calculate some metrics"
        # Populate data frame
        if fitResults.ndim == 1:
            self.fitResults = pd.DataFrame(data=fitResults[np.newaxis, :], columns=self.params_names)
        else:
            self.fitResults = pd.DataFrame(data=fitResults, columns=self.params_names)
        self.params = self.fitResults.mean().values

        # Store prediction, baseline, residual
        self.pred = self.predictedFID(mrs, mode='Full')
        self.pred_spec = FIDToSpec(self.pred)
        self.baseline = self.predictedFID(mrs, mode='Baseline')
        self.residuals = mrs.FID - self.pred

        # Calculate single point crlb and cov
        first, last = mrs.ppmlim_to_range(self.ppmlim)
        _, _, forward, _, _ = models.getModelFunctions(self.model)

        def forward_lim(p):
            return forward(p, mrs.frequencyAxis,
                           mrs.timeAxis,
                           mrs.basis,
                           self.base_poly,
                           self.metab_groups,
                           self.g)[first:last]
        data = mrs.get_spec(ppmlim=self.ppmlim)
        # self.crlb      = calculate_crlb(self.params,forward_lim,data)
        self.cov = calculate_lap_cov(self.params, forward_lim, data)
        self.crlb = np.diagonal(self.cov)
        std = np.sqrt(self.crlb)
        self.corr = self.cov / (std[:, np.newaxis] * std[np.newaxis, :])
        self.mse = np.mean(np.abs(FIDToSpec(self.residuals)[first:last])**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.perc_SD = np.sqrt(self.crlb) / self.params * 100
        self.perc_SD[self.perc_SD > 999] = 999   # Like LCModel :)
        self.perc_SD[np.isnan(self.perc_SD)] = 999

        # Calculate mcmc metrics
        if self.method == 'MH':
            self.mcmc_cov = self.fitResults.cov().values
            self.mcmc_cor = self.fitResults.corr().values
            self.mcmc_var = self.fitResults.var().values
            self.mcmc_samples = self.fitResults.values

        # VB metrics
        if self.method == 'VB':
            self.vb_cov = self.optim_out.cov
            self.vb_var = self.optim_out.var
            std = np.sqrt(self.vb_var)
            self.vb_corr = self.vb_cov / (std[:, np.newaxis] * std[np.newaxis, :])

        self.hzperppm = mrs.centralFrequency / 1E6

        # Calculate QC metrics
        self.FWHM, self.SNR = qc.calcQC(mrs, self, ppmlim=(0.2, 4.2))

        # Run relative concentration scaling to tCr in 'default' 1H MRS case.
        # Create combined metab at same time to avoid later errors.
        if (('Cr' in self.metabs) and ('PCr' in self.metabs)):
            self.combine([['Cr', 'PCr']])
            self.calculateConcScaling(mrs)

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
        :param internal_reference: Internal referencing emtabolite, defaults to ['Cr', 'PCr'] i.e. tCr
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
            newCRLB = jac @ self.cov @ jac
            self.crlb = np.concatenate((self.crlb, newCRLB[np.newaxis]))

        self.numMetabs = len(self.metabs)
        # self.params = self.fitResults.mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            params = self.fitResults.mean().values
            self.perc_SD = np.sqrt(self.crlb) / params * 100
        self.perc_SD[self.perc_SD > 999] = 999   # Like LCModel :)
        self.perc_SD[np.isnan(self.perc_SD)] = 999

        if self.method == 'MH':
            self.mcmc_cov = self.fitResults.cov().values
            self.mcmc_cor = self.fitResults.corr().values
            self.mcmc_var = self.fitResults.var().values

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
        if mode.lower() == 'full':
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
        else:
            raise ValueError('Unknown mode, must be one of: Full, baseline or a metabolite name.')
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

    def fill_names(self, names, baseline_order=0, metab_groups=None):
        """
        mrs            : MRS Object
        baseline_order : int
        metab_groups   : list (by default assumes single metab group)
        """
        self.metabs = deepcopy(names)
        self.original_metabs = deepcopy(names)
        self.numMetabs = len(self.metabs)

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
        if self.model.lower() == 'voigt':
            for i in range(g):
                self.params_names.append(f'sigma_{i}')

        for i in range(g):
            self.params_names.append(f"eps_{i}")

        self.params_names.extend(['Phi0', 'Phi1'])

        for i in range(0, baseline_order + 1):
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
            p0 = self.fitResults['Phi0'].values
            p1 = self.fitResults['Phi1'].values
        else:
            p0 = self.fitResults['Phi0'].apply(function)
            p1 = self.fitResults['Phi1'].apply(function)

        if phi0.lower() == 'degrees':
            p0 *= np.pi / 180.0
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
        if function is None:
            shiftParams = np.zeros([self.fitResults.shape[0], self.g])
            for g in range(self.g):
                shiftParams[:, g] = self.fitResults[f'eps_{g}'].values
        else:
            shiftParams = []
            for g in range(self.g):
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
        """ Return line broadening parameters (gamma and sigma) in specified units - default = Hz."""
        if self.model == 'lorentzian':
            if function is None:
                gamma = np.zeros([self.fitResults.shape[0], self.g])
                for g in range(self.g):
                    gamma[:, g] = self.fitResults[f'gamma_{g}'].values
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
        elif self.model == 'voigt':
            if function is None:
                gamma = np.zeros([self.fitResults.shape[0], self.g])
                sigma = np.zeros([self.fitResults.shape[0], self.g])
                for g in range(self.g):
                    gamma[:, g] = self.fitResults[f'gamma_{g}'].values
                    sigma[:, g] = self.fitResults[f'sigma_{g}'].values
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
                sigma *= 1 / (np.pi)
            elif units.lower() == 'ppm':
                gamma *= 1 / (np.pi * self.hzperppm)
                sigma *= 1 / (np.pi * self.hzperppm)
            elif units.lower() == 'raw':
                pass
            else:
                raise ValueError('Units must be Hz, ppm or raw.')

            combined = gamma / 2 + np.sqrt((gamma**2 / 4.0) + (sigma * 2 * np.log(2))**2)
            return combined, gamma, sigma

    def getBaselineParams(self, complex=True, normalise=True):
        """ Return normalised complex baseline parameters."""
        bParams = []
        for b in range(self.baseline_order + 1):
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
            elif self.method == 'VB':
                index = self.params_names_inc_comb.index(m)
                abs_std.append(np.sqrt(self.vb_var[index]))
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
            elif self.method == 'VB':
                index = self.params_names_inc_comb.index(internal_ref)
                internalRefSD = np.sqrt(self.vb_var[index])
            abs_std = np.sqrt(abs_std**2 + internalRefSD**2)
            return abs_std * self.concScalings['internal']
        elif type.lower() == 'percentage':
            vals = self.fitResults[metab].mean().to_numpy()
            perc_SD = abs_std / vals * 100
            perc_SD[perc_SD > 999] = 999   # Like LCModel :)
            perc_SD[np.isnan(perc_SD)] = 999
            return perc_SD
        else:
            raise ValueError('type must either be "absolute" or "percentage".')
