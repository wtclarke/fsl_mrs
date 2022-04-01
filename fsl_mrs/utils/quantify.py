# quantify.py - Quantify the results of MRS fits
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.utils.constants import H2O_MOLALITY, TISSUE_WATER_DENSITY,\
    STANDARD_T1, STANDARD_T2, H1_gamma,\
    H2O_PROTONS, WATER_SCALING_METAB,\
    WATER_SCALING_METAB_PROTONS,\
    WATER_SCALING_DEFAULT_LIMITS


class NoWaterScalingMetabolite(Exception):
    pass


class FIDIntegrator:
    def __init__(self, mrs_obj, limits=None):
        self.t_axis = mrs_obj.getAxes('time')
        self.ppm_axis = mrs_obj.getAxes()

        if limits is not None:
            first, last = mrs_obj.ppmlim_to_range(limits)
            self.limits = (first, last)
        else:
            self.limits = (0, self.t_axis.size)

        self.fid = None

    @property
    def integral(self):
        return self._calculate_area(self.fid)

    def _calculate_area(self, FID):
        """
            Calculate area of the abs real part of the spectrum between two limits
        """
        spec = FIDToSpec(FID, axis=0)[self.limits[0]:self.limits[1]]
        area = np.trapz(np.abs(np.real(spec)), axis=0)
        return area


class WaterRef(FIDIntegrator):
    def __init__(self, mrs_obj, limits=None):
        super().__init__(mrs_obj, limits)

        self.original_fid = mrs_obj.H2O

        self._fit_w_ref()

    def _fit_w_ref(self):
        '''Fit unsuppressed water with single voigt lineshape.
        Fitted fid is then phase and frequency corrected.'''
        def fid_func(t, amp, gamma, sigma, omega, phi):
            return amp\
                * np.exp(-t * (gamma + t * sigma + 1j * omega))\
                * np.exp(1j * phi)

        def fit_func(p):
            amp, gamma, sigma, omega, phi = p
            fid = fid_func(self.t_axis, amp, gamma, sigma, omega, phi)
            return np.mean(np.abs(fid - self.original_fid)**2)

        p0 = [np.mean(np.abs(self.original_fid[:5])), 10, 10, 0, 0]
        bounds = ((0, None),
                  (0, None),
                  (0, None),
                  (None, None),
                  (None, None))
        pout = minimize(fit_func, p0, bounds=bounds)
        self.fid = fid_func(self.t_axis, *pout.x[:3], 0, 0)

    def plot_fit(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        ax1.plot(self.t_axis, self.original_fid.real, label='original')
        ax1.plot(self.t_axis, self.fid.real, label='fit')
        ax1.set_xscale('log')

        ax2.plot(self.ppm_axis, FIDToSpec(self.original_fid).real, label='original')
        ax2.plot(self.ppm_axis, FIDToSpec(self.fid).real, label='fit')
        ax2.set_xlim([3.65, 5.65])
        return fig


class RefIntegral(FIDIntegrator):
    def __init__(self, mrs_obj, res_obj, metab, limits):
        super().__init__(mrs_obj, limits)
        self.original_fid = res_obj.predictedFID(
            mrs_obj, mode=metab, noBaseline=False, no_phase=False)
        self.fid = res_obj.predictedFID(
            mrs_obj, mode=metab, noBaseline=True, no_phase=True)


class QuantificationInfo(object):
    """ Class encapsulating the information required to run internal water quantification scaling.
        Requires (arguments to init):
            Echo time
            Repetition time
            List of metabolites in basis set used for fitting
            Central frequency (for determination of field strength)

        Might require (keyword arguments to init):
            T2 values - if override desired or no defaults
            T1 values - if override desired or no defaults

            Water scaling reference metabolite - if override desired or none of the defaults are included
            Water scaling reference metabolite, number of protons - if as previous
            Water scaling reference metabolite, integration limits - if as previous

        Optional (added through methods):
            Tissue fractions - if not provided pure water concentration assumed
            Tissue densitites - if defaults to be overridden
            H20 integration limits - defaults to be overridden
    """

    def __init__(self,
                 te,
                 tr,
                 metabolites,
                 cf,
                 fa=90,
                 t2=None,
                 t1=None,
                 water_ref_metab=None,
                 water_ref_metab_protons=None,
                 water_ref_metab_limits=None):
        """Constructor for QuantificationInfo. Requires sequence information.

        :param te: Sequence echo time in seconds
        :type te: float
        :param tr: Sequence repetition time in seconds
        :type tr: float
        :param metabolites: List of metabolite names in basis set
        :type metabolites: [str]
        :param cf: Central (spectrometer) frequency in MHz
        :type cf: float
        :param fa: Excitation flip-angle, defaults to 90
        :type fa: int, optional
        :param t2: Overwrite default T2s or provide if no defaults, defaults to None.
            Must be dict containing 'H2O_GM', 'H2O_WM', 'H2O_CSF', 'METAB' fields.
        :type t2: dict, optional
        :param t1: Overwrite default T1s or provide if no defaults, defaults to None.
            Must be dict containing 'H2O_GM', 'H2O_WM', 'H2O_CSF', 'METAB' fields.
        :type t1: dict, optional
        :param water_ref_metab: Override default water reference metabolite, defaults to None
        :type water_ref_metab: str, optional
        :param water_ref_metab_protons: Number of protons water_ref_metab is equivalent to, defaults to None
        :type water_ref_metab_protons: int, optional
        :param water_ref_metab_limits: Integration limits for water_ref_metab spectrum, defaults to None
        :type water_ref_metab_limits: tuple, optional
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        """
        if te > 0.5:
            print('te is an unusal value. Expecting a value < 0.5 s.')
        elif te < 0:
            raise ValueError('te must be larger than 0.0 s!')
        self.te = te

        if tr > 20:
            print('tr is an unusal value. Expecting a value < 20 s.')
        elif tr < 0:
            raise ValueError('tr must be larger than 0.0 s!')
        self.tr = tr

        if fa < 0:
            raise ValueError('fa must be larger than 0 degrees!')
        self.fa = fa

        # Set default values
        # Tissue densitites
        self._densitites = TISSUE_WATER_DENSITY

        # Tissue fractions
        self._fractions = None

        # T1s
        self._handle_t1(cf, specified=t1)
        # T2s
        self._handle_t2(cf, specified=t2)

        # Reference metabolite
        self._handle_ref_metab(metabolites,
                               specified_metab=water_ref_metab,
                               specified_protons=water_ref_metab_protons,
                               specified_limits=water_ref_metab_limits)

        # Water reference integral limits
        self.h2o_limits = (1.65, 7.65)

        # Additional correction term
        self._add_corr = 1.0

    def _handle_t1(self, cf, specified=None):
        """ Use specified values or select T1 values based on field strength (7T or 3T)"""
        if specified:
            required = ['H2O_GM', 'H2O_WM', 'H2O_CSF', 'METAB']
            if not all(item in specified.keys() for item in required):
                raise ValueError(f'Specified T1 values must contain entries for all of {required}.')
            self.t1 = specified
        else:
            field = cf / H1_gamma
            if field > 6.5 and field < 7.5:
                self.t1 = STANDARD_T1['7T']
            elif field > 2.5 and field < 3.5:
                self.t1 = STANDARD_T1['3T']
            else:
                raise ValueError(f'No stored T1 values for {field}T scanner. Specify values manually.')

    def _handle_t2(self, cf, specified=None):
        """ Use specified values or select T2 values based on field strength (7T or 3T)"""
        if specified:
            required = ['H2O_GM', 'H2O_WM', 'H2O_CSF', 'METAB']
            if not all(item in specified.keys() for item in required):
                raise ValueError(f'Specified T1 values must contain entries for all of {required}.')
            self.t2 = specified
        else:
            field = cf / H1_gamma
            if field > 6.5 and field < 7.5:
                self.t2 = STANDARD_T2['7T']
            elif field > 2.5 and field < 3.5:
                self.t2 = STANDARD_T2['3T']
            else:
                raise ValueError(f'No stored T2 values for {field}T scanner. Specify values manually.')

    def _handle_ref_metab(self, metab_list, specified_metab=None, specified_protons=None, specified_limits=None):
        """Set the water referencing metab using either the predefined list or the specified matabolite."""
        if specified_metab:
            if not isinstance(specified_metab, str):
                raise TypeError(f'Specified reference metabolite should be string not {type(specified_metab)}')
            if not isinstance(specified_protons, int):
                raise TypeError(f'Specified reference metabolite protons should be int not {type(specified_protons)}')
            if not isinstance(specified_limits, (list, tuple)):
                raise TypeError(f'Specified reference metabolite limits should be tuple not {type(specified_limits)}')

            if specified_metab in metab_list:
                self.ref_metab = specified_metab
                self.ref_protons = specified_protons
                self.ref_limits = specified_limits
            else:
                raise ValueError(f"Specified matabolite {specified_metab} isn't in the list of basis spectra.")
        else:
            # Select from a predetermined list of sensible options
            for wsm, nprot, lim in zip(WATER_SCALING_METAB,
                                       WATER_SCALING_METAB_PROTONS,
                                       WATER_SCALING_DEFAULT_LIMITS):
                if wsm in metab_list:
                    self.ref_metab = wsm
                    self.ref_protons = nprot
                    self.ref_limits = lim
                    return
            # If nothing is suitable ...
            raise NoWaterScalingMetabolite('No suitable metabolite has been identified for water scaling.')

    def set_fractions(self, fractions):
        """Set tissue volume fractions

        :param fractions: Tissue volume fractions, must contain 'WM', 'GM', 'CSF' fields.
        :type fractions: dict
        :raises ValueError: [description]
        :raises TypeError: Raised if fractions are not a dictionary
        """
        required = ['WM', 'GM', 'CSF']
        if isinstance(fractions, dict):
            if all(item in fractions.keys() for item in required)\
                    and np.isclose(np.sum([fractions['WM'], fractions['GM'], fractions['CSF']]), 1.0, atol=1E-2):
                self._fractions = fractions
            elif all(item in fractions.keys() for item in required)\
                    and np.isclose(np.sum([fractions['WM'], fractions['GM'], fractions['CSF']]), 1.0, atol=1E-1):
                original_frac = np.asarray([fractions['WM'], fractions['GM'], fractions['CSF']])
                sum_value = np.sum(original_frac)
                norm_vec = original_frac / sum_value
                norm_fractions = {'WM': norm_vec[0], 'GM': norm_vec[1], 'CSF': norm_vec[2]}
                self._fractions = norm_fractions
                warnings.warn("fractions should be a dict containing 'WM', 'GM', 'CSF' keys"
                              f", and must sum to 1. Currently they are: {fractions} (sum={sum_value:0.4f})."
                              f"This fraction is off by <10% and has been normalised to: {norm_fractions}.")
            else:
                sum_value = np.sum([fractions['WM'], fractions['GM'], fractions['CSF']])
                raise ValueError("fractions must be a dict containing 'WM', 'GM', 'CSF' keys"
                                 f", and must sum to 1. Currently they are: {fractions} (sum={sum_value:0.4f}).")
        else:
            raise TypeError(f'fractions must be a dict, not {type(fractions)}.')

    def set_densitites(self, densitites):
        """[summary]

        :param densitites: Tissue densities, must contain 'WM', 'GM', 'CSF' fields. In units of g/ml.
        :type densitites: dict
        :raises ValueError: [description]
        :raises TypeError: [description]
        """
        required = ['WM', 'GM', 'CSF']
        if isinstance(densitites, dict):
            if all(item in densitites.keys() for item in required):
                self._densitites = densitites
            else:
                raise ValueError("densitites must be a dict containing 'WM', 'GM', 'CSF' keys")
        else:
            raise TypeError(f'densitites must be a dict, not {type(densitites)}.')

    def set_h20_limits(self, low=1.65, high=7.65):
        """Set the water reference integral limits

        :param low: Lower limit, defaults to 1.65
        :type low: float, optional
        :param high: Upper limit, defaults to 7.65
        :type high: float, optional
        """
        self.h2o_limits = (low, high)

    # Properties
    # tissue densitites d_GM, d_WM, d_CSF
    @property
    def d_GM(self):
        return self._densitites['GM']

    @property
    def d_WM(self):
        return self._densitites['WM']

    @property
    def d_CSF(self):
        return self._densitites['CSF']

    # tissue volume fractions f_GM, f_WM, f_CSF
    @property
    def f_GM(self):
        if self._fractions:
            return self._fractions['GM']
        else:
            return None

    @property
    def f_WM(self):
        if self._fractions:
            return self._fractions['WM']
        else:
            return None

    @property
    def f_CSF(self):
        if self._fractions:
            return self._fractions['CSF']
        else:
            return None

    # tissue mole fractions f_GM_H2O, f_WM_H2O, f_CSF_H2O
    # Implementing equation 5 in https://doi.org/10.1002/nbm.4257
    # f_X_H2O = f_X*d_X / (f_GM*d_GM + f_WM*d_WM f_CSF*d_CSF)
    @property
    def f_GM_H2O(self):
        """Grey matter (GM) mole fraction

        :return: GM mole fraction
        :rtype: float
        """
        if self._fractions:
            return self.f_GM * self.d_GM\
                / (self.f_GM * self.d_GM
                   + self.f_WM * self.d_WM
                   + self.f_CSF * self.d_CSF)
        else:
            return None

    @property
    def f_WM_H2O(self):
        """White matter (WM) mole fraction

        :return: WM mole fraction
        :rtype: float
        """
        if self._fractions:
            return self.f_WM * self.d_WM\
                / (self.f_GM * self.d_GM
                   + self.f_WM * self.d_WM
                   + self.f_CSF * self.d_CSF)
        else:
            return None

    @property
    def f_CSF_H2O(self):
        """Cerebral spinal fluid (CSF) mole fraction

        :return: CSF mole fraction
        :rtype: float
        """
        if self._fractions:
            return self.f_CSF * self.d_CSF\
                / (self.f_GM * self.d_GM
                   + self.f_WM * self.d_WM
                   + self.f_CSF * self.d_CSF)
        else:
            return None

    # Relaxation parameters
    # R_H2O_GM, R_H2O_WM, R_H2O_CSF, R_H2O, R_M
    # Assumes TR >> TE and excitation FA = 90 deg
    def _calc_relax(self, t1, t2):
        t1_term = (1 - np.exp(-self.tr / t1))\
            / ((1 - np.cos(self.fa * np.pi / 180) * np.exp(-self.tr / t1)))
        t2_term = np.exp(-self.te / t2)
        return t1_term * t2_term

    @property
    def R_H2O_GM(self):
        return self._calc_relax(self.t1['H2O_GM'], self.t2['H2O_GM'])

    @property
    def R_H2O_WM(self):
        return self._calc_relax(self.t1['H2O_WM'], self.t2['H2O_WM'])

    @property
    def R_H2O_CSF(self):
        return self._calc_relax(self.t1['H2O_CSF'], self.t2['H2O_CSF'])

    # A 50/50 average between WM/GM
    # Used in the absence of fractions
    @property
    def R_H2O(self):
        t1 = (self.t1['H2O_WM'] + self.t1['H2O_GM']) / 2.0
        t2 = (self.t2['H2O_WM'] + self.t2['H2O_GM']) / 2.0
        return self._calc_relax(t1, t2)

    @property
    def R_M(self):
        return self._calc_relax(self.t1['METAB'], self.t2['METAB'])

    # Water concentrations
    # relax_corr_water_molal, relax_corr_water_molar
    # relax_corr_water_molal uses mole fractions
    # relax_corr_water_molar uses volume fractions * densitites
    @property
    def relax_corr_water_molal(self):
        """Relaxation (T1, T2) corrected water molality (mmol/kg).
        If volume fractions aren't availible then relaxation correction will be based on
        a 50/50 split of GM/WM T1/T2s and pure water will be assumed.

        :return: concentration
        :rtype: float
        """
        if self._fractions is None:
            return self.R_H2O * H2O_MOLALITY
        else:
            return H2O_MOLALITY * (self.f_GM_H2O * self.R_H2O_GM
                                   + self.f_WM_H2O * self.R_H2O_WM
                                   + self.f_CSF_H2O * self.R_H2O_CSF)

    @property
    def relax_corr_water_molar(self):
        """Relaxation (T1, T2) corrected water molariyt (mmol/dm^3 = mM).
        If volume fractions aren't availible then relaxation correction will be based on
        a 50/50 split of GM/WM T1/T2s and pure water will be assumed.

        :return: concentration
        :rtype: float
        """
        if self._fractions is None:
            return self.R_H2O * H2O_MOLALITY
        else:
            return H2O_MOLALITY * (self.f_GM * self.d_GM * self.R_H2O_GM
                                   + self.f_WM * self.d_WM * self.R_H2O_WM
                                   + self.f_CSF * self.d_CSF * self.R_H2O_CSF)

    # Metabolite relaxation term
    @property
    def relax_corr_metab(self):
        return 1 / self.R_M

    # Additional terms
    @property
    def csf_corr(self):
        if self._fractions is None:
            return 1.0
        else:
            return 1 / (1 - self.f_CSF)

    @property
    def add_corr(self):
        return self._add_corr

    @add_corr.setter
    def add_corr(self, value):
        self._add_corr = value


def quantifyInternal(reference, concentrations, names):
    """Calculate scaling for internal referencing

    :param reference: Metabolite to use for internal scaling
    :type reference: str
    :param concentrations: Metabolite concentrations in order of names
    :type concentrations: list
    :param names: Names of all metabolites listed in concentrations
    :type names: list
    :raises ValueError: If chosen reference doesn't appear in names
    :return: Internal reference scaling value
    :rtype: float
    """
    concSum = 0
    if isinstance(reference, list):
        for m in reference:
            if m not in names:
                raise ValueError(f'Internal reference {m} is not a recognised metabolite.')
            concSum += concentrations[names.index(m)]
    else:
        if reference not in names:
            raise ValueError(f'Internal reference {reference} is not a recognised metabolite.')
        concSum += concentrations[names.index(reference)]

    with np.errstate(divide='ignore', invalid='ignore'):
        return 1 / concSum


def quantifyWater(mrs, results, quant_info, verbose=False):
    """Calculate scalings required to take raw concentrations to molarity or molality units.

    Steps:
        1) Calculate areas of the peaks in the water reference and of the fitted reference peaks.
        2) Calculate the scalings between the reference peak area and water area
        3) Add in the scalings between the reference peak and all other peaks.

    :param mrs: MRS object with unsuppressed water reference data.
    :type mrs: fsl_mrs.core.mrs.MRS
    :param results: FSL-MRS results object.
    :type results: fsl_mrs.utils.results.FitRes
    :param quant_info: QuantificationInfo object containing required inputs.
    :type quant_info: fsl_mrs.utils.quantify.QuantificationInfo
    :param verbose: Enable verbose output, defaults to False
    :type verbose: bool, optional
    :return: conc_molal, scaling parameter to convert raw fitted concentrations to molality units of mols/kg
    :rtype: float
    :return: conc_molar, caling parameter to convert raw fitted concentrations to molarity units of mols/dm^3
    :rtype: float
    :return: Dict containing water and reference integration classes.
    :rtype: dict
    """

    # Calculate observed areas
    wref = WaterRef(mrs, quant_info.h2o_limits)
    SH2OObs = wref.integral

    mref = RefIntegral(mrs, results, quant_info.ref_metab, quant_info.ref_limits)
    SMObs = mref.integral

    # Calculate concentration scalings
    # EQ 4 and 6 in https://doi.org/10.1002/nbm.4257
    # conc_molal =  (SMObs *(Q.f_GM_H20*Q.R_H2O_GM + Q.f_WM_H20*Q.R_H2O_WM + Q.f_CSF_H20*Q.R_H2O_CSF)\
    #                       / (SH2OObs*(1-Q.f_CSF_H20)*Q.R_M)) \
    #                 * (H2O_PROTONS/refProtons)\
    #                 * H2O_MOLALITY

    # conc_molar =  (SMObs *(Q.f_GM*Q.d_GM*Q.R_H2O_GM + Q.f_WM*Q.d_WM*Q.R_H2O_WM + Q.f_CSF*Q.d_CSF*Q.R_H2O_CSF)\
    #                       / (SH2OObs*(1-Q.f_CSF)*Q.R_M))\
    #                 * (H2O_PROTONS/refProtons)\
    #                 * H2O_MOLALITY

    # Note the difference between Q.f_X and Q.f_X_H2O. Equation 5 of reference. With thanks to Alex Craig-Craven
    # for pointing this out.

    conc_molal = (SMObs / SH2OObs) * (H2O_PROTONS / quant_info.ref_protons) * \
        quant_info.relax_corr_water_molal * quant_info.csf_corr * quant_info.add_corr * quant_info.relax_corr_metab
    conc_molar = (SMObs / SH2OObs) * (H2O_PROTONS / quant_info.ref_protons) * \
        quant_info.relax_corr_water_molar * quant_info.csf_corr * quant_info.add_corr * quant_info.relax_corr_metab

    if verbose:
        rcorwaterconc = quant_info.relax_corr_water_molar
        metabRelaxCorr = quant_info.relax_corr_metab
        print(f'Metabolite area = {SMObs:0.2e}')
        print(f'Water area = {SH2OObs:0.2e}')
        print(f'Relaxation corrected water concentration (molal) = {quant_info.relax_corr_water_molal:0.2e} mmol/kg')
        print(f'Relaxation corrected water concentration (molar) = {rcorwaterconc:0.2e} mM')
        print(f'metabolite relaxation correction  = {metabRelaxCorr:0.2e}')
        print(f'H2O to ref molality scaling = {conc_molal:0.2e}')
        print(f'H2O to ref molarity scaling = {conc_molar:0.2e}')

    # Calculate other metabolites to reference scaling
    metabtoRefScaling = quantifyInternal(quant_info.ref_metab,
                                         results.getConc(),
                                         results.metabs)
    conc_molal *= metabtoRefScaling
    conc_molar *= metabtoRefScaling

    if verbose:
        print(f'Ref to other metabolite scaling = {metabtoRefScaling:0.2e}')
        print(f'Final molality scaling = {conc_molal:0.2e}')
        print(f'Final molarity scaling = {conc_molar:0.2e}')

    return conc_molal, conc_molar, {'metab_ref': mref, 'water_ref': wref}


def create_quant_info(header, mrs, tissueFractions=None, additional_scale=1.0):
    """ Create a QuantificationInfo object given NIFTI-MRS header and mrs."""
    q_info = QuantificationInfo(
        header['EchoTime'],
        header['RepetitionTime'],
        mrs.names,
        mrs.centralFrequency / 1E6)

    if tissueFractions:
        q_info.set_fractions(tissueFractions)
    q_info.add_corr = additional_scale
    return q_info
