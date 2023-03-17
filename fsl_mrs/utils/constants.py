#!/usr/bin/env python

# constants.py - Definition of all the constants
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

# From https://en.wikipedia.org/wiki/Gyromagnetic_ratio
# except for 1H https://physics.nist.gov/cgi-bin/cuu/Value?gammappbar
# MHz/tesla
from dataclasses import dataclass


GYRO_MAG_RATIO = {
    '1H': 42.576,
    '2H': 6.536,
    '13C': 10.7084,
    '31P': 17.235}

PPM_SHIFT = {
    '1H': 4.65,
    '2H': 4.65,  # Placeholder?
    '13C': 0.0,
    '31P': 0.0}

PPM_RANGE = {
    '1H': (0.2, 4.2),
    '2H': (0.0, 6),
    '13C': (10, 100),
    '31P': (-20, 10)}

NOISE_REGION = {
    '1H': ([None, int(-2)], [10, None]),
    '2H': ([None, int(-2)], [10, None]),
    '31P': ([None, int(-15)], [12, None])}


@dataclass
class Nucleus_Info:
    nucleus: str
    gamma: float
    ppm_shift: float
    ppm_range: tuple
    noise_range: tuple


def nucleus_constants(nuc: str) -> Nucleus_Info:
    """Return the constants stored for a particular nucleus

    :param nuc: Nucleus string. E.g. 1H, 2H, 31P, 13C
    :type nuc: str
    :return: Object storing nucleus information.
    :rtype: Nucleus_Info
    """
    def find_nuc_in_dict(n, d):
        if n in d:
            return d[n]
        else:
            return None

    return Nucleus_Info(
        nuc,
        find_nuc_in_dict(nuc, GYRO_MAG_RATIO),
        find_nuc_in_dict(nuc, PPM_SHIFT),
        find_nuc_in_dict(nuc, PPM_RANGE),
        find_nuc_in_dict(nuc, NOISE_REGION))


# Concentration scaling parameters
TISSUE_WATER_DENSITY = {'GM': 0.78, 'WM': 0.65, 'CSF': 0.97}
# TISSUE_WATER_DENSITY reference: Ernst T, Kreis R, Ross BD.
# Absolute Quantitation of Water and Metabolites in the Human Brain. I.
# Compartments and Water.
# Journal of Magnetic Resonance, Series B 1993;102:1–8
# doi: 10.1006/jmrb.1993.1055.

H2O_MOLECULAR_MASS = 18.01528   # g/mol
H2O_MOLALITY = 55.51E3    # mmol/kg
H2O_PROTONS = 2

# Water referencing metabolites
# Define a list of sensible metabolites to use and
# the number of protons between the default limits of 2 and 5
WATER_SCALING_METAB = ['Cr', 'PCr', 'NAA']
WATER_SCALING_METAB_PROTONS = [5, 5, 3]
WATER_SCALING_DEFAULT_LIMITS = [(2, 5), (2, 5), (1.8, 2.2)]

# T1 parameters
# Derived from a survey of the literature. References listed below.
# Metabolite values derived from an average of NAA, Cr and Cho peaks.
STANDARD_T1 = {'3T': {'H2O_WM': 0.97,  # Ref: 1-6
                      'H2O_GM': 1.50,  # Ref: 1-6
                      'H2O_CSF': 4.47,  # Ref: 4
                      'METAB': 1.29},  # Ref: 2, 7-9
               '7T': {'H2O_WM': 1.21,  # Ref: 1-6
                      'H2O_GM': 2.05,  # Ref: 1-6
                      'H2O_CSF': 4.43,  # Ref: 4
                      'METAB': 1.43}}  # Ref: 2, 7-9

# T2 parameters
STANDARD_T2 = {'3T': {'H2O_WM': 0.073,  # Ref: 1,3,10-11
                      'H2O_GM': 0.088,  # Ref: 1,3,10-11
                      'H2O_CSF': 2.030,  # Ref: 12
                      'METAB': 0.194},  # Ref: 7-9,13-15
               '7T': {'H2O_WM': 0.055,  # Ref: 1,3,10-11
                      'H2O_GM': 0.050,  # Ref: 1,3,10-11
                      'H2O_CSF': 1.050,  # Ref: 12
                      'METAB': 0.131}}  # Ref: 7-9,13-15

'''
T1 & T2 References:
1. Stanisz GJ et al. doi: 10.1002/mrm.20605.
2. Ethofer T et al. doi: 10.1002/mrm.10640.
3. Wansapura JP et al. doi: 10.1002/
(SICI)1522-2586(199904)9:4<531::AID-JMRI4>3.0.CO;2-L.
4. Rooney WD et al. doi: 10.1002/mrm.21122.
5. Dieringer MA et al. doi: 10.1371/journal.pone.0091318.
6. Wright PJ et al. doi: 10.1007/s10334-008-0104-8.
7. Mlynárik V et al. doi: 10.1002/nbm.713.
8. Li Y. doi: 10.4172/2155-9937.S1-002.
9. An L et al. doi: 10.1002/mrm.26612.
10. Gelman N et al. doi: 10.1148/radiology.210.3.r99fe41759.
11. Bartha R et al. doi: 10.1002/mrm.10112.
12. Spijkerman JM et al. doi: 10.1007/s10334-017-0659-3.
13. Marjańska M et al. doi: 10.1002/nbm.1754.
14. Träber F et al. doi: 10.1002/jmri.20053.
15. Wyss PO et al. doi: 10.1002/mrm.27067.
'''

'''
MCMC PRIORS
Modify these values to change the centre or width (SD) of
the Gaussian priors applied to the MCMC optimised fitting routine.
Priors are defined for the lorentzian and voigt (default) models.
Use the disable_mh_priors flag to fit_FSLModel or the
disable_MH_priors command line argument to set all priors to uniform.

'_loc' values are the centre and '_scale' values are the standard
deviation of the Guassian prior.
'''

MCMC_PRIORS = {'lorentzian': {'conc_loc': 0.0, 'conc_scale': 1E0,
                              'gamma_loc': 5.0, 'gamma_scale': 2.5,  # Hz
                              'eps_loc': 0.0, 'eps_scale': 0.005,    # ppm
                              'phi0_loc': 0.0, 'phi0_scale': 5.0,   # degrees
                              'phi1_loc': 0.0, 'phi1_scale': 1E-5},  # seconds
               'voigt': {'conc_loc': 0.0, 'conc_scale': 1E0,
                         'gamma_loc': 5.0, 'gamma_scale': 2.5,  # Hz
                         'sigma_loc': 5.0, 'sigma_scale': 2.5,  # Hz
                         'eps_loc': 0.0, 'eps_scale': 0.005,    # ppm
                         'phi0_loc': 0.0, 'phi0_scale': 5.0,   # degrees
                         'phi1_loc': 0.0, 'phi1_scale': 1E-5}}  # seconds

'''
DEFAULT MM BASIS SET
Shift and amplitude values for the default macromolecule basis set
'''
DEFAULT_MM_PPM = [0.9, 1.2, 1.4, 1.7, [2.08, 2.25, 1.95, 3.0]]
DEFAULT_MM_AMP = [3.0, 2.0, 2.0, 2.0, [1.33, 0.33, 0.33, 0.4]]

DEFAULT_MM_MEGA_PPM = [[0.915, 3.000], ]
DEFAULT_MM_MEGA_AMP = [[3.75, 2.0], ]
