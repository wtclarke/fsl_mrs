#!/usr/bin/env python

# constants.py - Definition of all the constants
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT


 
H2O_PPM_TO_TMS     = 4.65       # Shift of water to Tetramethylsilane
H1_gamma           = 42.576     # MHz/tesla

# Concentration scaling parameters
TISSUE_WATER_DENSITY = {'GM':0.78,'WM':0.65,'CSF':0.97} 
#TISSUE_WATER_DENSITY reference: Ernst T, Kreis R, Ross BD. Absolute Quantitation of Water and Metabolites in the Human Brain. I. Compartments and Water. Journal of Magnetic Resonance, Series B 1993;102:1–8 doi: 10.1006/jmrb.1993.1055.
H2O_MOLECULAR_MASS = 18.01528   # g/mol
H2O_MOLALITY = 55.51E3    # mmol/kg     
H2O_PROTONS = 2

# T1 parameters 
# Derived from a survey of the literature. References listed below.
# Metabolite values derived from an average of NAA, Cr and Cho peaks.
STANDARD_T1 = { '3T':{'H2O_WM':0.97, # Ref: 1-6
                      'H2O_GM':1.50, # Ref: 1-6
                      'H2O_CSF':4.47, # Ref: 4
                      'METAB':1.29}, # Ref: 2, 7-9
                '7T':{'H2O_WM':1.21, # Ref: 1-6
                      'H2O_GM':2.05, # Ref: 1-6
                      'H2O_CSF':4.43, # Ref: 4
                      'METAB':1.43}} # Ref: 2, 7-9

# T2 parameters
STANDARD_T2 = { '3T':{'H2O_WM':0.073, # Ref: 1,3,10-11
                      'H2O_GM':0.088, # Ref: 1,3,10-11
                      'H2O_CSF':2.030, # Ref: 12
                      'METAB':0.194}, # Ref: 7-9,13-15
                '7T':{'H2O_WM':0.055, # Ref: 1,3,10-11
                      'H2O_GM':0.050, # Ref: 1,3,10-11
                      'H2O_CSF':1.050, # Ref: 12
                      'METAB':0.131}} # Ref: 7-9,13-15

'''
T1 & T2 References:
1. Stanisz GJ et al. doi: 10.1002/mrm.20605.
2. Ethofer T et al. doi: 10.1002/mrm.10640.
3. Wansapura JP et al. doi: 10.1002/(SICI)1522-2586(199904)9:4<531::AID-JMRI4>3.0.CO;2-L.
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