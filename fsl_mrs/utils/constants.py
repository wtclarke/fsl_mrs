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
TISSUE_WATER_DENSITY = {'GM':0.78,'WM':0.65,'CSF':0.97} # Kreis R, Ernst T, Ross BD. Absolute quantitation of water and metabolites in the human brain. II. Metabolite concentrations. J Magn Reson B. 1993;102:9-19.
H2O_MOLECULAR_MASS = 18.01528   # g/mol
H2O_MOLALITY = 55.51E3    # mmol/kg     
H2O_PROTONS = 2

# T2 parameters
# 3T from https://onlinelibrary.wiley.com/doi/epdf/10.1002/nbm.3914
# 7T from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4549223/ - except copied CSF
STANDARD_T2 = { '7T':{'H2O_GM':0.05,
                     'H2O_WM':0.055,
                     'H2O_CSF':2.55,
                     'METAB':0.160},
                '3T':{'H2O_GM':0.11,
                     'H2O_WM':0.08,
                     'H2O_CSF':2.55,
                     'METAB':0.271}}
