# synthetic.py - Create synthetic data basis sets
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford 
# SHBASECOPYRIGHT

def syntheticFromBasis(basis,
                        concentrations,
                        broadening = 9.0,
                        shifting=0.0,
                        baseline = [0,0],
                        coilamps = [1.0],
                        coilphase = [0.0],
                        noisecovariance =[[0.1]],
                        bandwidth = 4000,
                        points = 2048,
                        centralfrequency = 123.0):

    # sort out inputs
