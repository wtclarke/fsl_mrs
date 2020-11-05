# synthetic.py - Create synthetic data from analytic functions
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

import numpy as np


def syntheticFID(coilamps=[1.0],
                 coilphase=[0.0],
                 noisecovariance=[[0.1]],
                 bandwidth=4000,
                 points=2048,
                 centralfrequency=123.2,
                 chemicalshift=[-2, 3],
                 amplitude=[1.0, 1.0],
                 phase=[0, 0],
                 damping=[20, 20],
                 linewidth=None,
                 g=[0.0, 0.0],
                 begintime=0.0,
                 nucleus='1H'):

    inputs = locals()
    # Check noisecovariance is Ncoils x Ncoils
    ncoils = len(coilamps)
    noisecovariance = np.asarray(noisecovariance)
    if len(coilphase) != ncoils:
        raise ValueError('Length of coilamps and coilphase must match.')
    if noisecovariance.shape != (ncoils, ncoils):
        raise ValueError('noisecovariance must be ncoils x ncoils.')

    noise = np.random.multivariate_normal(np.zeros((ncoils)),
                                          noisecovariance,
                                          points) + \
        1j * np.random.multivariate_normal(np.zeros((ncoils)),
                                           noisecovariance,
                                           points)

    dwelltime = 1/bandwidth
    taxis = np.linspace(0.0, dwelltime*(points - 1), points)
    syntheticFID = np.zeros(points, dtype=np.complex128)
    ttrue = taxis + begintime

    if linewidth is not None:
        damping = np.asarray(linewidth) * np.pi

    for a, p, d, cs, gg in zip(amplitude, phase, damping, chemicalshift, g):
        # Lorentzian peak at chemicalShift
        syntheticFID += a * np.exp(1j*p) * np.exp(-d*(1-gg+gg*ttrue)*ttrue
                                                  + 1j*2*np.pi
                                                  * cs
                                                  * centralfrequency
                                                  * ttrue)

    FIDs = []
    for cDx, (camp, cphs) in enumerate(zip(coilamps, coilphase)):
        FIDs.append((camp*np.exp(1j*cphs)*syntheticFID)+noise[:, cDx])

    freqAxis = np.linspace(-bandwidth/2, bandwidth/2, points)
    ppmaxis = freqAxis/centralfrequency

    headers = {'noiseless': syntheticFID,
               'cov': noisecovariance,
               'taxis': taxis,
               'faxis': freqAxis,
               'ppmaxis': ppmaxis,
               'inputopts': inputs,
               'centralFrequency': centralfrequency,
               'bandwidth': bandwidth,
               'dwelltime': 1/bandwidth,
               'ResonantNucleus': nucleus}

    return FIDs, headers
