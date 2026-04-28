# synthetic.py - Create synthetic data from analytic functions
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from nifti_mrs.axes import Axes


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
                 nucleus='1H',
                 seed=None):

    inputs = locals()
    # Check noisecovariance is Ncoils x Ncoils
    ncoils = len(coilamps)
    noisecovariance = np.asarray(noisecovariance)
    if len(coilphase) != ncoils:
        raise ValueError('Length of coilamps and coilphase must match.')
    if noisecovariance.shape != (ncoils, ncoils):
        raise ValueError('noisecovariance must be ncoils x ncoils.')

    rng = np.random.default_rng(seed=seed)  # create a random generator with a fixed seed
    noise = rng.multivariate_normal(np.zeros((ncoils)),
                                    noisecovariance,
                                    points) + \
        1j * rng.multivariate_normal(np.zeros((ncoils)),
                                     noisecovariance,
                                     points)

    # Create Axes object
    axes = Axes(
        npoints=points,
        ResonantNucleus=nucleus,
        SpectrometerFrequency=centralfrequency,
        dwelltime=1/bandwidth)
    syntheticFID = np.zeros(points, dtype=np.complex128)
    # zero start the timeAxis and shift by begintime
    ttrue = axes.timeAxis - axes.timeAxis[0] + begintime

    if linewidth is not None:
        damping = np.asarray(linewidth) * np.pi

    for a, p, d, cs, gg in zip(amplitude, phase, damping, chemicalshift, g):
        # Lorentzian peak at chemicalShift
        syntheticFID += a * np.exp(1j * p) * np.exp(-d * (1 - gg + gg * ttrue) * ttrue
                                                    + 1j * 2 * np.pi
                                                    * cs
                                                    * axes.SpectrometerFrequency
                                                    * ttrue)

    FIDs = []
    for cDx, (camp, cphs) in enumerate(zip(coilamps, coilphase)):
        FIDs.append((camp * np.exp(1j * cphs) * syntheticFID) + noise[:, cDx])

    # TODO simplify the header and its usage throughout since Axes holds a lot of this info
    headers = {'noiseless': syntheticFID,
               'cov': noisecovariance,
               'taxis': ttrue - begintime,
               'faxis': axes.frequencyAxis,
               'ppmaxis': axes.ppmAxis,
               'inputopts': inputs,
               'centralFrequency': axes.SpectrometerFrequency,
               'bandwidth': axes.SpectralWidth,
               'dwelltime': axes.dwelltime,
               'ResonantNucleus': axes.ResonantNucleus
               }

    return FIDs, headers, axes
