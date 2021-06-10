'''FSL-MRS test script

Test generation of synthetic spectra

Copyright Will Clarke, University of Oxford, 2021'''

from fsl_mrs.utils import synthetic as syn
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.utils import mrs_io
from fsl_mrs.core import MRS
import numpy as np
from pathlib import Path

testsPath = Path(__file__).parent
basis_path = testsPath / 'testdata/fsl_mrs/steam_basis'
dynamic_model_path = testsPath / 'testdata/dynamic/fmrs_sin_model.py'


def test_noisecov():
    # Create positive semi-definite noise covariance
    inputnoisecov = np.random.random((2, 2))
    inputnoisecov = np.dot(inputnoisecov, inputnoisecov.T)

    testFID, hdr = syn.syntheticFID(coilamps=[1.0, 1.0],
                                    coilphase=[0.0, 0.0],
                                    noisecovariance=inputnoisecov,
                                    amplitude=[0.0, 0.0],
                                    points=32768)

    outcov = np.cov(np.asarray(testFID))

    # Noise cov is for both real and imag, so multiply by 2
    assert np.allclose(outcov, 2 * inputnoisecov, atol=1E-1)


def test_syntheticFID():
    testFID, hdr = syn.syntheticFID(noisecovariance=[[0.0]], points=16384)

    # Check FID is sum of lorentzian lineshapes
    # anlytical solution
    T2 = 1 / (hdr['inputopts']['damping'][0])
    M0 = hdr['inputopts']['amplitude'][0]
    f0 = hdr['inputopts']['centralfrequency'] * hdr['inputopts']['chemicalshift'][0]
    f1 = hdr['inputopts']['centralfrequency'] * hdr['inputopts']['chemicalshift'][1]
    f = hdr['faxis']
    spec = (M0 * T2) \
        / (1 + 4 * np.pi**2 * (f0 - f)**2 * T2**2) \
        + 1j * (2 * np.pi * M0 * (f0 - f) * T2**2) \
        / (1 + 4 * np.pi**2 * (f0 - f)**2 * T2**2)
    spec += (M0 * T2) \
        / (1 + 4 * np.pi**2 * (f1 - f)**2 * T2**2) \
        + 1j * (2 * np.pi * M0 * (f1 - f) * T2**2) \
        / (1 + 4 * np.pi**2 * (f1 - f)**2 * T2**2)

    # Can't quite get the scaling right here.
    testSpec = FIDToSpec(testFID[0])
    spec /= np.max(np.abs(spec))
    testSpec /= np.max(np.abs(testSpec))

    assert np.allclose(spec, FIDToSpec(testFID[0]), atol=1E-2, rtol=1E0)


def test_syntheticFromBasis():
    fid, header, _ = syn.syntheticFromBasisFile(str(basis_path),
                                                ignore=['Scyllo'],
                                                baseline=[0.0, 0.0],
                                                concentrations={'Mac': 2.0},
                                                coilamps=[1.0, 1.0],
                                                coilphase=[0.0, np.pi],
                                                noisecovariance=[[0.1, 0.0], [0.0, 0.1]])

    assert fid.shape == (2048, 2)


def test_syntheticFromBasis_baseline():

    fid, header, _ = syn.syntheticFromBasisFile(str(basis_path),
                                                baseline=[0.0, 0.0],
                                                concentrations={'Mac': 2.0},
                                                noisecovariance=[[0.0]])

    mrs = MRS(FID=fid, header=header)
    mrs.conj_FID = True

    fid, header, _ = syn.syntheticFromBasisFile(str(basis_path),
                                                baseline=[1.0, 1.0],
                                                concentrations={'Mac': 2.0},
                                                noisecovariance=[[0.0]])

    mrs2 = MRS(FID=fid, header=header)
    mrs2.conj_FID = True

    assert np.allclose(mrs2.get_spec(), mrs.get_spec() + complex(1.0, -1.0))


def test_synthetic_spectra_from_model():

    names = mrs_io.read_basis(str(basis_path)).names

    time_var = np.arange(0, 10)
    period = 10.0
    time_var = time_var / period

    camp = [0] * len(names)
    camp[names.index('NAA')] = 2
    defined_vals = {'c_0': 'conc',
                    'c_amp': camp,
                    'gamma': (20, 0),
                    'sigma': (20, 0),
                    'b_intercept': [1, 1, 1, 1, 1, 1],
                    'b_slope': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}

    p_noise = {'eps': (0, [0.01])}
    p_rel_noise = {'conc': (0, [0.05])}

    mrs_list, _, _ = syn.synthetic_spectra_from_model(
        str(dynamic_model_path),
        time_var,
        str(basis_path),
        ignore=None,
        metab_groups=['Mac'],
        ind_scaling=['Mac'],
        baseline_order=2,
        concentrations={'Mac': 15},
        defined_vals=defined_vals,
        bandwidth=6000,
        points=2048,
        baseline_ppm=(0, 5),
        param_noise=p_noise,
        param_rel_noise=p_rel_noise,
        coilamps=[1.0, 1.0],
        coilphase=[0.0, np.pi],
        noisecovariance=[[0.1, 0.0], [0.0, 0.1]])

    assert len(mrs_list) == 10
    assert len(mrs_list[0]) == 2
    assert mrs_list[0][0].numBasis == len(names)
    assert mrs_list[0][0].numPoints == 2048
