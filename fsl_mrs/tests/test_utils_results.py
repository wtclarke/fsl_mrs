'''FSL-MRS test script

Test features of the results class

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.core import MRS
from fsl_mrs.utils.fitting import fit_FSLModel
from pytest import fixture
import numpy as np


# Set up some synthetic data to use
@fixture(scope='module')
def data():
    noiseCov = 0.001
    amplitude = np.asarray([0.5, 0.5, 1.0]) * 10
    chemshift = np.asarray([3.0, 3.05, 2.0]) - 4.65
    lw = [10, 10, 10]
    phases = [0, 0, 0]
    g = [0, 0, 0]
    basisNames = ['Cr', 'PCr', 'NAA']
    begintime = 0.00005

    basisFIDs = []
    basisHdr = None
    for idx, _ in enumerate(amplitude):
        tmp, basisHdr = syntheticFID(noisecovariance=[[0.0]],
                                     chemicalshift=[chemshift[idx] + 0.1],
                                     amplitude=[1.0],
                                     linewidth=[lw[idx] / 5],
                                     phase=[phases[idx]],
                                     g=[g[idx]],
                                     begintime=0)
        basisFIDs.append(tmp[0])
    basisFIDs = np.asarray(basisFIDs)

    synFID, synHdr = syntheticFID(noisecovariance=[[noiseCov]],
                                  chemicalshift=chemshift,
                                  amplitude=amplitude,
                                  linewidth=lw,
                                  phase=phases,
                                  g=g,
                                  begintime=begintime)

    synMRS = MRS(FID=synFID[0],
                 header=synHdr,
                 basis=basisFIDs,
                 basis_hdr=basisHdr,
                 names=basisNames)

    metab_groups = [0] * synMRS.numBasis
    Fitargs = {'ppmlim': [0.2, 4.2],
               'method': 'MH',
               'baseline_order': -1,
               'metab_groups': metab_groups,
               'MHSamples': 100,
               'disable_mh_priors': True}

    res = fit_FSLModel(synMRS, **Fitargs)

    return res, amplitude


def test_peakcombination(data):

    res = data[0]
    amplitudes = data[1]

    res.combine([['Cr', 'PCr']])

    fittedconcs = res.getConc()
    fittedRelconcs = res.getConc(scaling='internal')

    amplitudes = np.append(amplitudes, amplitudes[0] + amplitudes[1])

    assert 'Cr+PCr' in res.metabs
    assert np.allclose(fittedconcs, amplitudes, atol=2E-1)
    assert np.allclose(fittedRelconcs,
                       amplitudes / (amplitudes[0] + amplitudes[1]),
                       atol=2E-1)


def test_units(data):
    res = data[0]

    # Phase
    p0, p1 = res.getPhaseParams(phi0='degrees', phi1='seconds')
    assert np.isclose(p0, 0.0, atol=1E-1)
    assert np.isclose(p1, 0.00005, atol=3E-5)

    # Shift
    shift = res.getShiftParams(units='ppm')
    shift_hz = res.getShiftParams(units='Hz')
    assert np.isclose(shift, 0.1, atol=1E-2)
    assert np.isclose(shift_hz, 0.1 * 123.0, atol=1E-1)

    # Linewidth
    lw = res.getLineShapeParams(units='Hz')[0]
    lw_ppm = res.getLineShapeParams(units='ppm')[0]
    assert np.isclose(lw, 8.0, atol=1E-1)  # 10-2
    assert np.isclose(lw_ppm, 8.0 / 123.0, atol=1E-1)


def test_qcOutput(data):
    res = data[0]
    SNR, FWHM = res.getQCParams()

    assert np.allclose(FWHM, 10.0, atol=1E0)
    assert SNR.size == 3
