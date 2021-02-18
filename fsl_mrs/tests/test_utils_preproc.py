'''FSL-MRS test script

Test utils.preproc functions

Copyright Will Clarke, University of Oxford, 2021'''

import fsl_mrs.utils.preproc as preproc
import fsl_mrs.utils.synthetic as syn
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.core import MRS
import numpy as np


# Test frequency shift by finding max on freq/ppm axis after a shift
def test_freqshift():
    testFID, testHdrs = syn.syntheticFID(amplitude=[0.0, 1.0])  # Single peak at 3 ppm
    # Shift to 0 ppm
    dt = 1 / testHdrs['inputopts']['bandwidth']
    shift = testHdrs['inputopts']['centralfrequency'] * -3.0
    shiftedFID = preproc.freqshift(testFID[0], dt, shift)

    maxindex = np.argmax(np.abs(FIDToSpec(shiftedFID)))
    freqOfMax = testHdrs['faxis'][maxindex]

    assert freqOfMax < 5 and freqOfMax > -5


# Test timeshift
def test_timeshift():
    # Create data with lots of points and some begin time delay
    testFID, testHdrs = syn.syntheticFID(begintime=-0.001, points=4096, noisecovariance=[[0.0]])
    testFID2, testHdrs2 = syn.syntheticFID(begintime=0.000, points=4096, noisecovariance=[[0.0]])

    # Reduce points and pad to remove first order phase
    shiftedFID, _ = preproc.timeshift(testFID[0],
                                      1 / testHdrs['inputopts']['bandwidth'],
                                      0.001,
                                      0.0,
                                      samples=4096)

    # assert shiftedFID.size == 2048
    assert np.allclose(shiftedFID, testFID2[0], atol=1E-1)


# Test combine_FIDs:
# Test mean by calculating mean of anti phase signals
# Test averaging using weights to zero signal
# Test coil combination against analytical Roemer eqn given known coil complex weights.
def test_combine_FIDs():
    testFIDs, testHdrs = syn.syntheticFID(noisecovariance=np.zeros((2, 2)), coilamps=[1.0, 1.0], coilphase=[0.0, np.pi])

    combfids = preproc.combine_FIDs(testFIDs, 'mean')

    assert np.allclose(combfids, 0)

    testFIDs, testHdrs = syn.syntheticFID(noisecovariance=np.zeros((4, 4)),
                                          coilamps=[1.0, 1.0, 1.0, 1.0],
                                          coilphase=[0.0, 0.0, 0.0, 0.0])

    weigths = [np.exp(1j * 0), np.exp(1j * np.pi / 2), np.exp(1j * np.pi), np.exp(1j * 3 * np.pi / 2)]
    combfids = preproc.combine_FIDs(testFIDs, 'weighted', weights=weigths)

    assert np.allclose(combfids, 0)

    # Generate high SNR data
    coilvar = 0.001
    noiseCov = [[coilvar, 0], [0, coilvar]]
    coilamps = 0.5 + np.random.rand(2) * 0.4
    coilphs = np.random.rand(2) * np.pi * 2

    testFIDs, testHdrs = syn.syntheticFID(noisecovariance=noiseCov,
                                          coilamps=coilamps,
                                          coilphase=coilphs,
                                          points=4096)

    invcovMat = coilvar * np.linalg.inv(noiseCov)
    analyticalRoemer = []
    testFIDs = np.asarray(testFIDs).T
    cmplxW = coilamps * np.exp(1j * coilphs)
    for f in testFIDs:
        tmp = (f @ invcovMat @ cmplxW.conj()) / np.sqrt(cmplxW.conj() @ invcovMat @ cmplxW)
        analyticalRoemer.append(tmp)
    analyticalRoemer = np.asarray(analyticalRoemer)

    combfid = preproc.combine_FIDs(testFIDs, 'svd', do_prewhiten=True)
    # Check only the first few points otherwise the relative tolarence has to be very high.
    assert np.allclose(np.abs(combfid[:200]), np.abs(analyticalRoemer[:200]), atol=1E-2, rtol=1E-1)


# Test the alignment by aligning based on two sets of offset peaks with
# different offsets and measuring combined peak height
def test_phase_freq_align():

    peak1Shift = np.random.rand(10) * 0.1
    peak1Phs = np.random.randn(10) * 2 * np.pi
    shiftedFIDs = []
    testHdrs = None
    for s, p in zip(peak1Shift, peak1Phs):
        testFIDs, testHdrs = syn.syntheticFID(amplitude=[1, 1],
                                              chemicalshift=[-2 + s, 3],
                                              phase=[p, 0.0],
                                              points=2048,
                                              noisecovariance=[[1E-1]])
        shiftedFIDs.append(testFIDs[0])

    # Align across shifted peak
    alignedFIDs, _, _ = preproc.phase_freq_align(shiftedFIDs,
                                                 testHdrs['bandwidth'],
                                                 testHdrs['centralFrequency'] * 1E6,
                                                 niter=2,
                                                 verbose=False,
                                                 ppmlim=(-2.2, -1.7),
                                                 shift=False)

    meanFID = preproc.combine_FIDs(alignedFIDs, 'mean')
    assert np.max(np.abs(FIDToSpec(meanFID))) > 0.09

    # Align across fixed peak
    alignedFIDs, _, _ = preproc.phase_freq_align(shiftedFIDs,
                                                 testHdrs['bandwidth'],
                                                 testHdrs['centralFrequency'] * 1E6,
                                                 niter=2,
                                                 verbose=False,
                                                 ppmlim=(2, 4),
                                                 shift=False)

    meanFID = preproc.combine_FIDs(alignedFIDs, 'mean')
    assert np.max(np.abs(FIDToSpec(meanFID))) > 0.09


def test_truncate():
    testFIDs, testHdrs = syn.syntheticFID()
    truncatedFID1 = preproc.truncate(testFIDs[0], 1, 'first')
    assert (testFIDs[0][1:] == truncatedFID1).all()
    truncatedFID1 = preproc.truncate(testFIDs[0], 1, 'last')
    assert (testFIDs[0][:-1] == truncatedFID1).all()


def test_pad():
    testFIDs, testHdrs = syn.syntheticFID()
    paddedFID1 = preproc.pad(testFIDs[0], 1, 'first')
    assert paddedFID1[0] == 0.0
    paddedFID1 = preproc.pad(testFIDs[0], 1, 'last')
    assert paddedFID1[-1] == 0.0


def test_apodize():
    mockFID = np.full((1024), 1.0)
    dt = 1 / 2000
    apodFID = preproc.apodize(mockFID, dt, [10])
    assert apodFID[-1] == np.exp(-dt * 1023 * 10)
    assert apodFID[0] == 1.0

    apodFID = preproc.apodize(mockFID, dt, [10, 0.01], 'l2g')

    t = dt * 1023
    Tl = 1 / 10
    Tg = 1 / 0.01
    assert apodFID[-1] == np.exp(t / Tl) * np.exp(t**2 / Tg**2)
    assert apodFID[0] == 1.0


# Test hlsvd by removing one peak from a two peak fid
def test_hlsvd():
    # low noise
    testFIDs, testHdrs = syn.syntheticFID(noisecovariance=[[1E-6]], amplitude=[1.0, 1.0], chemicalshift=[-2, 0])
    limits = [-2.5, -1.5]
    # (FID,dwelltime,centralFrequency,limits,limitUnits = 'ppm',numSingularValues=50)

    removedFID = preproc.hlsvd(testFIDs[0],
                               testHdrs['dwelltime'],
                               testHdrs['centralFrequency'],
                               limits,
                               limitUnits='ppm',
                               numSingularValues=20)

    onResFID = np.exp(-testHdrs['inputopts']['damping'][1] * testHdrs['taxis'])

    assert np.allclose(np.real(removedFID), np.real(onResFID), atol=1E-2, rtol=1E-2)


def test_pad_truncate():
    mockFID = np.full((1024), 1.0)
    testFID = preproc.pad(mockFID, 1, first_or_last='last')
    assert testFID.size == 1025
    assert testFID[-1] == 0.0

    mockFID = np.full((1024), 1.0)
    testFID = preproc.pad(mockFID, 2, first_or_last='first')
    assert testFID.size == 1026
    assert testFID[1] == 0.0

    mockFID = np.full((1024), 1.0)
    testFID = preproc.truncate(mockFID, 1, first_or_last='last')
    assert testFID.size == 1023

    mockFID = np.full((1024), 1.0)
    testFID = preproc.truncate(mockFID, 2, first_or_last='first')
    assert testFID.size == 1022


def test_phaseCorrect():
    testFIDs, testHdrs = syn.syntheticFID(amplitude=[1.0],
                                          chemicalshift=[0.0],
                                          phase=[np.pi / 2],
                                          noisecovariance=[[1E-5]])

    corrected, phs, pos = preproc.phaseCorrect(testFIDs[0],
                                               testHdrs['bandwidth'],
                                               testHdrs['centralFrequency'],
                                               ppmlim=(-0.5, 0.5),
                                               shift=False)
    assert np.isclose(phs, -np.pi / 2, atol=1E-2)


def test_add_subtract():
    mockFID = np.random.random(1024) + 1j * np.random.random(1024)
    testFID = preproc.add(mockFID.copy(), mockFID.copy())
    assert np.allclose(testFID, (mockFID * 2.0) / 2.0)  # Averaging op

    testFID = preproc.subtract(mockFID.copy(), mockFID.copy())
    assert np.allclose(testFID, np.zeros(1024))


def test_align_diff():
    shift0 = np.random.randn(10) * 0.15
    phs = np.random.randn(10) * 0.1 * np.pi
    shiftedFIDs0 = []
    shiftedFIDs1 = []
    testHdrs = None
    for s0, p in zip(shift0, phs):
        testFIDs, testHdrs = syn.syntheticFID(
            amplitude=[1, 1], chemicalshift=[-2 + s0, 3 + s0], phase=[p + np.pi, p],
            damping=[100, 100], points=2048, noisecovariance=[[1E-6]])
        shiftedFIDs0.append(testFIDs[0])

        testFIDs, testHdrs = syn.syntheticFID(
            amplitude=[1, 1], chemicalshift=[-2, 3], phase=[0, 0],
            damping=[100, 100], points=2048, noisecovariance=[[1E-6]])
        shiftedFIDs1.append(testFIDs[0])

    testFIDs0, _ = syn.syntheticFID(
        amplitude=[2, 1], chemicalshift=[-2, 3], phase=[np.pi, 0],
        damping=[100, 100], points=2048, noisecovariance=[[1E-6]])
    testFIDs1, _ = syn.syntheticFID(
        amplitude=[2, 1], chemicalshift=[-2, 3], phase=[0, 0],
        damping=[100, 100], points=2048, noisecovariance=[[1E-6]])

    tgt = testFIDs0[0] + testFIDs1[0]
    # mrs = MRS(FID=tgt, header=testHdrs)

    sub0_aa, sub1_aa, phi, eps = preproc.phase_freq_align_diff(shiftedFIDs0,
                                                               shiftedFIDs1,
                                                               testHdrs['bandwidth'],
                                                               testHdrs['centralFrequency'],
                                                               target=tgt,
                                                               shift=False,
                                                               ppmlim=(-5, 5))

    shiftInHz = shift0 * testHdrs['centralFrequency']
    assert np.allclose(eps, shiftInHz, atol=1E-0)
    assert np.allclose(phi, phs, atol=1E-0)


def test_shiftToRef():
    testFIDs, testHdrs = syn.syntheticFID(
        amplitude=[1, 0], chemicalshift=[-2.1, 0], phase=[0, 0], points=1024, noisecovariance=[[1E-3]])

    shiftFID, _ = preproc.shiftToRef(testFIDs[0],
                                     -2.0,
                                     testHdrs['bandwidth'],
                                     testHdrs['centralFrequency'],
                                     ppmlim=(-2.2, -2.0),
                                     shift=False)

    mrs = MRS(FID=shiftFID, header=testHdrs)

    maxindex = np.argmax(mrs.get_spec(shift=False))
    position = mrs.getAxes(axis='ppm')[maxindex]

    assert np.isclose(position, -2.0, atol=1E-1)
