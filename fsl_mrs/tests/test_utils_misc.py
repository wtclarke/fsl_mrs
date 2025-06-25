'''FSL-MRS test script

Test functions that appear in utils.misc module

Copyright Will Clarke, University of Oxford, 2021'''
import pytest
from pathlib import Path

import numpy as np

from fsl_mrs.utils.mrs_io.main import read_FID
from fsl_mrs.utils import misc
from fsl_mrs.utils import synthetic as synth

testsPath = Path(__file__).parent
basis_path = testsPath / 'testdata/fsl_mrs/steam_basis'
data_path = testsPath / 'testdata/fsl_mrs/metab.nii.gz'


def test_ppm2hz_hz2ppm():
    cf = 300E6
    ppm = 1.0
    shift = 4.65
    assert misc.ppm2hz(cf, ppm, shift=False) == (1.0 * 300)
    assert misc.ppm2hz(cf, ppm, shift=True) == ((1.0 - shift) * 300)

    hz = 300
    assert misc.hz2ppm(cf, hz, shift=False) == 1.0
    assert misc.hz2ppm(cf, hz, shift=True) == (1.0 + shift)


def test_FIDToSpec_SpecToFID():
    testFID, hdr = synth.syntheticFID(amplitude=[10], chemicalshift=[0], phase=[0], damping=[20])

    # SVS case
    spec = misc.FIDToSpec(testFID[0])
    reformedFID = misc.SpecToFID(spec)
    assert np.allclose(reformedFID, testFID)

    testMRSI = np.tile(testFID, (4, 4, 4, 1)).T
    testspec = misc.FIDToSpec(testMRSI)
    assert np.argmax(np.abs(testspec[:, 2, 2, 2])) == 1024

    testspec = misc.FIDToSpec(testMRSI.T, axis=3)
    assert np.argmax(np.abs(testspec[2, 2, 2, :])) == 1024

    reformedFID = misc.SpecToFID(testspec, axis=3)
    assert np.allclose(reformedFID, testMRSI.T)

    reformedFID = misc.SpecToFID(testspec.T)
    assert np.allclose(reformedFID, testMRSI)

    # Odd number of points - guard against fftshift/ifftshift errors
    testFID, hdr = synth.syntheticFID(amplitude=[1], chemicalshift=[0], phase=[0], damping=[20], points=1025)
    assert np.allclose(misc.SpecToFID(misc.FIDToSpec(testFID[0])), testFID)


def test_checkCFUnits():
    assert misc.checkCFUnits(10, units='Hz') == 10E6
    assert misc.checkCFUnits(10E6, units='Hz') == 10E6

    assert misc.checkCFUnits(10, units='MHz') == 10
    assert misc.checkCFUnits(10E6, units='MHz') == 10


def test_limit_to_range():
    axis = np.arange(0, 10)
    assert misc.limit_to_range(axis, (1, 5)) == (1, 5)
    assert misc.limit_to_range(axis, None) == (0, 10)

    axis = np.arange(-10.5, 10.5)
    assert misc.limit_to_range(axis, (1, 5)) == (11, 15)
    assert misc.limit_to_range(axis, None) == (0, 21)


def test_parse_metab_groups():

    mrs = read_FID(str(data_path)).mrs(basis_file=str(basis_path))

    # combine all
    assert misc.parse_metab_groups(mrs, 'combine_all')\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # separate all
    assert misc.parse_metab_groups(mrs, 'separate_all')\
        == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # One metabolite
    assert misc.parse_metab_groups(mrs, 'Mac')\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # Two metabolites
    assert misc.parse_metab_groups(mrs, ['Mac', 'NAA'])\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]

    # Combine two
    assert misc.parse_metab_groups(mrs, 'NAA+NAAG')\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    # Combine three
    assert misc.parse_metab_groups(mrs, ['NAA+NAAG+Cr'])\
        == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    # Combine three and separate
    assert misc.parse_metab_groups(mrs, ['Mac', 'NAA+NAAG+Cr'])\
        == [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0]

    # Single integer
    assert misc.parse_metab_groups(mrs, 1)\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # List of integers
    assert misc.parse_metab_groups(mrs, [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])\
        == [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_interpolation():
    target_bw = 2000
    target_n = 1024
    fid_full, hdr_full = synth.syntheticFID(bandwidth=8000, points=8192, noisecovariance=[[0.0]])
    fid_reduced, hdr_reduced = synth.syntheticFID(bandwidth=target_bw, points=target_n, noisecovariance=[[0.0]])

    interp_lin = misc.ts_to_ts(fid_full[0], 1 / 8000, 1 / target_bw, target_n)
    interp_ft = misc.ts_to_ts_ft(fid_full[0], 1 / 8000, 1 / target_bw, target_n)

    # import matplotlib.pyplot as plt

    # plt.plot(hdr_full['taxis'], np.squeeze(np.real(fid_full)), '-x')
    # plt.plot(hdr_reduced['taxis'], np.squeeze(np.real(fid_reduced)), '--x')
    # plt.plot(hdr_reduced['taxis'], np.squeeze(np.real(interp_lin)), ':x')
    # plt.plot(hdr_reduced['taxis'], np.squeeze(np.real(interp_ft)), ':x')
    # plt.xlim([-0.001, 0.1])
    # plt.show()

    # fig = plt.figure(figsize=(15,6))
    # plt.plot(hdr_full['faxis'], np.real(plot.FID2Spec(np.asarray(np.squeeze(fid_full)))), '-')
    # plt.plot(hdr_reduced['faxis'], np.real(plot.FID2Spec(np.asarray(np.squeeze(fid_reduced)))), '-')
    # plt.plot(hdr_reduced['faxis'], np.squeeze(np.real(plot.FID2Spec(np.asarray(interp_lin)))), ':')
    # plt.plot(hdr_reduced['faxis'], np.squeeze(np.real(plot.FID2Spec(np.asarray(interp_ft)))), ':')
    # plt.xlim([-500,0])
    # plt.show()

    assert np.allclose(interp_lin, fid_reduced[0])

    # We know the first few points are corrupted in the fft version, but that will appear at edge
    # of the spectrum
    assert np.allclose(interp_ft[10:-10], np.asarray(fid_reduced[0])[10:-10], atol=1E-1)


def test_link_creation(tmp_path):
    misc.create_rel_symlink(data_path, tmp_path, 'test1')

    assert (tmp_path / 'test1.nii.gz').is_symlink()
    assert read_FID(tmp_path / 'test1.nii.gz').shape == (1, 1, 1, 4095)

    misc.create_rel_symlink(data_path, tmp_path, 'test2', match_ext=False)
    assert (tmp_path / 'test2').is_symlink()

    dummy = Path(tmp_path / 'dummy.nii')
    dummy.touch()
    misc.create_rel_symlink(dummy, tmp_path, 'test3')
    assert (tmp_path / 'test3.nii').is_symlink()

    dummy_dir = Path(tmp_path / 'dummy_dir')
    dummy_dir.mkdir()
    misc.create_rel_symlink(dummy_dir, tmp_path, 'test4')
    assert (tmp_path / 'test4').is_symlink()


def test_create_peak():
    dwell = 1 / 2000
    t_axis = np.arange(0, dwell * 256, dwell)

    with pytest.raises(
            ValueError,
            match='ppm and amp should have the same length, currently'):
        misc.create_peak(
            t_axis,
            120,
            [0, 1, 2],
            [1.0, ],
            gamma=10)
    with pytest.raises(
            ValueError,
            match='ppm and phase should have the same length, currently'):
        misc.create_peak(
            t_axis,
            120,
            [0, 1, 2],
            [1, 1, 1],
            phase=[0, 0])

    zero_ppm = misc.create_peak(
        t_axis,
        120,
        0.0,
        1.0,
        gamma=10)

    zero_ppm_neg = misc.create_peak(
        t_axis,
        120,
        0.0,
        1.0,
        gamma=10,
        phase=np.pi)

    assert np.allclose(zero_ppm, -zero_ppm_neg)

    one_ppm = misc.create_peak(
        t_axis,
        120,
        1.0,
        1.0,
        gamma=10)

    both_peaks = misc.create_peak(
        t_axis,
        120,
        [0.0, 1.0],
        [1.0, 1.0],
        gamma=10)

    assert np.allclose(one_ppm + zero_ppm, both_peaks)


def test_detect_conjugation():
    FIDs, headers = synth.syntheticFID(
        chemicalshift=[-2, -3])
    FID = FIDs[0]
    FIDs = np.stack((FIDs[0], FIDs[0], FIDs[0].conj()))

    assert not misc.detect_conjugation(
        FID,
        headers['ppmaxis'],
        (0, -4))
    assert misc.detect_conjugation(
        FID.conj(),
        headers['ppmaxis'],
        (0, -4))

    assert misc.detect_conjugation(
        FIDs.conj(),
        headers['ppmaxis'],
        (0, -4))

    # Test wrong orientation of data raises error
    with pytest.raises(ValueError):
        misc.detect_conjugation(
            FIDs.T,
            headers['ppmaxis'],
            (0, -4))


def test_check_nucleus_format():
    for correct_value in ["1H", "31P", "23Na", "129Xe"]:
        assert misc.check_nucleus_format(correct_value)

    for incorrect_value in ["H1", "31p", "23NA", "1229Xe", "129Xee"]:
        print(incorrect_value)
        assert not misc.check_nucleus_format(incorrect_value)
