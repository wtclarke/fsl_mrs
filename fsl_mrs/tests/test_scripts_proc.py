'''FSL-MRS test script

Tests for the individual proc functions.
These tests don't test theat the actual algorithms are doing the right thing,
simply that the script handles SVS data and MRSI data properly and that the
results from the command line program matches that of the underlying
algorithms in nifti_mrs_proc.py

Copyright Will Clarke, University of Oxford, 2021'''

import pytest
import os.path as op
import subprocess
import warnings
from pathlib import Path

import numpy as np

from fsl.data.image import Image

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.utils.mrs_io import read_FID
from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

testsPath = Path(__file__).parent
test_data = testsPath / 'testdata'


# construct some test data using synth
@pytest.fixture
def svs_data(tmp_path):
    reps = 3
    noiseconv = 0.1 * np.eye(reps)
    coilamps = np.ones(reps)
    coilphs = np.zeros(reps)
    FID, hdr = syntheticFID(noisecovariance=noiseconv,
                            coilamps=coilamps,
                            coilphase=coilphs,
                            points=512)

    FID = np.asarray(FID).T
    FID = FID.reshape((1, 1, 1) + FID.shape)

    nmrs = gen_nifti_mrs(
        FID,
        hdr['dwelltime'],
        hdr['centralFrequency'],
        dim_tags=['DIM_DYN', None, None])

    testname = 'svsdata.nii'
    testfile = op.join(tmp_path, testname)

    nmrs.save(testfile)

    return testfile, nmrs


@pytest.fixture
def mrsi_data(tmp_path):
    reps = 3
    noiseconv = 0.1 * np.eye(reps)
    coilamps = np.ones(reps)
    coilphs = np.zeros(reps)
    FID, hdr = syntheticFID(noisecovariance=noiseconv,
                            coilamps=coilamps,
                            coilphase=coilphs,
                            points=512)

    FID = np.asarray(FID).T
    FID = np.tile(FID, (2, 2, 2, 1, 1))

    nmrs = gen_nifti_mrs(
        FID,
        hdr['dwelltime'],
        hdr['centralFrequency'],
        dim_tags=['DIM_DYN', None, None])

    testname = 'mrsidata.nii'
    testfile = op.join(tmp_path, testname)
    nmrs.save(testfile)

    return testfile, nmrs


@pytest.fixture
def svs_data_uncomb(tmp_path):
    coils = 4
    noiseconv = 0.1 * np.eye(coils)
    coilamps = np.random.randn(coils)
    coilphs = np.random.random(coils) * 2 * np.pi
    FID, hdr = syntheticFID(noisecovariance=noiseconv,
                            coilamps=coilamps,
                            coilphase=coilphs,
                            points=512)

    FID = np.asarray(FID).T
    FID = FID.reshape((1, 1, 1) + FID.shape)

    nmrs = gen_nifti_mrs(
        FID,
        hdr['dwelltime'],
        hdr['centralFrequency'],
        dim_tags=['DIM_COIL', None, None])

    testname = 'svsdata_uncomb.nii'
    testfile = op.join(tmp_path, testname)
    nmrs.save(testfile)

    return testfile, nmrs


@pytest.fixture
def mrsi_data_uncomb(tmp_path):
    coils = 4
    noiseconv = 0.1 * np.eye(coils)
    coilamps = np.random.randn(coils)
    coilphs = np.random.random(coils) * 2 * np.pi
    FID, hdr = syntheticFID(noisecovariance=noiseconv,
                            coilamps=coilamps,
                            coilphase=coilphs,
                            points=512)

    FID = np.asarray(FID).T
    FID = np.tile(FID, (2, 2, 2, 1, 1))

    nmrs = gen_nifti_mrs(
        FID,
        hdr['dwelltime'],
        hdr['centralFrequency'],
        dim_tags=['DIM_COIL', None, None])

    testname = 'mrsidata_uncomb.nii'
    testfile = op.join(tmp_path, testname)
    nmrs.save(testfile)

    return testfile, nmrs


@pytest.fixture
def svs_data_diff(tmp_path):
    reps = 2
    noiseconv = 0.1 * np.eye(reps)
    coilamps = np.ones(reps)
    coilphs = np.zeros(reps)
    FID, hdr = syntheticFID(noisecovariance=noiseconv,
                            coilamps=coilamps,
                            coilphase=coilphs,
                            points=512)

    coilamps = -1 * np.ones(reps)
    coilphs = np.random.randn(reps)
    FID2, hdr = syntheticFID(noisecovariance=noiseconv,
                             coilamps=coilamps,
                             coilphase=coilphs,
                             points=512)

    FID = np.asarray(FID).T
    FID2 = np.asarray(FID2).T
    FID_comb = np.stack((FID, FID2), axis=2)
    FID_comb = FID_comb.reshape((1, 1, 1) + FID_comb.shape)

    nmrs = gen_nifti_mrs(
        FID_comb,
        hdr['dwelltime'],
        hdr['centralFrequency'],
        dim_tags=['DIM_DYN', 'DIM_EDIT', None])

    testname = 'svsdata_diff.nii'
    testfile = op.join(tmp_path, testname)
    nmrs.save(testfile)

    return testfile, nmrs


@pytest.fixture
def mrsi_data_diff(tmp_path):
    reps = 2
    noiseconv = 0.1 * np.eye(reps)
    coilamps = np.ones(reps)
    coilphs = np.zeros(reps)
    FID, hdr = syntheticFID(noisecovariance=noiseconv,
                            coilamps=coilamps,
                            coilphase=coilphs,
                            points=512)

    coilamps = -1 * np.ones(reps)
    coilphs = np.random.randn(reps)
    FID2, hdr = syntheticFID(noisecovariance=noiseconv,
                             coilamps=coilamps,
                             coilphase=coilphs,
                             points=512)

    FID = np.asarray(FID).T
    FID2 = np.asarray(FID2).T
    FID_comb = np.stack((FID, FID2), axis=2)
    FID_comb = np.tile(FID_comb, (2, 2, 2, 1, 1, 1))

    nmrs = gen_nifti_mrs(
        FID_comb,
        hdr['dwelltime'],
        hdr['centralFrequency'],
        dim_tags=['DIM_DYN', 'DIM_EDIT', None])

    testname = 'mrsidata_diff.nii'
    testfile = op.join(tmp_path, testname)
    nmrs.save(testfile)

    return testfile, nmrs


@pytest.fixture
def svs_data_uncomb_reps(tmp_path):
    coils = 2
    noiseconv = 0.1 * np.eye(coils)
    coilamps = np.random.randn(coils)
    coilphs = np.random.random(coils) * 2 * np.pi
    FID, hdr = syntheticFID(noisecovariance=noiseconv,
                            coilamps=coilamps,
                            coilphase=coilphs,
                            points=512)

    FID2, hdr = syntheticFID(noisecovariance=noiseconv,
                             coilamps=coilamps,
                             coilphase=coilphs,
                             points=512)

    FID = np.stack((np.asarray(FID).T, np.asarray(FID2).T), axis=2)
    FID = FID.reshape((1, 1, 1) + FID.shape)

    nmrs = gen_nifti_mrs(
        FID,
        hdr['dwelltime'],
        hdr['centralFrequency'],
        dim_tags=['DIM_COIL', 'DIM_DYN', None])

    testname = 'svsdata_uncomb_reps.nii'
    testfile = op.join(tmp_path, testname)
    nmrs.save(testfile)

    return testfile, nmrs


def splitdata(svs, mrsi):
    return svs[0], mrsi[0], svs[1], mrsi[1]


def test_filecreation(svs_data, mrsi_data, svs_data_uncomb, mrsi_data_uncomb, svs_data_uncomb_reps):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    data = read_FID(svsfile)
    assert data.shape == (1, 1, 1, 512, 3)
    assert np.allclose(data[:], svsdata[:])

    data = read_FID(mrsifile)
    assert data.shape == (2, 2, 2, 512, 3)
    assert np.allclose(data[:], mrsidata[:])

    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data_uncomb,
                                                     mrsi_data_uncomb)

    data = read_FID(svsfile)
    assert data.shape == (1, 1, 1, 512, 4)
    assert np.allclose(data[:], svsdata[:])

    data = read_FID(mrsifile)
    assert data.shape == (2, 2, 2, 512, 4)
    assert np.allclose(data[:], mrsidata[:])

    # Uncombined and reps
    data = read_FID(svs_data_uncomb_reps[0])
    assert data.shape == (1, 1, 1, 512, 2, 2)
    assert np.allclose(data[:], svs_data_uncomb_reps[1][:])


def test_coilcombine(svs_data_uncomb, mrsi_data_uncomb, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data_uncomb,
                                                     mrsi_data_uncomb)

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'coilcombine',
                           '--file', svsfile,
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        directRun = preproc.coilcombine(svsdata)

    assert np.allclose(data[:], directRun[:])

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'coilcombine',
                           '--file', mrsifile,
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        directRun = preproc.coilcombine(mrsidata)

    assert np.allclose(data[:], directRun[:])

    # Test with covariance matrix.
    np.savetxt(tmp_path / 'cov.txt', 0.1 * np.eye(4))

    subprocess.check_call(['fsl_mrs_proc',
                           'coilcombine',
                           '--file', svsfile,
                           '--covariance', str(tmp_path / 'cov.txt'),
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.coilcombine(svsdata, covariance=0.1 * np.eye(4))

    assert np.allclose(data[:], directRun[:])

    # Test with noise.
    from fsl_mrs.core.nifti_mrs import create_nmrs
    cov = 0.1 * np.eye(4) / 2
    mean = np.zeros((4,))
    rng = np.random.default_rng(seed=1)
    noise = rng.multivariate_normal(mean, cov, (20000,))\
        + 1j * rng.multivariate_normal(mean, cov, (20000,))

    create_nmrs.gen_nifti_mrs(
        noise.reshape((1, 1, 1, ) + noise.shape),
        1 / 1000,
        123.2,
        dim_tags=['DIM_COIL', None, None]
    ).save(tmp_path / 'noise.nii.gz')

    subprocess.check_call(['fsl_mrs_proc',
                           'coilcombine',
                           '--file', svsfile,
                           '--noise', str(tmp_path / 'noise.nii.gz'),
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.coilcombine(svsdata, noise=noise.T)

    assert np.allclose(data[:], directRun[:])


def test_coilcombine_datachecks(tmp_path):

    # Test unaveraged reference data
    met_raw = test_data / 'fsl_mrs_preproc' / 'metab_raw.nii.gz'
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _ = subprocess.run(
            ['fsl_mrs_proc',
             'coilcombine',
             '--file', str(met_raw),
             '--reference', str(met_raw),
             '--output', tmp_path,
             '--filename', 'tmp'],
            check=True,
            capture_output=True)
    assert exc_info.type is subprocess.CalledProcessError

    # Test already combined data
    met_proc = test_data / 'fsl_mrs' / 'metab.nii.gz'
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _ = subprocess.run(
            ['fsl_mrs_proc',
             'coilcombine',
             '--file', str(met_proc),
             '--output', tmp_path,
             '--filename', 'tmp'],
            check=True,
            capture_output=True)
    assert exc_info.type is subprocess.CalledProcessError


def test_average(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'average',
                           '--file', svsfile,
                           '--dim', 'DIM_DYN',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.average(svsdata, 'DIM_DYN')

    assert np.allclose(data[:], directRun[:])

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'average',
                           '--file', mrsifile,
                           '--dim', 'DIM_DYN',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.average(mrsidata, 'DIM_DYN')

    assert np.allclose(data[:], directRun[:])


def test_align(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run align on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'align',
                           '--dim', 'DIM_DYN',
                           '--file', svsfile,
                           '--ppm', '-10', '10',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.align(svsdata, 'DIM_DYN', ppmlim=(-10, 10))

    assert np.allclose(data[:], directRun[:])

    subprocess.check_call(['fsl_mrs_proc',
                           'align',
                           '--dim', 'DIM_DYN',
                           '--file', mrsifile,
                           '--ppm', '-10', '10',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.align(mrsidata, 'DIM_DYN', ppmlim=(-10, 10))

    assert np.allclose(data[:], directRun[:], atol=1E-1, rtol=1E-1)


def test_align_all(svs_data_uncomb_reps, tmp_path):
    svsfile, svsdata = svs_data_uncomb_reps[0], svs_data_uncomb_reps[1]

    # Run align on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'align',
                           '--dim', 'all',
                           '--file', svsfile,
                           '--ppm', '-10', '10',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.align(svsdata, 'all', ppmlim=(-10, 10))

    assert np.allclose(data[:], directRun[:])


def test_ecc(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'ecc',
                           '--file', svsfile,
                           '--reference', svsfile,
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.ecc(svsdata, svsdata)

    assert np.allclose(data[:], directRun[:])

    # Run coil ecc on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'ecc',
                           '--file', mrsifile,
                           '--reference', mrsifile,
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.ecc(mrsidata, mrsidata)

    assert np.allclose(data[:], directRun[:])


def test_remove(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run remove on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'remove',
                           '--file', svsfile,
                           '--ppm', '-10', '10',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.remove_peaks(svsdata, (-10, 10))

    assert np.allclose(data[:], directRun[:])

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'remove', '--file',
                           mrsifile,
                           '--ppm', '-10', '10',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.remove_peaks(mrsidata, (-10, 10))

    assert np.allclose(data[:], directRun[:])


def test_model(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run model on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'model',
                           '--file', svsfile,
                           '--ppm', '-10', '10',
                           '--components', '5',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.hlsvd_model_peaks(svsdata, (-10, 10), components=5)

    assert np.allclose(data[:], directRun[:])

    # Run model on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'model',
                           '--file', mrsifile,
                           '--ppm', '-10', '10',
                           '--components', '5',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.hlsvd_model_peaks(mrsidata, (-10, 10), components=5)

    assert np.allclose(data[:], directRun[:])


def test_align_diff(svs_data_diff, mrsi_data_diff, tmp_path):
    svsfile, svsdata = svs_data_diff[0], svs_data_diff[1]

    # Run alignment via commandline
    subprocess.check_call(['fsl_mrs_proc',
                           'align-diff',
                           '--file', svsfile,
                           '--ppm', '-10', '10',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.aligndiff(svsdata,
                                  'DIM_DYN',
                                  'DIM_EDIT',
                                  'add',
                                  ppmlim=(-10, 10))

    assert np.allclose(data[:], directRun[:])
    # TODO: finish MRSI test


def test_fshift(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    subprocess.check_call(['fsl_mrs_proc',
                           'fshift',
                           '--file', svsfile,
                           '--shiftppm', '1.0',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.fshift(svsdata, 1.0 * 123.2)

    assert np.allclose(data[:], directRun[:])

    subprocess.check_call(['fsl_mrs_proc',
                           'fshift',
                           '--file', svsfile,
                           '--shiftRef',
                           '--ppm', '-5.0', '5.0',
                           '--target', '4.0',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.shift_to_reference(svsdata, 4.0, (-5.0, 5.0))

    assert np.allclose(data[:], directRun[:])

    # MRSI test with single shift
    subprocess.check_call(['fsl_mrs_proc',
                           'fshift',
                           '--file', mrsifile,
                           '--shifthz', '10.0',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.fshift(mrsidata, 10.0)

    assert np.allclose(data[:], directRun[:])

    # MRSI test with multiple shift
    shifts = np.ones(mrsidata.shape[:3] + mrsidata.shape[4:])
    from fsl.data.image import Image
    Image(shifts).save(tmp_path / 'multi_shift.nii.gz')
    subprocess.check_call(['fsl_mrs_proc',
                           'fshift',
                           '--file', mrsifile,
                           '--shifthz', str(tmp_path / 'multi_shift.nii.gz'),
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.fshift(mrsidata, shifts)

    assert np.allclose(data[:], directRun[:])


def test_conj(svs_data, mrsi_data, tmp_path):
    """ Test fsl_mrs_proc conj"""
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run remove on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'conj',
                           '--file', svsfile,
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.conjugate(svsdata)

    assert np.allclose(data[:], directRun[:])

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'conj',
                           '--file', mrsifile,
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.conjugate(mrsidata)

    assert np.allclose(data[:], directRun[:])


def test_fixed_phase(svs_data, mrsi_data, tmp_path):
    """ Test fsl_mrs_proc fixed_phase"""
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run remove on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'fixed_phase',
                           '--file', svsfile,
                           '--p0', '90',
                           '--p1', '0.001',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.apply_fixed_phase(svsdata, 90, 0.001)

    assert np.allclose(data[:], directRun[:])

    # Run with linphase
    subprocess.check_call(['fsl_mrs_proc',
                           'fixed_phase',
                           '--file', svsfile,
                           '--p0', '90',
                           '--p1', '0.001',
                           '--p1_type', 'linphase',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.apply_fixed_phase(svsdata, 90, 0.001, p1_type='linphase')

    assert np.allclose(data[:], directRun[:])

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'fixed_phase',
                           '--file', mrsifile,
                           '--p0', '90',
                           '--p1', '0.001',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.apply_fixed_phase(mrsidata, 90, 0.001)

    assert np.allclose(data[:], directRun[:])


def test_apodize(svs_data, mrsi_data, tmp_path):
    """ Test fsl_mrs_proc apodize"""
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    # Run remove on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'apodize',
                           '--file', svsfile,
                           '--filter', 'exp',
                           '--amount', '10',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.apodize(svsdata, (10,))

    assert np.allclose(data[:], directRun[:])

    # Run coil combination on both sets of data using the command line
    subprocess.check_call(['fsl_mrs_proc',
                           'apodize',
                           '--file', mrsifile,
                           '--filter', 'l2g',
                           '--amount', '10', '1',
                           '--output', tmp_path,
                           '--filename', 'tmp'])

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun = preproc.apodize(mrsidata, (10, 1), filter='l2g')

    assert np.allclose(data[:], directRun[:])


def test_unlike(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, _ = splitdata(svs_data, mrsi_data)

    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            ['fsl_mrs_proc',
             'unlike',
             '--file', mrsifile,
             '--output', tmp_path,
             '--filename', 'tmp'],
            check=True,
            capture_output=True)

    _ = subprocess.run(
        ['fsl_mrs_proc',
         'unlike',
         '--file', svsfile,
         '--output', tmp_path,
         '--sd', '1.0',
         '--iter', '3',
         '--outputbad',
         '-r',
         '--verbose',
         '--filename', 'tmp'],
        check=True,
        capture_output=True)

    assert (tmp_path / 'tmp_FAIL.nii.gz').is_file()
    assert len(list(tmp_path.glob('report*.html'))) == 1

    # Load result for comparison
    data = read_FID(op.join(tmp_path, 'tmp.nii.gz'))

    # Run directly
    directRun, _ = preproc.remove_unlike(svsdata, sdlimit=1.0, niter=3)

    assert np.allclose(data[:], directRun[:])

    # Check results if no filename provided
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'unlike',
         '--file', svsfile,
         '--output', tmp_path / 'nofname',
         '--sd', '1.0',
         '--iter', '3',
         '--outputbad',
         '-r',
         '--verbose'],
        check=True,
        capture_output=True)

    assert (tmp_path / 'nofname' / 'svsdata.nii').is_file()
    assert (tmp_path / 'nofname' / 'svsdata_FAIL.nii.gz').is_file()

    # Check that the process doesn't crash if no bad transients
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'unlike',
         '--file', svsfile,
         '--output', tmp_path,
         '--sd', '10.0',
         '--iter', '1',
         '--outputbad',
         '--filename', 'tmp'],
        check=True,
        capture_output=True)


def test_tshift(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'tshift',
         '--file', svsfile,
         '--output', tmp_path,
         '--tshiftStart', '10.0',
         '--tshiftEnd', '10.0',
         '--samples', '1024',
         '--filename', 'tmp0',
         '-r',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp0.nii.gz'))
    directRun = preproc.tshift(svsdata, tshiftStart=10.0, tshiftEnd=10.0, samples=1024)
    assert np.allclose(data[:], directRun[:])
    assert len(list(tmp_path.glob('report*.html'))) == 1

    # Alternative inputs
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'tshift',
         '--file', svsfile,
         '--output', tmp_path,
         '--tshiftStart', '-10.0',
         '--tshiftEnd', '-10.0',
         '--samples', '512',
         '--filename', 'tmp1',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp1.nii.gz'))
    directRun = preproc.tshift(svsdata, tshiftStart=-10.0, tshiftEnd=-10.0, samples=512)
    assert np.allclose(data[:], directRun[:])


def test_truncate(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)
    # Add points
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'truncate',
         '--file', svsfile,
         '--output', tmp_path,
         '--points', '10',
         '--pos', 'first',
         '--filename', 'tmp0',
         '-r',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp0.nii.gz'))
    directRun = preproc.truncate_or_pad(svsdata, 10, position='first')
    assert np.allclose(data[:], directRun[:])
    assert len(list(tmp_path.glob('report*.html'))) == 1

    # Remove points
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'truncate',
         '--file', svsfile,
         '--output', tmp_path,
         '--points', '-10',
         '--pos', 'first',
         '--filename', 'tmp1',
         '-r',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp1.nii.gz'))
    directRun = preproc.truncate_or_pad(svsdata, -10, position='first')
    assert np.allclose(data[:], directRun[:])

    # Remove points at end
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'truncate',
         '--file', svsfile,
         '--output', tmp_path,
         '--points', '-10',
         '--pos', 'last',
         '--filename', 'tmp2',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp2.nii.gz'))
    directRun = preproc.truncate_or_pad(svsdata, -10, position='last')
    assert np.allclose(data[:], directRun[:])


def test_phase(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'phase',
         '--file', svsfile,
         '--output', tmp_path,
         '--ppm', '0', '4',
         '--filename', 'tmp0',
         '-r',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp0.nii.gz'))
    directRun = preproc.phase_correct(svsdata, (0, 4))
    assert np.allclose(data[:], directRun[:])
    assert len(list(tmp_path.glob('report*.html'))) == 1

    # With hlsvd
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'phase',
         '--file', svsfile,
         '--output', tmp_path,
         '--ppm', '0', '4',
         '--hlsvd',
         '--filename', 'tmp1',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp1.nii.gz'))
    directRun = preproc.phase_correct(svsdata, (0, 4), hlsvd=True)
    assert np.allclose(data[:], directRun[:])

    # With avg
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'phase',
         '--file', svsfile,
         '--output', tmp_path,
         '--ppm', '0', '4',
         '--use_avg',
         '--filename', 'tmp2',
         '--verbose'],
        check=True,
        capture_output=True)

    # Load result for comparison against direct run
    data = read_FID(op.join(tmp_path, 'tmp2.nii.gz'))
    directRun = preproc.phase_correct(svsdata, (0, 4), use_avg=True)
    assert np.allclose(data[:], directRun[:])


def test_add(svs_data_uncomb_reps, tmp_path):
    svsfile, svsdata = svs_data_uncomb_reps
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'add',
         '--file', svsfile,
         '--reference', svsfile,
         '--output', tmp_path,
         '--filename', 'tmp0',
         '--verbose'],
        check=True,
        capture_output=True)

    data = read_FID(op.join(tmp_path, 'tmp0.nii.gz'))
    directRun = preproc.add(svsdata, svsdata)
    assert np.allclose(data[:], directRun[:])

    # Single file add across dim
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'add',
         '--file', svsfile,
         '--dim', 'DIM_DYN',
         '--output', tmp_path,
         '--filename', 'tmp1',
         '-r',
         '--verbose'],
        check=True,
        capture_output=True)

    data = read_FID(op.join(tmp_path, 'tmp1.nii.gz'))
    directRun = preproc.add(svsdata, dim='DIM_DYN')
    assert np.allclose(data[:], directRun[:])
    assert len(list(tmp_path.glob('report*.html'))) == 1


def test_subtract(svs_data_uncomb_reps, tmp_path):
    svsfile, svsdata = svs_data_uncomb_reps
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'subtract',
         '--file', svsfile,
         '--reference', svsfile,
         '--output', tmp_path,
         '--filename', 'tmp0',
         '--verbose'],
        check=True,
        capture_output=True)

    data = read_FID(op.join(tmp_path, 'tmp0.nii.gz'))
    directRun = preproc.subtract(svsdata, svsdata)
    assert np.allclose(data[:], directRun[:])

    # Single file subtract across dim
    _ = subprocess.run(
        ['fsl_mrs_proc',
         'subtract',
         '--file', svsfile,
         '--dim', 'DIM_DYN',
         '--output', tmp_path,
         '--filename', 'tmp1',
         '-r',
         '--verbose'],
        check=True,
        capture_output=True)

    data = read_FID(op.join(tmp_path, 'tmp1.nii.gz'))
    directRun = preproc.subtract(svsdata, dim='DIM_DYN')
    assert np.allclose(data[:], directRun[:])
    assert len(list(tmp_path.glob('report*.html'))) == 1


def test_mrsi_align(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)

    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            ['fsl_mrs_proc',
             'mrsi-align',
             '--file', svsfile,
             '--output', tmp_path,
             '--filename', 'tmp'],
            check=True,
            capture_output=True)

    _ = subprocess.run(
        ['fsl_mrs_proc',
            'mrsi-align',
            '--file', mrsifile,
            '--freq-align',
            '--phase-correct',
            '--save-params',
            '--zpad', '1',
            '--ppm', '0.2', '4.0',
            '--output', tmp_path,
            '--filename', 'tmp'],
        check=True,
        capture_output=True)

    assert (tmp_path / 'tmp.nii.gz').exists()
    assert (tmp_path / 'tmp_shifts_hz.nii.gz').exists()
    assert (tmp_path / 'tmp_phase_deg.nii.gz').exists()

    proc_data = read_FID(tmp_path / 'tmp.nii.gz')
    assert proc_data.shape == mrsidata.shape

    shifts = Image(tmp_path / 'tmp_shifts_hz.nii.gz')
    phs = Image(tmp_path / 'tmp_phase_deg.nii.gz')

    assert shifts.shape == (mrsidata.shape[:3] + mrsidata.shape[4:])
    assert phs.shape == (mrsidata.shape[:3] + mrsidata.shape[4:])


def test_mrsi_lipid(svs_data, mrsi_data, tmp_path):
    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data, mrsi_data)
    mask = Image(np.ones(mrsidata.shape[:3], dtype=np.int16))
    mask.save(tmp_path / 'mask.nii.gz')

    with pytest.raises(subprocess.CalledProcessError):
        _ = subprocess.run(
            ['fsl_mrs_proc',
             'mrsi-lipid',
             '--file', svsfile,
             '--output', tmp_path,
             '--filename', 'tmp'],
            check=True,
            capture_output=True)

    _ = subprocess.run(
        ['mrs_tools',
         'split',
         '--file', mrsifile,
         '--dim', 'DIM_DYN',
         '--index', '0',
         '--output', tmp_path,
         '--filename', 'split'])

    _ = subprocess.run(
        ['fsl_mrs_proc',
            'mrsi-lipid',
            '--file', tmp_path / 'split_low.nii.gz',
            '--mask', tmp_path / 'mask.nii.gz',
            '--beta', '0.0001',
            '--output', tmp_path,
            '--filename', 'tmp'],
        check=True,
        capture_output=True)

    assert (tmp_path / 'tmp.nii.gz').exists()
