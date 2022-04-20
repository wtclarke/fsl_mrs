'''FSL-MRS test script

Tests for the individual proc functions.
These tests don't test theat the actual algorithms are doing the right thing,
simply that the script handles SVS data and MRSI data properly and that the
results from the command line program matches that of the underlying
algorithms in nifti_mrs_proc.py

Copyright Will Clarke, University of Oxford, 2021'''


import pytest
import os.path as op
from fsl_mrs.core.nifti_mrs import gen_new_nifti_mrs
from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.utils.mrs_io import read_FID
from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc
import numpy as np
import subprocess


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

    nmrs = gen_new_nifti_mrs(FID,
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

    nmrs = gen_new_nifti_mrs(FID,
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

    nmrs = gen_new_nifti_mrs(FID,
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

    nmrs = gen_new_nifti_mrs(FID,
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

    nmrs = gen_new_nifti_mrs(FID_comb,
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

    nmrs = gen_new_nifti_mrs(FID_comb,
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

    nmrs = gen_new_nifti_mrs(FID,
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
    assert np.allclose(data.data, svsdata.data)

    data = read_FID(mrsifile)
    assert data.shape == (2, 2, 2, 512, 3)
    assert np.allclose(data.data, mrsidata.data)

    svsfile, mrsifile, svsdata, mrsidata = splitdata(svs_data_uncomb,
                                                     mrsi_data_uncomb)

    data = read_FID(svsfile)
    assert data.shape == (1, 1, 1, 512, 4)
    assert np.allclose(data.data, svsdata.data)

    data = read_FID(mrsifile)
    assert data.shape == (2, 2, 2, 512, 4)
    assert np.allclose(data.data, mrsidata.data)

    # Uncombined and reps
    data = read_FID(svs_data_uncomb_reps[0])
    assert data.shape == (1, 1, 1, 512, 2, 2)
    assert np.allclose(data.data, svs_data_uncomb_reps[1].data)


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
    directRun = preproc.coilcombine(mrsidata)

    assert np.allclose(data[:], directRun[:])


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

    assert np.allclose(data.data, directRun.data, atol=1E-1, rtol=1E-1)


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
    # TODO: finish MRSI test


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
