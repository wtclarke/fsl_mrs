'''FSL-MRS tests

Test utils.preproc.combine functions and
the implementation in nifti_mrs_proc in depth.

Copyright Will Clarke, University of Oxford, 2023'''
import warnings

import numpy as np
import pytest

from fsl_mrs.utils.preproc import combine
from fsl_mrs.utils import synthetic as syn
from fsl_mrs.utils.preproc.nifti_mrs_proc import coilcombine
from fsl_mrs.core.nifti_mrs import create_nmrs


def test_covar_est(capsys):

    rng = np.random.default_rng(seed=1)

    random_cov = rng.standard_normal((4, 4))
    random_cov = random_cov.T @ random_cov

    # Test case with sufficient samples (and repetitions)
    mean = np.zeros((4,))
    shape = (1000, 1000)
    noise = rng.multivariate_normal(mean, random_cov / 2, shape)\
        + 1j * rng.multivariate_normal(mean, random_cov / 2, shape)

    est_cov = combine.estimate_noise_cov(noise, prop=0.1)

    assert np.allclose(est_cov, random_cov, atol=5E-2)

    # Test case with sufficient samples (no repetitions)
    mean = np.zeros((4,))
    shape = (int(1E6),)
    noise = rng.multivariate_normal(mean, random_cov / 2, shape)\
        + 1j * rng.multivariate_normal(mean, random_cov / 2, shape)

    est_cov = combine.estimate_noise_cov(noise, prop=0.1)

    assert np.allclose(est_cov, random_cov, atol=5E-2)

    # Test case with low samples
    mean = np.zeros((4,))
    shape = (2, 1000)
    noise = rng.multivariate_normal(mean, random_cov / 2, shape)\
        + 1j * rng.multivariate_normal(mean, random_cov / 2, shape)

    warn_str = \
        'You may not have enough samples to accurately estimate the noise covariance, '\
        '10^5 samples recommended.\n'
    _ = combine.estimate_noise_cov(noise, prop=0.1)
    captured = capsys.readouterr()
    assert captured.out == warn_str

    # Test case with insufficient samples
    mean = np.zeros((4,))
    shape = (100)
    noise = rng.multivariate_normal(mean, random_cov / 2, shape)\
        + 1j * rng.multivariate_normal(mean, random_cov / 2, shape)

    with pytest.raises(combine.CovarianceEstimationError):
        _ = combine.estimate_noise_cov(noise, prop=0.1)


def test_prewhiten():
    rng = np.random.default_rng(seed=1)
    random_cov = rng.standard_normal((4, 4))
    random_cov = random_cov.T @ random_cov

    mean = np.zeros((4,))
    shape = (1000, 1000)
    noise = rng.multivariate_normal(mean, random_cov / 2, shape)\
        + 1j * rng.multivariate_normal(mean, random_cov / 2, shape)

    est_cov = combine.estimate_noise_cov(noise, prop=0.1)

    pwdata, W, C = combine.prewhiten(noise, C=random_cov)
    est_pw_cov = combine.estimate_noise_cov(pwdata, prop=0.1)
    assert np.allclose(est_pw_cov, np.eye(4), atol=1E-1)

    pwdata, W, C = combine.prewhiten(noise, C=est_cov)
    est_pw_cov = combine.estimate_noise_cov(pwdata, prop=0.1)
    assert np.allclose(est_pw_cov, np.eye(4), atol=1E-1)

    pwdata, W, C = combine.prewhiten(noise)
    est_pw_cov = combine.estimate_noise_cov(pwdata, prop=0.1)
    assert np.allclose(est_pw_cov, np.eye(4), atol=1E-1)


def test_svd_reduce():

    cov = 1E-4 * np.asarray(
        [[1, 0.4, 0.01],
         [0.4, 0.9, 0.4],
         [0.01, 0.4, 1.1]])
    coil_weights = np.array([1 + 1j * 0, 0.6 + 1j * 0.2, 0.2 - 1j * 0.3])

    fids, _ = syn.syntheticFID(
        coilamps=np.abs(coil_weights),
        coilphase=np.angle(coil_weights),
        noisecovariance=cov,
        bandwidth=1000,
        points=1024,
        chemicalshift=[-2, -3],
        amplitude=[1, 1],
        phase=[0, 0],
        damping=[30, 30],
        g=[0, 0],
        nucleus='1H')

    fids = np.asarray(fids).T
    pwfids, W, C = combine.prewhiten(fids, C=cov)
    comb_fid, alpha, amps = combine.svd_reduce(pwfids, W, C, return_alpha=True)

    assert comb_fid.shape == (1024,)

    post_calc_comb = np.sum(alpha * fids, axis=1)
    assert np.allclose(post_calc_comb, comb_fid)

    svd_rescale = np.linalg.norm(coil_weights) * (coil_weights[0] / np.abs(coil_weights[0]))
    scaled_true_weights = coil_weights / svd_rescale

    assert np.allclose(amps.conj(), scaled_true_weights, atol=1E-1)


def test_combine():

    cov = 1E-4 * np.asarray(
        [[1, 0.4, 0.01],
         [0.4, 0.9, 0.4],
         [0.01, 0.4, 1.1]])
    coil_weights = np.array([1 + 1j * 0, 0.6 + 1j * 0.2, 0.2 - 1j * 0.3])

    fids, _ = syn.syntheticFID(
        coilamps=np.abs(coil_weights),
        coilphase=np.angle(coil_weights),
        noisecovariance=cov,
        bandwidth=1000,
        points=1024,
        chemicalshift=[-2, -3],
        amplitude=[1, 1],
        phase=[0, 0],
        damping=[30, 30],
        g=[0, 0],
        nucleus='1H')

    comb_fid1 = combine.combine_FIDs(fids, 'svd', cov=cov, do_prewhiten=True)
    comb_fid2, weights, amps = combine.combine_FIDs(fids, 'svd_weights', cov=cov, do_prewhiten=True)
    comb_fid3 = combine.combine_FIDs(fids, 'weighted', weights=weights)

    assert comb_fid1.shape == (1024, )
    assert np.allclose(comb_fid1, comb_fid2)
    assert np.allclose(comb_fid1, comb_fid3)

    post_calc_comb = np.sum(weights * np.asarray(fids).T, axis=1)
    assert np.allclose(post_calc_comb, comb_fid3)

    svd_rescale = np.linalg.norm(coil_weights) * (coil_weights[0] / np.abs(coil_weights[0]))
    scaled_true_weights = coil_weights / svd_rescale
    assert np.allclose(amps.conj(), scaled_true_weights, atol=1E-1)


def test_single_coil_combine():

    fids, _ = syn.syntheticFID(
        noisecovariance=[[1]],
        bandwidth=1000,
        points=1024,
        chemicalshift=[-2, -3],
        amplitude=[1, 1],
        phase=[0, 0],
        damping=[30, 30],
        g=[0, 0],
        nucleus='1H')

    assert len(fids) == 1
    print(np.asarray(fids).T.shape)
    comb_fid = combine.combine_FIDs(fids, 'svd', do_prewhiten=True)
    assert np.allclose(comb_fid, fids)


def test_nifti_mrs_coilcomb():
    cov = 1E-4 * np.asarray(
        [[1, 0.4, 0.01],
         [0.4, 0.9, 0.4],
         [0.01, 0.4, 1.1]])
    voxel_coilweights = np.array([
        [1 + 1j * 0, 0.6 + 1j * 0.2, 0.2 - 1j * 0.3],
        [0.8 + 1j * 0.2, 0.6 + 1j * 0.1, 0.5 - 1j * 0.5]])

    all_fids = []
    for idx in range(10):
        curr_fids = []
        for coilweight in voxel_coilweights:
            fids, info = syn.syntheticFID(
                coilamps=np.abs(coilweight),
                coilphase=np.angle(coilweight),
                noisecovariance=cov,
                bandwidth=1000,
                points=1024,
                chemicalshift=[-2, -3],
                amplitude=[1, 1],
                phase=[0, 0],
                damping=[30, 30],
                g=[0, 0],
                nucleus='1H')
            curr_fids.append(fids)
        all_fids.append(np.stack(curr_fids))

    all_fids = np.stack(all_fids)
    all_fids = all_fids.reshape((1, 1) + all_fids.shape)
    all_fids = np.moveaxis(all_fids, (3, -1, 2), (0, 3, -1))
    data = create_nmrs.gen_nifti_mrs(
        all_fids,
        1 / 1000,
        123.2,
        dim_tags=['DIM_COIL', 'DIM_DYN', None]
    )

    noise = np.random.multivariate_normal(
        np.zeros((3)),
        cov,
        int(1E6)) + \
        1j * np.random.multivariate_normal(
        np.zeros((3)),
        cov,
        int(1E6))

    comb_test_v1 = combine.combine_FIDs(data[0, 0, 0, :, :, 0], 'svd', cov=cov, do_prewhiten=True)
    comb_test_v2 = combine.combine_FIDs(data[1, 0, 0, :, :, 0], 'svd', cov=cov, do_prewhiten=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_wcov = coilcombine(data, covariance=cov)
        combined_wnoise = coilcombine(data, noise=noise.T)

        combined_wocov = coilcombine(data)

    assert np.allclose(combined_wcov[0, 0, 0, :, 0], comb_test_v1)
    assert np.allclose(combined_wcov[1, 0, 0, :, 0], comb_test_v2)

    assert np.allclose(combined_wnoise[0, 0, 0, :, 0], comb_test_v1, atol=5E-4)
    assert np.allclose(combined_wnoise[1, 0, 0, :, 0], comb_test_v2, atol=5E-4)

    assert np.allclose(combined_wocov[0, 0, 0, :, 0], comb_test_v1, atol=1E-2)
    assert np.allclose(combined_wocov[1, 0, 0, :, 0], comb_test_v2, atol=1E-2)

    # Repeat with different dimension ordering to make sure of implementation
    all_fids = np.swapaxes(all_fids, -1, -2)
    data = create_nmrs.gen_nifti_mrs(
        all_fids,
        1 / 1000,
        123.2,
        dim_tags=['DIM_DYN', 'DIM_COIL', None]
    )

    comb_test_v1 = combine.combine_FIDs(data[0, 0, 0, :, 0, :], 'svd', cov=cov, do_prewhiten=True)
    comb_test_v2 = combine.combine_FIDs(data[1, 0, 0, :, 0, :], 'svd', cov=cov, do_prewhiten=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_wcov = coilcombine(data, covariance=cov)
        combined_wnoise = coilcombine(data, noise=noise.T)

        combined_wocov = coilcombine(data)

    assert np.allclose(combined_wcov[0, 0, 0, :, 0], comb_test_v1)
    assert np.allclose(combined_wcov[1, 0, 0, :, 0], comb_test_v2)

    assert np.allclose(combined_wnoise[0, 0, 0, :, 0], comb_test_v1, atol=5E-4)
    assert np.allclose(combined_wnoise[1, 0, 0, :, 0], comb_test_v2, atol=5E-4)

    assert np.allclose(combined_wocov[0, 0, 0, :, 0], comb_test_v1, atol=1E-2)
    assert np.allclose(combined_wocov[1, 0, 0, :, 0], comb_test_v2, atol=1E-2)

    # Repeat with reference (use same data)
    from fsl_mrs.core import nifti_mrs as ntools
    ref_data, _ = ntools.split(data, 'DIM_DYN', 0)
    print(ref_data.shape)
    combined_wref = coilcombine(data, covariance=cov, reference=ref_data)

    assert np.allclose(combined_wref[0, 0, 0, :, 0], comb_test_v1)
    assert np.allclose(combined_wref[1, 0, 0, :, 0], comb_test_v2)
