'''FSL-MRS test script

Test MRSI preprocessing functions

Copyright Will Clarke, University of Oxford, 2023
'''

from pathlib import Path

import numpy as np
from pytest import fixture, raises

from fsl.data.image import Image

from fsl_mrs.utils.preproc import mrsi
from fsl_mrs.utils.mrs_io import read_FID, read_basis
from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl_mrs.core import nifti_mrs as ntools


# Algorithm tests
@fixture
def test_data():
    fids, hdrs = syntheticFID(
        coilamps=[1, 1],
        coilphase=[0, np.pi],
        noisecovariance=np.zeros((2, 2)),
        chemicalshift=[1, 3])

    fids_shift, hdrs = syntheticFID(
        coilamps=[1, 1],
        coilphase=[0, np.pi],
        noisecovariance=np.zeros((2, 2)),
        chemicalshift=[1.1, 3.1])

    return np.concatenate((fids, fids_shift))


def test_xcorr_align(test_data):
    # Test default
    sfids, shifts = mrsi.xcorr_align(test_data, 1 / 4000)

    assert np.isclose(shifts[-1], -123.2 / 10, atol=1E0)
    assert sfids.shape == test_data.shape

    # Test apodisation and zeropadding options
    sfids, shifts = mrsi.xcorr_align(test_data, 1 / 4000, apodize_hz=5)

    assert np.isclose(shifts[-1], -123.2 / 10, atol=1E0)
    assert sfids.shape == test_data.shape

    sfids, shifts = mrsi.xcorr_align(test_data, 1 / 4000, zpad_factor=0)
    assert np.isclose(shifts[-1], -123.2 / 10, atol=1E0)
    assert sfids.shape == test_data.shape

    sfids, shifts = mrsi.xcorr_align(test_data, 1 / 4000, zpad_factor=2)
    assert np.isclose(shifts[-1], -123.2 / 10, atol=1E0)
    assert sfids.shape == test_data.shape

    # Test Target
    sfids, shifts = mrsi.xcorr_align(
        test_data,
        1 / 4000,
        target=test_data[0, :])
    assert np.allclose(shifts, [0, 0, -12.3, -12.3], atol=1E0)
    assert sfids.shape == test_data.shape

    with raises(ValueError):
        sfids, shifts = mrsi.xcorr_align(
            test_data,
            1 / 4000,
            target=np.zeros(100))


def test_phase_corr_max_real(test_data):
    pfids, phases = mrsi.phase_corr_max_real(test_data, 1 / 4000)

    assert np.allclose(np.abs(phases), [0, np.pi, 0, np.pi], atol=1E-1)
    assert pfids.shape == test_data.shape


# Integration tests for NIfTI-MRS objects
testsPath = Path(__file__).parent
data = testsPath / 'testdata' / 'fsl_mrsi'
metab_path = data / 'FID_Metab.nii.gz'
mask_path = data / 'small_mask.nii.gz'
basis_path = data / '3T_slaser_32vespa_1250_wmm'


def test_mrsi_freq_align():
    mrsi_data = read_FID(metab_path)
    mask = Image(mask_path)

    aligned_data, shift_img = mrsi.mrsi_freq_align(mrsi_data, mask=mask)

    assert aligned_data.shape == mrsi_data.shape
    assert shift_img.shape == mrsi_data.shape[:3]

    assert np.allclose(aligned_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(shift_img.voxToWorldMat, mrsi_data.voxToWorldMat)

    # Now with no mask and no zero padding and fixed apod
    aligned_data, shift_img = mrsi.mrsi_freq_align(mrsi_data, zpad_factor=0, apodize=20)

    assert aligned_data.shape == mrsi_data.shape
    assert shift_img.shape == mrsi_data.shape[:3]

    assert np.allclose(aligned_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(shift_img.voxToWorldMat, mrsi_data.voxToWorldMat)

    # Test with single voxel as target
    target = gen_nifti_mrs(
        mrsi_data[24, 24, 0, :].reshape((1, 1, 1, -1)),
        mrsi_data.dwelltime,
        mrsi_data.spectrometer_frequency[0]
    )
    aligned_data, shift_img = mrsi.mrsi_freq_align(
        mrsi_data,
        mask=mask,
        target=target)

    assert aligned_data.shape == mrsi_data.shape
    assert shift_img.shape == mrsi_data.shape[:3]

    assert np.allclose(aligned_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(shift_img.voxToWorldMat, mrsi_data.voxToWorldMat)

    # Test with basis
    target = read_basis(basis_path)
    aligned_data, shift_img = mrsi.mrsi_freq_align(
        mrsi_data,
        mask=mask,
        target=target)

    aligned_data, shift_img = mrsi.mrsi_freq_align(
        mrsi_data,
        mask=mask,
        target=target,
        basis_ignore=['GABA', 'Lac', 'Glu'],
        apodize=0)

    assert aligned_data.shape == mrsi_data.shape
    assert shift_img.shape == mrsi_data.shape[:3]

    assert np.allclose(aligned_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(shift_img.voxToWorldMat, mrsi_data.voxToWorldMat)

    # Test with higher dimensions
    import fsl_mrs.core.nifti_mrs as ntools
    mrsi_higher = ntools.reorder(mrsi_data, ['DIM_DYN', None, None])
    mrsi_higher = ntools.merge((mrsi_higher, mrsi_higher), 'DIM_DYN')

    aligned_data, shift_img = mrsi.mrsi_freq_align(
        mrsi_higher,
        mask=mask,
        target=target,
        basis_ignore=['GABA', 'Lac', 'Glu'],
        apodize=0,
        higher_dimensions='separate')

    aligned_data, shift_img = mrsi.mrsi_freq_align(
        mrsi_higher,
        mask=mask,
        target=target,
        basis_ignore=['GABA', 'Lac', 'Glu'],
        apodize=0,
        higher_dimensions='combine')

    aligned_data, shift_img = mrsi.mrsi_freq_align(
        mrsi_higher,
        mask=mask,
        target=target,
        basis_ignore=['GABA', 'Lac', 'Glu'],
        apodize=0,
        higher_dimensions=1)


def test_mrsi_phase_corr():
    mrsi_data = read_FID(metab_path)
    mask = Image(mask_path)

    phased_data, phs_img = mrsi.mrsi_phase_corr(mrsi_data, mask=mask)

    assert phased_data.shape == mrsi_data.shape
    assert phs_img.shape == mrsi_data.shape[:3]

    assert np.allclose(phased_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(phs_img.voxToWorldMat, mrsi_data.voxToWorldMat)

    # No mask with a ppm limit
    phased_data, phs_img = mrsi.mrsi_phase_corr(mrsi_data, ppmlim=(0, 3))

    assert phased_data.shape == mrsi_data.shape
    assert phs_img.shape == mrsi_data.shape[:3]

    assert np.allclose(phased_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(phs_img.voxToWorldMat, mrsi_data.voxToWorldMat)

    # handling multi-dimensional data
    mrsi_data_hd = ntools.merge([
        ntools.reorder(mrsi_data, ['DIM_DYN', None, None]),
        ntools.reorder(mrsi_data, ['DIM_DYN', None, None])],
        'DIM_DYN')

    phased_data_hd, phs_hd_avg = mrsi.mrsi_phase_corr(mrsi_data_hd, mask=mask)
    assert phased_data_hd.shape == mrsi_data_hd.shape

    phased_data_hd, phs_hd_single = mrsi.mrsi_phase_corr(mrsi_data_hd, mask=mask, higher_dim_index=[0, ])
    assert phased_data_hd.shape == mrsi_data_hd.shape
    assert np.allclose(phs_hd_single[:], phs_hd_avg[:])


@fixture
def lipid_test_data():
    fid_met, hdrs = syntheticFID(
        coilamps=[1],
        coilphase=[0],
        noisecovariance=[[0]],
        chemicalshift=[-2],
        points=512,
        bandwidth=1000)

    fid_lipid, hdrs = syntheticFID(
        coilamps=[100],
        coilphase=[0],
        noisecovariance=[[0]],
        chemicalshift=[-4],
        points=512,
        bandwidth=1000)

    mrsi_data = np.zeros((8, 8, 1, 512), dtype=complex)
    mrsi_data[1:7, 1:7, 0, :] = fid_lipid
    mrsi_data[2:6, 2:6, 0, :] = fid_met

    mask_data = np.zeros((8, 8, 1), dtype=np.int16)
    mask_data[1:7, 1:7, 0] = 1
    mask_data[2:6, 2:6, 0] = 0

    data = gen_nifti_mrs(mrsi_data, 1 / 1000, 123.2, '1H')
    return data, Image(mask_data), np.concatenate((fid_lipid, fid_lipid, fid_lipid, fid_lipid)).T


def test_lipid_removal_l2(lipid_test_data):
    data, mask, basis = lipid_test_data

    # Wrong shaped data
    from fsl_mrs.core.nifti_mrs import merge, reorder
    wrong_data = merge(
        (reorder(data, ['DIM_DYN', None, None]), reorder(data, ['DIM_DYN', None, None])),
        dimension='DIM_DYN')
    with raises(ValueError):
        mrsi.lipid_removal_l2(wrong_data, beta=1E-5, lipid_mask=mask, lipid_basis=None)

    # No mask or basis
    with raises(TypeError):
        mrsi.lipid_removal_l2(data)

    # Mask errors
    # Wrong type
    with raises(TypeError):
        mrsi.lipid_removal_l2(data, lipid_mask=mask[:])
    # No voxels selected
    with raises(ValueError):
        mrsi.lipid_removal_l2(data, lipid_mask=Image(np.zeros((8, 8, 1), dtype=np.int16)))

    # basis errors
    with raises(ValueError):
        mrsi.lipid_removal_l2(data, lipid_basis=basis[:256, :])

    data_lipid_removed = mrsi.lipid_removal_l2(data, beta=1E-5, lipid_mask=mask)

    assert data_lipid_removed.shape == data.shape
    assert np.max(np.abs(data_lipid_removed[1, 1, 0, :])) < 10

    data_lipid_removed = mrsi.lipid_removal_l2(data, beta=1E-5, lipid_basis=basis)

    assert data_lipid_removed.shape == data.shape
    assert np.max(np.abs(data_lipid_removed[1, 1, 0, :])) < 10
