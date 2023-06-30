'''FSL-MRS test script

Test MRSI preprocessing functions

Copyright Will Clarke, University of Oxford, 2023
'''

from pathlib import Path

import numpy as np
from pytest import fixture

from fsl.data.image import Image

from fsl_mrs.utils.preproc import mrsi
from fsl_mrs.utils.mrs_io import read_FID
from fsl_mrs.utils.synthetic import syntheticFID


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
    sfids, shifts = mrsi.xcorr_align(test_data, 1 / 4000)

    assert np.isclose(shifts[-1], -123.2 / 10, atol=1E0)
    assert sfids.shape == test_data.shape


def test_phase_corr_max_real(test_data):
    pfids, phases = mrsi.phase_corr_max_real(test_data)

    assert np.allclose(np.abs(phases), [0, np.pi, 0, np.pi], atol=1E-1)
    assert pfids.shape == test_data.shape


# Integration tests for NIfTI-MRS objects
testsPath = Path(__file__).parent
data = testsPath / 'testdata' / 'fsl_mrsi'
metab_path = data / 'FID_Metab.nii.gz'
mask_path = data / 'small_mask.nii.gz'


def test_mrsi_freq_align():
    mrsi_data = read_FID(metab_path)
    mask = Image(mask_path)

    aligned_data, shift_img = mrsi.mrsi_freq_align(mrsi_data, mask)

    assert aligned_data.shape == mrsi_data.shape
    assert shift_img.shape == mrsi_data.shape[:3]

    assert np.allclose(aligned_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(shift_img.voxToWorldMat, mrsi_data.voxToWorldMat)

    # Now with no mask and no zero padding
    aligned_data, shift_img = mrsi.mrsi_freq_align(mrsi_data, zpad_factor=0)

    assert aligned_data.shape == mrsi_data.shape
    assert shift_img.shape == mrsi_data.shape[:3]

    assert np.allclose(aligned_data.voxToWorldMat, mrsi_data.voxToWorldMat)
    assert np.allclose(shift_img.voxToWorldMat, mrsi_data.voxToWorldMat)


def test_mrsi_phase_corr():
    mrsi_data = read_FID(metab_path)
    mask = Image(mask_path)

    phased_data, phs_img = mrsi.mrsi_phase_corr(mrsi_data, mask)

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
