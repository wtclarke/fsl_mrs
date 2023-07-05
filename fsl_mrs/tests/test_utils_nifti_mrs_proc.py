'''FSL-MRS test script

Test NIfTI-MRS processing

Copyright Will Clarke, University of Oxford, 2021
'''
from pathlib import Path
from re import escape

from pytest import raises
import numpy as np

from fsl_mrs.utils.preproc import nifti_mrs_proc as nproc
from fsl_mrs.utils.mrs_io import read_FID
from fsl_mrs.core.nifti_mrs import split
from fsl_mrs import __version__


testsPath = Path(__file__).parent
data = testsPath / 'testdata' / 'fsl_mrs_preproc'
metab = data / 'metab_raw.nii.gz'
wrefc = data / 'wref_raw.nii.gz'
wrefq = data / 'quant_raw.nii.gz'
ecc = data / 'ecc.nii.gz'


def test_update_processing_prov():
    nmrs_obj = read_FID(metab)

    assert 'ProcessingApplied' not in nmrs_obj.hdr_ext

    nproc.update_processing_prov(nmrs_obj, 'test', 'test_str')

    assert 'ProcessingApplied' in nmrs_obj.hdr_ext
    assert isinstance(nmrs_obj.hdr_ext['ProcessingApplied'], list)
    assert 'Time' in nmrs_obj.hdr_ext['ProcessingApplied'][0]
    assert nmrs_obj.hdr_ext['ProcessingApplied'][0]['Program'] == 'FSL-MRS'
    assert nmrs_obj.hdr_ext['ProcessingApplied'][0]['Version'] == __version__
    assert nmrs_obj.hdr_ext['ProcessingApplied'][0]['Method'] == 'test'
    assert nmrs_obj.hdr_ext['ProcessingApplied'][0]['Details'] == 'test_str'


def test_coilcombine():
    nmrs_obj = read_FID(metab)
    nmrs_ref_obj = read_FID(wrefc)
    nmrs_ref_obj = nproc.average(nmrs_ref_obj, 'DIM_DYN')

    combined = nproc.coilcombine(nmrs_obj, reference=nmrs_ref_obj)

    assert combined.hdr_ext['ProcessingApplied'][0]['Method'] == 'RF coil combination'
    assert combined.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.coilcombine, reference=wref_raw.nii.gz, no_prewhiten=False.'


def test_average():
    nmrs_ref_obj = read_FID(wrefc)
    combined = nproc.average(nmrs_ref_obj, 'DIM_DYN')

    assert combined.hdr_ext['ProcessingApplied'][0]['Method'] == 'Signal averaging'
    assert combined.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.average, dim=DIM_DYN.'


def test_align():
    nmrs_obj = read_FID(metab)
    with_coils, _ = split(nmrs_obj, 'DIM_COIL', 3)
    aligned1 = nproc.align(with_coils, 'DIM_DYN', ppmlim=(1.0, 4.0), niter=1, apodize=5)

    assert aligned1.hdr_ext['ProcessingApplied'][0]['Method'] == 'Frequency and phase correction'
    assert aligned1.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.align, dim=DIM_DYN, '\
           'target=None, ppmlim=(1.0, 4.0), niter=1, apodize=5.'

    combined = nproc.coilcombine(nmrs_obj)
    aligned2 = nproc.align(combined, 'DIM_DYN', ppmlim=(1.0, 4.0), niter=1, apodize=5)

    assert aligned2.hdr_ext['ProcessingApplied'][1]['Method'] == 'Frequency and phase correction'
    assert aligned2.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.align, dim=DIM_DYN, '\
           'target=None, ppmlim=(1.0, 4.0), niter=1, apodize=5.'

    # Align across all spectra
    aligned3 = nproc.align(with_coils, 'all', ppmlim=(1.0, 4.0), niter=1, apodize=5)

    assert aligned3.hdr_ext['ProcessingApplied'][0]['Method'] == 'Frequency and phase correction'
    assert aligned3.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.align, dim=all, '\
           'target=None, ppmlim=(1.0, 4.0), niter=1, apodize=5.'


def test_aligndiff():
    # For want of data this is a bizzare way of using this function.
    nmrs_obj = read_FID(wrefc)
    aligned = nproc.aligndiff(nmrs_obj, 'DIM_COIL', 'DIM_DYN', 'add', ppmlim=(1.0, 4.0))

    assert aligned.hdr_ext['ProcessingApplied'][0]['Method'] == 'Alignment of subtraction sub-spectra'
    assert aligned.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.aligndiff, dim_align=DIM_COIL, '\
           'dim_diff=DIM_DYN, diff_type=add, target=None, ppmlim=(1.0, 4.0).'


def test_ecc():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')
    ref_obj = read_FID(ecc)

    corrected = nproc.ecc(nmrs_obj, reference=ref_obj)

    assert corrected.hdr_ext['ProcessingApplied'][1]['Method'] == 'Eddy current correction'
    assert corrected.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.ecc, reference=ecc.nii.gz.'


def test_remove():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    corrected = nproc.remove_peaks(nmrs_obj, (4, 5.30))

    assert corrected.hdr_ext['ProcessingApplied'][1]['Method'] == 'Nuisance peak removal'
    assert corrected.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.remove_peaks, limits=(4, 5.3), limit_units=ppm+shift.'


def test_hlsvd_model():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    modeled = nproc.hlsvd_model_peaks(nmrs_obj, (4, 5.30), components=3)

    assert modeled.hdr_ext['ProcessingApplied'][1]['Method'] == 'HLSVD modeling'
    assert modeled.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.hlsvd_model_peaks,'\
           ' limits=(4, 5.3), limit_units=ppm+shift, components=3.'


def test_tshift():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    shifted = nproc.tshift(nmrs_obj, tshiftStart=0.001, tshiftEnd=0.001, samples=1024)

    assert shifted.hdr_ext['ProcessingApplied'][1]['Method'] == 'Temporal resample'
    assert shifted.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.tshift, tshiftStart=0.001, tshiftEnd=0.001, samples=1024.'


def test_truncate_or_pad():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    shifted = nproc.truncate_or_pad(nmrs_obj, -2, 'last')

    assert shifted.hdr_ext['ProcessingApplied'][1]['Method'] == 'Zero-filling'
    assert shifted.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.truncate_or_pad, npoints=-2, position=last.'


def test_apodize():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    apodized = nproc.apodize(nmrs_obj, (10.0,))

    assert apodized.hdr_ext['ProcessingApplied'][1]['Method'] == 'Apodization'
    assert apodized.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.apodize, amount=(10.0,), filter=exp.'


def test_fshift():
    nmrs_obj = read_FID(wrefc)

    print(nmrs_obj.shape)
    # Test shift per FID
    with raises(
        ValueError,
        match=escape(
            'Shift map must be the same size as the NIfTI-MRS spatial + higher dimensions. '
            'Current size = (32, 1), required shape = (1, 1, 1, 32, 2).')):
        shifted = nproc.fshift(nmrs_obj, np.ones((32, 1)))

    shifted = nproc.fshift(nmrs_obj, np.ones((1, 1, 1, 32, 2)))

    assert shifted.shape == nmrs_obj.shape
    assert shifted.hdr_ext['ProcessingApplied'][0]['Method'] == 'Frequency and phase correction'
    assert shifted.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.fshift, amount=per-voxel shifts specified.'

    # Test a single value shift
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')
    shifted = nproc.fshift(nmrs_obj, 10.0)

    assert shifted.hdr_ext['ProcessingApplied'][1]['Method'] == 'Frequency and phase correction'
    assert shifted.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.fshift, amount=10.0.'


def test_shift_to_reference():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    shifted = nproc.shift_to_reference(nmrs_obj, 4.65, (4.0, 5.0))

    assert shifted.hdr_ext['ProcessingApplied'][1]['Method'] == 'Frequency and phase correction'
    assert shifted.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.shift_to_reference, '\
           'ppm_ref=4.65, peak_search=(4.0, 5.0), use_avg=False.'


def test_shift_to_reference_no_avg():
    nmrs_obj = read_FID(wrefc)

    shifted = nproc.shift_to_reference(nmrs_obj, 4.65, (4.0, 5.0), use_avg=True)

    assert shifted.hdr_ext['ProcessingApplied'][0]['Method'] == 'Frequency and phase correction'
    assert shifted.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.shift_to_reference, '\
           'ppm_ref=4.65, peak_search=(4.0, 5.0), use_avg=True.'


def test_remove_unlike():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.coilcombine(nmrs_obj)
    processed, _ = nproc.remove_unlike(nmrs_obj)

    assert processed.hdr_ext['ProcessingApplied'][1]['Method'] == 'Outlier removal'
    assert processed.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.remove_unlike, ppmlim=None, sdlimit=1.96, niter=2.'


def test_phase_correct():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    phased = nproc.phase_correct(nmrs_obj, (4.0, 5.0), hlsvd=False)

    assert phased.hdr_ext['ProcessingApplied'][1]['Method'] == 'Phasing'
    assert phased.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.phase_correct, ppmlim=(4.0, 5.0), hlsvd=False, use_avg=False.'


def test_phase_correct_use_avg():
    nmrs_obj = read_FID(wrefc)
    phased = nproc.phase_correct(nmrs_obj, (4.0, 5.0), hlsvd=False, use_avg=True)

    assert phased.hdr_ext['ProcessingApplied'][0]['Method'] == 'Phasing'
    assert phased.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.phase_correct, ppmlim=(4.0, 5.0), hlsvd=False, use_avg=True.'


def test_apply_fixed_phase():
    nmrs_obj = read_FID(wrefc)
    nmrs_obj = nproc.average(nmrs_obj, 'DIM_DYN')

    phased = nproc.apply_fixed_phase(nmrs_obj, 180.0, p1=0.001)

    assert phased.hdr_ext['ProcessingApplied'][1]['Method'] == 'Phasing'
    assert phased.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.apply_fixed_phase, p0=180.0, p1=0.001, p1_type=shift.'


def test_subtract():
    nmrs_obj = read_FID(wrefc)

    subtracted = nproc.subtract(nmrs_obj, dim='DIM_DYN')

    assert subtracted.hdr_ext['ProcessingApplied'][0]['Method'] == 'Subtraction of sub-spectra'
    assert subtracted.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.subtract, data1=None, dim=DIM_DYN.'


def test_add():
    nmrs_obj = read_FID(wrefc)

    added = nproc.add(nmrs_obj, dim='DIM_DYN')

    assert added.hdr_ext['ProcessingApplied'][0]['Method'] == 'Addition of sub-spectra'
    assert added.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.add, data1=None, dim=DIM_DYN.'


def test_conjugate():
    nmrs_obj = read_FID(wrefc)

    conjugated = nproc.conjugate(nmrs_obj)

    assert conjugated.hdr_ext['ProcessingApplied'][0]['Method'] == 'Conjugation'
    assert conjugated.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.conjugate.'
