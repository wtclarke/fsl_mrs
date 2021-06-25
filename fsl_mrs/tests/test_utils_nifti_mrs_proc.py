'''FSL-MRS test script

Test NIfTI-MRS processing

Copyright Will Clarke, University of Oxford, 2021
'''
from pathlib import Path

from fsl_mrs.utils.preproc import nifti_mrs_proc as nproc
from fsl_mrs.utils.mrs_io import read_FID
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
    combined = nproc.coilcombine(nmrs_obj)
    aligned = nproc.align(combined, 'DIM_DYN', ppmlim=(1.0, 4.0), niter=1, apodize=5)

    assert aligned.hdr_ext['ProcessingApplied'][1]['Method'] == 'Frequency and phase correction'
    assert aligned.hdr_ext['ProcessingApplied'][1]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.align, dim=DIM_DYN, '\
           'target=None, ppmlim=(1.0, 4.0), niter=1, apodize=5.'


def test_aligndiff():
    # For want of data this is a bizzare way of using this function.
    nmrs_obj = read_FID(wrefc)
    aligned = nproc.aligndiff(nmrs_obj, 'DIM_COIL', 'DIM_DYN', 'add', ppmlim=(1.0, 4.0))

    assert aligned.hdr_ext['ProcessingApplied'][0]['Method'] == 'Alignment of subtraction sub-spectra'
    assert aligned.hdr_ext['ProcessingApplied'][0]['Details']\
        == 'fsl_mrs.utils.preproc.nifti_mrs_proc.aligndiff, dim_align=DIM_COIL, '\
           'dim_diff=DIM_DYN, diff_type=add, target=None, ppmlim=(1.0, 4.0).'
