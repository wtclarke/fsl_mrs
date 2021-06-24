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
        == 'Coil combination, reference data used (wref_raw.nii.gz), prewhitening applied.'
