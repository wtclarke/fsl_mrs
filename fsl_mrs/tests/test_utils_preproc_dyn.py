'''FSL-MRS test script

Test dynamic fitting based preprocessing

Copyright Will Clarke, University of Oxford, 2021
'''
from pathlib import Path

from fsl_mrs.utils.preproc import nifti_mrs_proc as nproc
from fsl_mrs.utils.preproc import dyn_based_proc as dproc
from fsl_mrs.utils.mrs_io import read_FID, read_basis
from fsl_mrs.utils.nifti_mrs_tools import split
from fsl_mrs.utils import basis_tools as btools


testsPath = Path(__file__).parent
data = testsPath / 'testdata' / 'fsl_mrs_preproc'
metab = data / 'metab_raw.nii.gz'
wrefc = data / 'wref_raw.nii.gz'
basis_path = testsPath / 'testdata' / 'fsl_mrs' / 'steam_basis'


def test_dyn_align(tmp_path):
    nmrs_obj = read_FID(metab)
    nmrs_ref_obj = read_FID(wrefc)
    nmrs_ref_obj = nproc.average(nmrs_ref_obj, 'DIM_DYN')

    combined = nproc.coilcombine(nmrs_obj, reference=nmrs_ref_obj)

    reduced_data, _ = split(combined, 'DIM_DYN', 2)

    aligned_1 = nproc.align(reduced_data, 'DIM_DYN', ppmlim=(0.2, 4.2), apodize=0.0)

    basis = btools.conjugate_basis(read_basis(basis_path))

    fitargs = {'ppmlim': (0.2, 4.2), 'baseline_order': 1}
    aligned_2 = dproc.align_by_dynamic_fit(aligned_1, basis, fitargs=fitargs)
    dproc.align_by_dynamic_fit_report(
        aligned_1,
        aligned_2[0],
        aligned_2[1],
        aligned_2[2],
        ppmlim=fitargs['ppmlim'],
        html=str(tmp_path / 'align_report.html'))

    assert aligned_2[0].hdr_ext['ProcessingApplied'][2]['Method'] == 'Frequency and phase correction'
    assert aligned_2[0].hdr_ext['ProcessingApplied'][2]['Details']\
        == "fsl_mrs.utils.preproc.dyn_based_proc.align_by_dynamic_fit, "\
        "fitargs={'ppmlim': (0.2, 4.2), 'baseline_order': 1}."
    assert (tmp_path / 'align_report.html').is_file()
