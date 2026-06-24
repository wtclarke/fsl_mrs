'''FSL-MRS test script

Test dynamic fitting based preprocessing

Copyright Will Clarke, University of Oxford, 2021
'''
from pathlib import Path

from fsl_mrs import proc, split, read_FID, read_basis
from fsl_mrs.utils.preproc import dyn_based_proc as dproc
from fsl_mrs.utils import basis_tools as btools


testsPath = Path(__file__).parent
data = testsPath / 'testdata' / 'fsl_mrs_preproc'
metab = data / 'metab_raw.nii.gz'
wrefc = data / 'wref_raw.nii.gz'
basis_path = testsPath / 'testdata' / 'fsl_mrs' / 'steam_basis'


def test_dyn_align(tmp_path):
    nmrs_obj = read_FID(metab)
    nmrs_ref_obj = read_FID(wrefc)
    nmrs_ref_obj = proc.average(nmrs_ref_obj, 'DIM_DYN')

    combined = proc.coilcombine(nmrs_obj, reference=nmrs_ref_obj)

    reduced_data, _ = split(combined, 'DIM_DYN', 2)

    aligned_1 = proc.align(reduced_data, 'DIM_DYN', ppmlim=(0.2, 4.2))

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
