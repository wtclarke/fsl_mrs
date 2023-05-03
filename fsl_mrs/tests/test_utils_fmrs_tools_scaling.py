"""Test the contrast scaling functions used for fMRS analysis

Test functions that appear in utils.fmrs_tools.scaling module

Copyright Will Clarke, University of Oxford, 2023"""

from pathlib import Path

import pytest
import numpy as np

import fsl_mrs.utils.fmrs_tools.scalings as scalings
import fsl_mrs.utils.fmrs_tools.utils as util

testsPath = Path(__file__).parent
sim_results = testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/ctrl'


# Very basic tests, needs work
def test_internal_ref(tmp_path):

    with pytest.raises(ValueError):
        _ = scalings.fmrs_internal_reference(
            sim_results,
            'notPresent')

    comp = util.load_dyn_res(sim_results)
    out = scalings.fmrs_internal_reference(
        sim_results,
        'conc_Cr_beta3')

    scaled = out[0]['conc_NAA_beta3']
    comp_val = comp[0]['conc_NAA_beta3'] / comp[0]['conc_Cr_beta3']
    assert np.isclose(scaled, comp_val)
