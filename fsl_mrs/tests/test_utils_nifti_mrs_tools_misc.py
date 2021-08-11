from pathlib import Path

import numpy as np

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils import nifti_mrs_tools as nmrs_tools


testsPath = Path(__file__).parent
test_data = testsPath / 'testdata' / 'fsl_mrs_preproc' / 'metab_raw.nii.gz'


def test_conjugate():
    # Data is (1, 1, 1, 4096, 32, 64) ['DIM_COIL', 'DIM_DYN', None]
    nmrs = mrs_io.read_FID(test_data)

    conjugated = nmrs_tools.conjugate(nmrs)

    assert np.allclose(conjugated.data, np.conjugate(nmrs.data))
