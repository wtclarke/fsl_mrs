'''FSL-MRS test script

Test baseline tools

Copyright Will Clarke, University of Oxford, 2021'''

from fsl_mrs.utils import baseline
from fsl_mrs.core import MRS
import numpy as np


def test_regressor_creation():

    # Create dummy mrs
    mrs = MRS(FID=np.zeros((1000,)), cf=100, bw=2000, nucleus='1H')
    baseline_order = 3
    ppmlim = (-1, 1)
    B = baseline.prepare_baseline_regressor(mrs, baseline_order, ppmlim)

    # Need some better tests...
    assert B.shape == (1000, 8)
    assert np.sum(B[:, 0]) == 100
    assert np.sum(B[:, 1]) == 100j
