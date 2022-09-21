'''
FSL-MRS test script

Test packaged examples.

Copyright Will Clarke, University of Oxford, 2022
'''
from fsl_mrs.utils import example
from fsl_mrs.core.mrs import MRS


def test_simulated():
    for idx in range(1, 29):
        assert isinstance(example.simulated(idx), MRS)
