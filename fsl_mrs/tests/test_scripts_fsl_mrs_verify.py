'''FSL-MRS test script

Test the installation verification script

Copyright Will Clarke, University of Oxford, 2022'''

import os

from fsl_mrs.scripts import fsl_mrs_verify


def test_verify(tmp_path):
    os.chdir(tmp_path)

    try:
        fsl_mrs_verify.main()
    except Exception as exc:
        assert False, f"'fsl_mrs_verify.main()' fialed and raised an exception {exc}"
