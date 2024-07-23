'''FSL-MRS test script

Test the installation verification script

Copyright Will Clarke, University of Oxford, 2022'''

from contextlib import contextmanager
import os
from pathlib import Path

from fsl_mrs.scripts import fsl_mrs_verify


@contextmanager
def set_directory(path: Path):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def test_verify(tmp_path):

    with set_directory(tmp_path):
        try:
            fsl_mrs_verify.main()
        except Exception as exc:

            assert False, f"'fsl_mrs_verify.main()' failed and raised an exception {exc}"
