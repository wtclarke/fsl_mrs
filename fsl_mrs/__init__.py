from . import _version
__version__ = _version.get_versions()['version']

# Define the public API of the package
from .utils.mrs_io import read_FID, read_basis
from .utils.preproc import nifti_mrs_proc as proc
from .core.nifti_mrs import NIFTI_MRS, merge, split, reorder, reshape
__all__ = ["read_FID", "read_basis", "proc", "NIFTI_MRS", "merge", "split", "reorder", "reshape"]
