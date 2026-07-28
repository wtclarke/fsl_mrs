from importlib.metadata import PackageNotFoundError, version as package_version

try:
    __version__ = package_version("spec2nii")
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version
    except ImportError:
        __version__ = "0+unknown"
    else:
        __version__ = get_version(root="..", relative_to=__file__)

# Define the public API of the package
from .utils.mrs_io import read_FID, read_basis
from .utils.preproc import nifti_mrs_proc as proc
from .core.nifti_mrs import NIFTI_MRS, merge, split, reorder, reshape
__all__ = ["read_FID", "read_basis", "proc", "NIFTI_MRS", "merge", "split", "reorder", "reshape"]
