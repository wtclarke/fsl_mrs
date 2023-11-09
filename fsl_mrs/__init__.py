from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# from fsl_mrs.core import MRS
# from fsl_mrs.core import MRSI

from . import _version
__version__ = _version.get_versions()['version']
