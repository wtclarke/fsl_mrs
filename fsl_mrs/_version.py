from importlib.metadata import PackageNotFoundError, version as package_version

try:
    __version__ = package_version("fsl_mrs")
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version
    except ImportError:
        __version__ = "0+unknown"
    else:
        __version__ = get_version(root="..", relative_to=__file__)
