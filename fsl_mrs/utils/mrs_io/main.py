# mrs_io.py - I/O utilities for FSL_MRS
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

from pathlib import Path
import re

import numpy as np

from nifti_mrs.nifti_mrs import NotNIFTI_MRS

from fsl_mrs.utils.mrs_io import fsl_io as fsl, jmrui_io
from fsl_mrs.utils.mrs_io import lcm_io as lcm
from fsl_mrs.utils.mrs_io import jmrui_io as jmrui
from fsl_mrs.core import nifti_mrs as fsl_nmrs
from fsl_mrs.core import basis as bmod
import fsl.utils.path as fslpath


class FileNotRecognisedError(Exception):
    pass


class UnknownBasisFormat(Exception):
    pass


class IncompatibleBasisFormat(Exception):
    pass


# Data reading functions:
# Load FIDs from NIFTI, .raw (LCMODEL style ) or .txt (jMRUI style) files
# These read functions should in general take a file name as a string and
# return the data in a numpy array with headers (in dict format) containing
# the following mandatory fields:
# Reciever bandwidth
# Dwell time
# central frequency
def _check_datatype(filename):
    """
    Identify the file type (.nii(.gz),.RAW/.H2O,.txt)
    Returns one of: 'NIFTI', 'RAW', 'TXT', 'Unknown'

    :param filename: Path object to file
    :type filename: patlib.Path
    :return: type of file
    :rtype: str
    :return: extension
    :rtype: str
    """
    ext = ''.join(filename.suffixes)

    ptrn_nii = re.compile(r'\.nii(\.gz)?', flags=re.IGNORECASE)
    ptrn_lcm = re.compile(r'(\.raw)|(\.h2o)', flags=re.IGNORECASE)
    ptrn_jmrui = re.compile(r'\.txt', flags=re.IGNORECASE)

    if ptrn_nii.search(ext):
        return 'NIFTI', ext
    elif ptrn_lcm.search(ext):
        return 'RAW', ext
    elif ptrn_jmrui.search(ext):
        return 'TXT', ext
    else:
        return 'Unknown', ext


def read_FID(filename):
    """
     Read FID file. Tries to detect type automatically

     Parameters
     ----------
     filename : str or pathlib.Path

     Returns:
     --------
     array-like (complex)
     dict (header info)
    """

    # Resolve symlinks
    # Once this is done for fslpy's Image class then this can be removed from here.
    if Path(filename).is_symlink():
        filename = Path(filename).resolve()

    try:
        return fsl_nmrs.NIFTI_MRS(filename, validate_on_creation=False)
    except (NotNIFTI_MRS, fslpath.PathError):
        data_type, id_ext = _check_datatype(Path(filename))

    # Handle pathlib Path objects for individual loader interfaces
    filename = str(filename)

    if data_type == 'RAW':
        return lcm.read_lcm_raw_h2o(filename)
    elif data_type == 'NIFTI':
        return fsl.readNIFTI(filename)
    elif data_type == 'TXT':
        return jmrui.readjMRUItxt_fid(filename)
    else:
        raise FileNotRecognisedError(
            f'Cannot read data format {data_type} '
            f'for file {filename} with identified extension {id_ext}.')


# Basis reading functions
# Formats accepted are .json, .basis/.raw (LCMODEL style) or .txt (jMRUI style)
# Now handled by the Basis class methods
def read_basis(filename):
    """
    Read basis file(s) to generate a Basis object

    Load the basis fids, names and headers for each format handled.
    Ensures similar sorting by name for each type.

    :param filepath: Path to basis file or folder
    :type filepath: str or pathlib.Path
    :return: A Basis class object
    :rtype: fsl_mrs.core.basis.Basis
    """

    # Handle str objects
    if isinstance(filename, str):
        filename = Path(filename)

    # LCModel BASIS format format
    if filename.is_file():
        if filename.suffix.lower() == '.basis':
            basis, names, header = lcm.readLCModelBasis(filename)
            # Sort by name to match sorted filenames of other formats
            so = np.argsort(names)
            basis = basis[:, so]
            names = list(np.array(names)[so])
            header = list(np.array(header)[so])

            # Add missing hdr field
            for hdr in header:
                hdr['fwhm'] = None
        elif filename.suffix.lower() == '.txt':
            basis, names, header = jmrui_io.read_txtBasis_files([filename, ])
        else:
            raise UnknownBasisFormat(f'Cannot read data format {filename.suffix}')

    elif filename.is_dir():
        fslfiles = sorted(list(filename.glob('*.json')))
        rawfiles = sorted(list(filename.glob('*.RAW')) + list(filename.glob('*.raw')))
        txtfiles = sorted(list(filename.glob('*.txt')))
        if fslfiles:
            basis, names, header = fsl.readFSLBasisFiles(filename)
        elif txtfiles:
            basis, names, header = jmrui.read_txtBasis_files(txtfiles)
        elif rawfiles:
            raise IncompatibleBasisFormat("LCModel raw files don't contain enough information"
                                          " to generate a Basis object. Please use fsl_mrs.utils.mrs_io"
                                          ".lcm_io.read_basis_files to load the partial information.")
        else:
            raise UnknownBasisFormat(f'{filename} contains neither .json, .txt, or .raw basis files!')
    else:
        raise UnknownBasisFormat(f'{filename} is neither a file nor a folder!')

    # Handle single basis spectra
    if basis.ndim == 1:
        basis = basis[:, np.newaxis]

    return bmod.Basis(basis, names, header)
