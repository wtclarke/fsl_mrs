#!/usr/bin/env python

# mrs_io.py - I/O utilities for FSL_MRS
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import os, glob
import numpy as np
import nibabel as nib

from fsl_mrs.utils.mrs_io import fsl_io as fsl
from fsl_mrs.utils.mrs_io import lcm_io as lcm
from fsl_mrs.utils.mrs_io import jmrui_io as jmrui

# Data reading functions:
# Load FIDs from NIFTI, .raw (LCMODEL style ) or .txt (jMRUI style) files
# These read functions should in general take a file name as a string and 
# return the data in a numpy array with headers (in dict format) containing 
# the following mandatory fields:
# Reciever bandwidth 
# Dwell time
# central frequency 
def check_datatype(filename):
    """ 
    Identify the file type (.nii(.gz),.RAW/.H2O,.txt)
    Returns one of: 'NIFTI','RAW','TXT','Unknown'
    """
    try:
        garbage = nib.load(filename)   
    except:

        _,ext = os.path.splitext(filename)
        if ext.lower()=='.raw' or ext.lower()=='.h2o':
            garbage = lcm.readLCModelRaw(filename)
            return 'RAW'
        elif ext.lower()=='.txt':
            garbage = jmrui.readjMRUItxt(filename)
            return 'TXT'
        else:
            return 'Unknown'

    else:
        return 'NIFTI'

def read_FID(filename,squeezeSVS=True):
    """
     Read FID file. Tries to detect type automatically

     Parameters
     ----------
     filename : str
     squeezeSVS : optional , bool Remove signleton dimensions in nifti load of svs

     Returns:
     --------
     array-like (complex)
     dict (header info)
    """
    data_type = check_datatype(filename)
    if data_type == 'RAW':
        data,header = lcm.readLCModelRaw(filename)
    elif data_type == 'NIFTI':
        data,header = fsl.readNIFTI(filename,squeezeSVS=squeezeSVS)
    elif data_type == 'TXT':
        data,header = jmrui.readjMRUItxt(filename)
    else:
        raise ValueError(f'Cannot read data format {data_type} for file {filename}.')
    return data,header

# Basis reading functions
# Formats accepted are .json, .basis/.raw (LCMODEL style) or .txt (jMRUI style)
def read_basis(filename):
    """
    Read basis file(s). Function determines file type and calls appropriate loading function.
    Parameters
    ----------
    filename : string
        Name of basis file or folder
    
    Returns
    -------
    array-like
        Complex basis FIDS
    list
        List of metabolite names
    Dict
        Header information 
    """
    if os.path.isfile(filename):
        _,ext = os.path.splitext(filename)
        if ext.lower() == '.basis':
            basis,names,header = lcm.readLCModelBasis(filename)
            # Sort by name to match sorted filenames of other formats
            so = np.argsort(names)
            basis = basis[:,so]
            names = list(np.array(names)[so])
            header = list(np.array(header)[so])

        else:
            raise(Exception('Cannot read data format {}'.format(ext)))
    elif os.path.isdir(filename):
        folder = filename
        fslfiles = sorted(glob.glob(os.path.join(folder,'*.json')))
        rawfiles = sorted(glob.glob(os.path.join(folder,'*.RAW')))
        txtfiles = sorted(glob.glob(os.path.join(folder,'*.txt')))
        if fslfiles:
            basis,names,header = fsl.readFSLBasisFiles(folder)
        elif txtfiles: 
            basis,names,header = jmrui.read_txtBasis_files(txtfiles)            
        elif rawfiles: 
            basis,names = lcm.read_basis_files(rawfiles)
            header = None
        else:
            raise(Exception(f'{folder} contains neither .json basis files or .raw files!'))
    else:
        raise(Exception('{} is neither a file nor a folder!'.format(filename)))
        
    return basis,names,header



