# jmrui_io.py - I/O utilities for jmrui file formats in FSL_MRS
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford 
# SHBASECOPYRIGHT
import numpy as np
import re
import os.path as op

# Read jMRUI style text files
def readjMRUItxt(filename,unpack_header=True):
    """
    Read .txt format file
    Parameters
    ----------
    filename : string
        Name of jmrui .txt file
    
    Returns
    -------
    array-like
        Complex data
    list (or dict if unpack_header==True)
        Header information
    
    """
    signalRe = re.compile(r'Signal (\d{1,}) out of (\d{1,}) in file')
    headerRe = re.compile(r'(\w*):(.*)')
    header = {}
    data   = [] 
    recordData = False
    with open(filename,'r') as txtfile:
        for line in txtfile:
            headerComp = headerRe.match(line)
            if headerComp:
                value = headerComp[2].strip()                
                header.update({headerComp[1]:num(value)})

            signalIndices = signalRe.match(line)
            if signalIndices:
                recordData = True
                continue
                
            if recordData:
                data.append(list(map(float,line.split())))

    # Reshape data
    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j*data[1::2]).astype(np.complex)

    # Clean up header
    header = translateHeader(header)

    return data,header

# Translate jMRUI header to mandatory fields
def translateHeader(header):
    newHeader = {'jmrui':header}
    newHeader.update({'centralFrequency':header['TransmitterFrequency']})
    newHeader.update({'bandwidth':1/(header['SamplingInterval']*1E-3)})
    newHeader.update({'dwelltime':header['SamplingInterval']*1E-3})
    return newHeader

def num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

# Read jMRUI .txt files containing basis
def read_txtBasis_files(txtfiles):
    basis = []
    names = []
    header = []
    for file in txtfiles:
        b,h = readjMRUItxt(file)
        basis.append(b)

        prefix, _ = op.splitext(op.basename(file))
        names.append(prefix)

        header.append(h)

    basis = np.array(basis).conj().T
    return basis,names,header

# Write functions
def writejMRUItxt(fileout,FID,paramDict):
    pass