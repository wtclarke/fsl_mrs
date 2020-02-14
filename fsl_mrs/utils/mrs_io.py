#!/usr/bin/env python

# mrs_io.py - I/O utilities for FSL_MRS
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT


import sys, os, glob
import numpy as np
import re
import scipy.signal as ss
import nibabel as nib
import json

# Generic I/O functions



def tidy(x):
    """
      removes ',' from string x
    """
    return x.lower().replace(',','')
    
def unpackHeader(header):
    """
       Extracts useful info from header into dict

       Including central frequency, dwelltime, echotime
    """
        
    
    tidy_header = dict()
    tidy_header['centralFrequency'] = None
    tidy_header['bandwidth'] = None
    tidy_header['echotime'] = None
    for line in header:
        if line.lower().find('hzpppm')>0:
            #print(tidy(line).split()[-1])
            tidy_header['centralFrequency'] = float(tidy(line).split()[-1])*1E6
        if line.lower().find('dwelltime')>0:
            tidy_header['dwelltime'] = float(tidy(line).split()[-1])
            tidy_header['bandwidth'] = 1/float(tidy(line).split()[-1])
        if line.lower().find('echot')>0:
            tidy_header['echotime'] = float(tidy(line).split()[-1])/1e3  # convert to secs
        if line.lower().find('badelt') > 0:
            tidy_header['dwelltime'] = float(tidy(line).split()[-1])
            tidy_header['bandwidth'] = 1/float(tidy(line).split()[-1])
        
            
    return tidy_header

def readLCModelRaw(filename, unpack_header=True):
    """
    Read .RAW format file
    Parameters
    ----------
    filename : string
        Name of .RAW file
    
    Returns
    -------
    array-like
        Complex data
    list (or dict if unpack_header==True)
        Header information
    
    """
    header = []
    data   = [] 
    in_header = False
    with open(filename,'r') as f:
        for line in f:
            if (line.find('$')>0):
                in_header = True
            if in_header:
                header.append(line)
            else:
                data.append(list(map(float,line.split())))
                
            if line.find('$END')>0:
                in_header = False

    # Reshape data
    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j*data[1::2]).astype(np.complex)

    # Tidy header info
    if unpack_header:
        header = unpackHeader(header)
    
    return data,header


def siv_basis_header(header):
    """
      extracts metab names and ishift
    """
    metabo = []
    shifts = []
    counter = 0
    for txt in header:
        # enter new section
        #if txt == "$BASIS":
         #   
        # end section
        #if txt == "$END":
        #    counter += 1
            
        if txt.lower().find('metabo')>0:
            if txt.lower().find('metabo_')<0:            
                content = re.search(r"'\s*([^']+?)\s*'", txt).groups()[0]
                metabo.append(content)
        if txt.lower().find('ishift') > 0:
            shifts.append(int(tidy(txt).split()[-1]))


    return metabo, shifts

def readLCModelBasis(filename,N=None,doifft=True):
    """
    Read .BASIS format file
    Parametersd
    ----------
    filename : string
        Name of .BASIS file
    
    Returns
    -------
    array-like
        Complex data
    string
        Metabolite names
    string
        Header information
    
    """
    metabo = []
    data, header = readLCModelRaw(filename, unpack_header=False)

    # extract metabolite names and shifts
    metabo, shifts = siv_basis_header(header)
    
    if len(metabo)>1:
        data = data.reshape(len(metabo),-1).T

    # apply ppm shift found in header
    for idx in range(data.shape[1]):
        data[:,idx] = np.roll(data[:,idx],-shifts[idx])
        
    # Resample if necessary? --> should not be allowed actually
    if N is not None:
        if N != data.shape[0]:
            data = ss.resample(data,N)

    # if freq domain --> turn to time domain
    if doifft:
        data = np.fft.ifft(data,axis=0)

    # deal with single metabo case
    if len(data.shape)==1:
        data = data[:,None]
        if len(metabo)==0:
            metabo = ['Unknown']

    # This will further extract dwelltime, useful if it is not matching
    # the FID
    header       = unpackHeader(header)

    
    return data, metabo, header

# Read a folder containing json files in the FLS basis style.
# Optionally allows recalculation of the FID using the stored density matrix.
# This will take longer but avoids the need for interpolation.
# It also allows for arbitrary shifting of the readout central frequency
def readFSLBasisFiles(basisFolder,readoutShift=4.65,bandwidth=None,points=None):
    if not os.path.isdir(basisFolder):
        raise ValueError(' ''basisFolder'' must be a folder containing basis json files.')
    # loop through all files in folder 
    basisfiles = sorted(glob.glob(os.path.join(basisFolder,'*.json')))
    basis,names,header = [],[],[]
    for bfile in basisfiles:        
        if bandwidth is None or points is None:
            # If simple read operation call readFSLBasis
            b, n, h  = readFSLBasis(bfile)
            basis.append(b)
            names.append(n)
            header.append(h)
        else:
            # If recalculation requested loop through files calling readAndGenFSLBasis
            b, n, h  = readAndGenFSLBasis(bfile,readoutShift,bandwidth,points)
            basis.append(b)
            names.append(n)
            header.append(h)
    basis = np.array(basis)        
    return basis,names,header

# Read the FID within the FSL basis json file. Returns equivalent outputs to the LCModel style basis files.
def readFSLBasis(filename,N=None,dofft=False):
    with open(filename,'r') as basisFile:
        jsonString = basisFile.read()
        basisFileParams = json.loads(jsonString)
        if 'basis' in basisFileParams:
            basis = basisFileParams['basis']
            
            data = np.array(basis['basis_re'])+1j*np.array(basis['basis_im'])

            if dofft: # Go to frequency domain from timedomain
                data = np.fft.fftshift(np.fft.fft(data))

            # Resample if necessary? --> should not be allowed actually
            if N is not None:
                if N != data.shape[0]:
                    data = ss.resample(data,N)

            header = {'centralFrequency':basis['basis_centre'],
                    'bandwidth':1/basis['basis_dwell'],
                    'dwelltime':basis['basis_dwell'],
                    'fwhm':basis['basis_width']}
            # header['echotime'] Not clear how to calculate this in the general case.
            
            metabo = basis['basis_name']
            
        else: #No basis information found
            raise ValueError('FSL basis file must have a ''basis'' field.')

    return data, metabo, header

# Load an FSL basis file.
# Recalculate the FID on a defined time axis.
# Relies on all fields being populated appropriately
def readAndGenFSLBasis(file,readoutShift,bandwidth,points):
    from fsl_mrs.denmatsim import utils as simutils
    with open(file,'r') as basisFile:
        jsonString = basisFile.read()
        basisFileParams = json.loads(jsonString)
    
    if 'MM' in basisFileParams:
        import fsl_mrs.utils.misc as misc
        FID, metabo, header = readFSLBasis(file)
        old_dt = 1/header['bandwidth']
        new_dt = 1/bandwidth
        FID     = misc.ts_to_ts(FID,old_dt,new_dt,points)
        return FID, metabo, header

    if 'seq' not in basisFileParams:
        raise ValueError('To recalculate the basis json must contain a seq field containing a B0 field.')
    else:
        B0 = basisFileParams['seq']['B0']
        if 'Rx_Phase' not in basisFileParams['seq']:
            rxphs = 0
        else:
            rxphs = basisFileParams['seq']['Rx_Phase']
    
    if 'spinSys' not in basisFileParams:
        raise ValueError('To recalculate the basis json must contain a spinSys field.')
    else:
        spins = basisFileParams['spinSys']

    if 'outputDensityMatrix' not in basisFileParams:
        raise ValueError('To recalculate the basis json must contain a outputDensityMatrix field.')
    else:
        p = []
        for re,im in zip(basisFileParams['outputDensityMatrix']['re'],basisFileParams['outputDensityMatrix']['im']):
            p.append(np.array(re)+1j*np.array(im))
        if len(p)==1:
            p = p[0] # deal with single spin system case
    
    lw = basisFileParams['basis']['basis_width']
    FID = simutils.FIDFromDensityMat(p,spins,B0,points,1/bandwidth,lw,offset=readoutShift,recieverPhs=rxphs)
        
    metabo = basisFileParams['basis']['basis_name']
    cf = basisFileParams['basis']['basis_centre']
    header = {'centralFrequency':basisFileParams['basis']['basis_centre'],
                    'bandwidth':bandwidth,
                    'dwelltime':1/bandwidth,
                    'fwhm':lw}

    return FID, metabo, header

def saveRAW(filename,FID,info=None):
    """
      Save .RAW file

      Parameters
      ----------
      filename : string
      FID      : array-like
      info     : dict
             Stores info[key] = value in header
    """

    header = '$NMID\n'
    if info is not None:
        for key,val in info.items():
            header += '{} = {}\n'.format(key,val)
    header += '$END'

    rFID = np.real(FID)[:,None]
    iFID = np.imag(FID)[:,None]
    

    np.savetxt(filename,
               np.append(rFID,iFID,axis=1),
               header=header)
               

def check_datatype(filename):
    try:
        garbage = nib.load(filename)   
    except:
        try:
            garbage = readLCModelRaw(filename)
        except:
            return 'Unknown'
        else:
            return 'RAW'
    else:
        return 'NIFTI'

def read_FID(filename):
    """
     Read FID file. Tries to detect type automatically

     Parameters
     ----------
     filename : str

     Returns:
     --------
     array-like (complex)
     dict (header info)
    """
    data_type = check_datatype(filename)
    if data_type == 'RAW':
        data,header = readLCModelRaw(filename)
    elif data_type == 'NIFTI':
        data,header = readNIFTI(filename)
    else:
        raise(Exception('Cannot read data format {}'.format(data_type)))
    return data,header

def read_basis_files(basisfiles,ignore=[]):
    """
     Reads basis files and extracts name of metabolite from file name
     Assumes .RAW files are FIDs (not spectra)
    """
    basis = []
    names = []
    for file in basisfiles:
        data,header = readLCModelRaw(file)
        name = os.path.splitext(os.path.split(file)[-1])[-2]
        if name not in ignore:
            names.append(name)
            basis.append(data)                
    basis = np.asarray(basis).astype(np.complex).T
    return basis,names

def read_basis(filename):
    if os.path.isfile(filename):
        data_type = check_datatype(filename)
        if data_type == 'RAW':
            basis,names,header = readLCModelBasis(filename)
        else:
            raise(Exception('Cannot read data format {}'.format(data_type)))
    elif os.path.isdir(filename):
        folder = filename
        basisfiles = sorted(glob.glob(os.path.join(folder,'*.RAW')))
        basis,names = read_basis_files(basisfiles)
        header = None
    else:
        raise(Exception('{} is neither a file nor a folder!'.format(filename)))
        
    return basis,names,header

# NIFTI I/O
def readNIFTI(datafile):
    data_hdr = nib.load(datafile)
    data = np.asanyarray(data_hdr.dataobj) 
    return data,data_hdr

def saveNIFTI(datafile,data,affine):
    img = nib.Nifti1Image(data,affine=affine)
    nib.save(img,datafile)
    return
