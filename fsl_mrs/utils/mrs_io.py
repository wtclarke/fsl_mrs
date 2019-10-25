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
               
        
