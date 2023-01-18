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
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs


def readjMRUItxt_fid(txtfile):
    '''Read FID files in jMRUI txt format
    :param txtfile: .txt format file path
    :return: NIFTI_MRS
    '''

    data, header = readjMRUItxt(txtfile)
    # Nsig, Npoints to Npoints, Nsig
    data = data.T

    if data.shape[1] > 1:
        dim_tags = ['DIM_USER_0', None, None]
    else:
        dim_tags = [None, None, None]

    if 'TypeOfNucleus' in header['jmrui']:
        nucleus = header['jmrui']['TypeOfNucleus']
        if nucleus == 0.0:
            # Not sure if this is correct interpretation of TypeOfNucleus
            nucleus = '1H'
    else:
        nucleus = '1H'

    data = data.reshape((1, 1, 1) + data.shape)

    return gen_nifti_mrs(data, header['dwelltime'], header['centralFrequency'], nucleus=nucleus, dim_tags=dim_tags)


# Read jMRUI .txt files containing basis
def read_txtBasis_files(txtfiles):
    """Read a list of files containing a jMRUI basis set

    :param txtfiles: List of files to read basis from. Can be a single file/element.
    :type txtfiles: List
    :return: Tuple of basis, names, headers
    :rtype: tuple
    """
    basis = []
    names = []
    header = []
    for file in txtfiles:
        # Special case for the VESPA information file that can be packaged in JMRUI basisets
        if op.basename(file) == 'jmrui-text_output_summary.txt':
            continue

        b, h = readjMRUItxt(file)
        basis.append(b)

        try:
            split_str = h['jmrui']['SignalNames'].split(';')
            if split_str[-1] == '':
                split_str.pop()
            names += split_str
        except KeyError:
            names.append(
                op.splitext(h['jmrui']['Filename'])[0])
        header += [h, ] * b.shape[0]
    basis = np.concatenate(basis, axis=0)
    basis = basis.conj().T

    # Add missing field that fsl expects.
    for hdr in header:
        hdr['fwhm'] = None

    # Strip any file extensions in the names.
    names = [name.replace('.txt', '') for name in names]

    return basis, names, header


# generically read jMRUI style text files
def readjMRUItxt(filename):
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
    """
    signalRe = re.compile(r'Signal (\d{1,}) out of (\d{1,}) in file')
    headerRe = re.compile(r'(\w*):(.*)')
    header = {}
    data   = []
    recordData = False
    nsig = 0
    with open(filename, 'r') as txtfile:
        for line in txtfile:
            headerComp = headerRe.match(line)
            if headerComp:
                value = headerComp[2].strip()
                header.update({headerComp[1]: num(value)})

            signalIndices = signalRe.match(line)
            if signalIndices:
                nsig += 1
                recordData = True
                continue

            if recordData:
                curr_data = line.split()
                if len(curr_data) > 2:
                    curr_data = curr_data[:2]
                data.append(list(map(float, curr_data)))

    # Reshape data
    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j * data[1::2]).astype(complex)
    data = data.reshape(nsig, -1)

    # Clean up header
    header = translateHeader(header)

    return data, header


# Translate jMRUI header to mandatory fields
def translateHeader(header):
    newHeader = {'jmrui': header}
    newHeader.update({'centralFrequency': header['TransmitterFrequency']})
    newHeader.update({'bandwidth': 1 / (header['SamplingInterval'] * 1E-3)})
    newHeader.update({'dwelltime': header['SamplingInterval'] * 1E-3})
    return newHeader


def num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# Write functions
def writejMRUItxt(fileout, FID, paramDict):

    if isinstance(FID, list):
        numFIDs = len(FID)
    else:
        numFIDs = 1
        FID = [FID]

    samplingint = paramDict['dwelltime'] * 1E3
    cf = paramDict['centralFrequency'] * 1E6
    with open(fileout, 'w') as txtfile:
        txtfile.write('jMRUI Data Textfile\n')
        txtfile.write('\n')
        txtfile.write(f'Filename: {op.basename(fileout)}\n')
        txtfile.write('\n')
        txtfile.write(f'PointsInDataset: {FID[0].shape[0]}\n')
        txtfile.write(f'DatasetsInFile: {numFIDs}\n')
        txtfile.write(f'SamplingInterval: {samplingint}\n')
        txtfile.write('ZeroOrderPhase: 0E0\n')
        txtfile.write('BeginTime: 0E0\n')
        txtfile.write(f'TransmitterFrequency: {cf}\n')
        txtfile.write('MagneticField: 0E0\n')
        txtfile.write('TypeOfNucleus: 0E0\n')
        txtfile.write('NameOfPatient: \n')
        txtfile.write('DateOfExperiment: \n')
        txtfile.write('Spectrometer: \n')
        txtfile.write('AdditionalInfo: \n')
        txtfile.write('SignalNames: {op.basename(fileout)}\n')
        txtfile.write('\n\n')
        txtfile.write('Signal and FFT\n')
        txtfile.write('sig(real)	sig(imag)\n')
        for idx, f in enumerate(FID):
            txtfile.write(f'Signal {idx} out of {numFIDs} in file\n')
            for t in f:
                txtfile.write(f'{np.real(t)}\t{np.imag(t)}\n')
            txtfile.write('\n')
