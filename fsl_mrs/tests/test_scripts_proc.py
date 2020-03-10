# Tests for the individual proc functions. These tests don't test theat the actual algorithms are doing the right thing, simply that the script handles SVS data and MRSI data properly and that the results from the command line program matches that of the underlying algorithms in preproc.py

import pytest
import os.path as op
from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.utils.mrs_io import fsl_io
from fsl_mrs.utils import preproc
import numpy as np
import subprocess
# construct some test data using synth
@pytest.fixture
def svs_data(tmp_path):    
    reps = 3
    noiseconv = 0.1*np.eye(reps)
    coilamps = np.ones(reps)
    coilphs = np.zeros(reps)     
    FID,hdr = syntheticFID(noisecovariance = noiseconv,coilamps=coilamps,coilphase=coilphs)

    testFile = []
    data = []
    for idx,f in enumerate(FID):
        testname = f'svsdata_{idx}.nii'
        testFile.append(op.join(tmp_path,testname))

        affine = np.eye(4)
        data.append(f)
        fsl_io.saveNIFTI(testFile[idx],data[idx],hdr,affine=affine)

    return testFile,data

@pytest.fixture
def mrsi_data(tmp_path):    
    reps = 3
    noiseconv = 0.1*np.eye(reps)
    coilamps = np.ones(reps)
    coilphs = np.zeros(reps)     
    FID,hdr = syntheticFID(noisecovariance = noiseconv,coilamps=coilamps,coilphase=coilphs)

    testFile = []
    data = []
    for idx,f in enumerate(FID):
        testname = f'mrsidata_{idx}.nii'
        testFile.append(op.join(tmp_path,testname)) 

        affine = np.eye(4)
        data.append(np.tile(f,(3,3,3,1)))
        fsl_io.saveNIFTI(testFile[idx],data[idx],hdr,affine=affine)

    return testFile,data


@pytest.fixture
def svs_data_uncomb(tmp_path):    
    coils = 4
    noiseconv = 0.1*np.eye(coils)
    coilamps = np.random.randn(coils)
    coilphs = np.random.random(coils)*2*np.pi     
    FID,hdr = syntheticFID(noisecovariance = noiseconv,coilamps=coilamps,coilphase=coilphs)

    testname = 'svsdata_uncomb.nii'
    testFile = op.join(tmp_path,testname)

    affine = np.eye(4)
    data = np.tile(np.asarray(FID).T,(1,1,1,1,1))
    fsl_io.saveNIFTI(testFile,data,hdr,affine=affine)

    return testFile,data

@pytest.fixture
def mrsi_data_uncomb(tmp_path):
    coils = 4
    noiseconv = 0.1*np.eye(coils)
    coilamps = np.random.randn(coils)
    coilphs = np.random.random(coils)*2*np.pi     
    FID,hdr = syntheticFID(noisecovariance = noiseconv,coilamps=coilamps,coilphase=coilphs)

    testname = 'mrsidata_uncomb.nii'
    testFile = op.join(tmp_path,testname)

    affine = np.eye(4)
    data = np.tile(np.asarray(FID).T,(3,3,3,1,1))
    fsl_io.saveNIFTI(testFile,data,hdr,affine=affine)

    return testFile,data

def splitdata(svs,mrsi):
    return svs[0],mrsi[0],svs[1],mrsi[1]

def test_filecreation(svs_data,mrsi_data,svs_data_uncomb,mrsi_data_uncomb):
    svsfile,mrsifile,svsdata,mrsidata = splitdata(svs_data,mrsi_data)

    data,hdr = fsl_io.readNIFTI(svsfile[0],squeezeSVS=False)
    assert data.shape == (1,1,1,2048)
    assert np.isclose(data,svsdata[0]).all()

    data,hdr = fsl_io.readNIFTI(mrsifile[0],squeezeSVS=False)
    assert data.shape == (3,3,3,2048)
    assert np.isclose(data,mrsidata[0]).all()

    svsfile,mrsifile,svsdata,mrsidata = splitdata(svs_data_uncomb,mrsi_data_uncomb)

    data,hdr = fsl_io.readNIFTI(svsfile,squeezeSVS=False)
    assert data.shape == (1,1,1,2048,4)
    assert np.isclose(data,svsdata).all()

    data,hdr = fsl_io.readNIFTI(mrsifile,squeezeSVS=False)
    assert data.shape == (3,3,3,2048,4)
    assert np.isclose(data,mrsidata).all()


def test_coilcombine(svs_data_uncomb,mrsi_data_uncomb,tmp_path):
    svsfile,mrsifile,svsdata,mrsidata = splitdata(svs_data_uncomb,mrsi_data_uncomb)

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','coilcombine','--file',svsfile])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    directRun = preproc.combine_FIDs(svsdata[0,0,0,...],'svd',do_prewhiten=True)

    assert np.isclose(data,directRun).all()

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','coilcombine','--file',mrsifile])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    directRun = preproc.combine_FIDs(mrsidata[2,2,2,...],'svd',do_prewhiten=True)

    assert np.isclose(data[2,2,2,...],directRun).all()
    
def test_average(svs_data,mrsi_data,tmp_path):
    svsfile,mrsifile,svsdata,mrsidata = splitdata(svs_data,mrsi_data)

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','average','--file',svsfile[0],svsfile[1],svsfile[2],'--avgfiles'])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    allFileData = np.array([d for d in svsdata])
    directRun = preproc.combine_FIDs(allFileData.T,'mean').T

    assert np.isclose(data,directRun).all()

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','average','--file',mrsifile[0],mrsifile[1],mrsifile[2],'--avgfiles'])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    allFileData = np.array([d for d in mrsidata])
    directRun = preproc.combine_FIDs(allFileData.T,'mean').T

    assert np.isclose(data,directRun).all()

def test_align(svs_data,mrsi_data,tmp_path):
    svsfile,mrsifile,svsdata,mrsidata = splitdata(svs_data,mrsi_data)

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','align','--file',svsfile[0],svsfile[1],svsfile[2],'--ppm','-10','10'])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp_000.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    allFileData =[d for d in svsdata]
    directRun = preproc.phase_freq_align(allFileData,4000,123E6,niter=2,ppmlim=[-10.0,10.0],verbose=False,target=None)

    assert np.isclose(data,directRun[0]).all()

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','align','--file',mrsifile[0],mrsifile[1],mrsifile[2],'--ppm','-10','10'])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp_000.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    allFileData = [d[2,2,2,...] for d in mrsidata]
    directRun = preproc.phase_freq_align(allFileData,4000,123E6,niter=2,ppmlim=[-10.0,10.0],verbose=False,target=None)

    assert np.isclose(data[2,2,2,...],directRun[0],atol=1E-3,rtol=1E-3).all()

def test_ecc(svs_data,mrsi_data,tmp_path):
    svsfile,mrsifile,svsdata,mrsidata = splitdata(svs_data,mrsi_data)

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','ecc','--file',svsfile[0],'--reference',svsfile[1]])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    directRun = preproc.eddy_correct(svsdata[0],svsdata[1])

    assert np.isclose(data,directRun).all()

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','ecc','--file',mrsifile[0],'--reference',mrsifile[1]])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    directRun = preproc.eddy_correct(mrsidata[0][2,2,2,...],mrsidata[1][2,2,2,...])

    assert np.isclose(data[2,2,2,...],directRun).all()

def test_remove(svs_data,mrsi_data,tmp_path):
    svsfile,mrsifile,svsdata,mrsidata = splitdata(svs_data,mrsi_data)

    # Run remove on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','remove','--file',svsfile[0],'--ppm','-10','10'])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    freqlimits = (np.array((-10,10))-4.65)*123
    directRun = preproc.hlsvd(svsdata[0],1/4000,freqlimits)

    assert np.isclose(data,directRun).all()

    # Run coil combination on both sets of data using the command line
    subprocess.call(['fsl_mrs_proc','--output',tmp_path,'--filename','tmp','remove','--file',mrsifile[0],'--ppm','-10','10'])

    # Load result for comparison
    data,hdr = fsl_io.readNIFTI(op.join(tmp_path,'tmp.nii.gz'),squeezeSVS=True)

    # Run using preproc.py directly
    freqlimits = (np.array((-10,10))-4.65)*123
    directRun = preproc.hlsvd(mrsidata[0][2,2,2,...],1/4000,freqlimits)

    assert np.isclose(data[2,2,2,...],directRun).all()

