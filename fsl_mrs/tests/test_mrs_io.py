# Test the loading functions 
import fsl_mrs.utils.mrs_io as mrsio
from fsl_mrs.utils import plotting
import numpy as np
import os.path as op

testsPath = op.dirname(__file__)
SVSTestData = {'nifti':op.join(testsPath,'testdata/mrs_io/metab.nii'),
                'raw':op.join(testsPath,'testdata/mrs_io/metab.RAW'),
                'txt':op.join(testsPath,'testdata/mrs_io/metab.txt')}

headerReqFields = ['centralFrequency','bandwidth','dwelltime']

def test_read_FID_SVS():
    # Test the loading of the three types of data we handle for SVS data
    # nifti + json
    # .raw
    # .txt

    data_nifti,header_nifti = mrsio.read_FID(SVSTestData['nifti'])
    data_raw,header_raw = mrsio.read_FID(SVSTestData['raw'])
    data_txt,header_txt = mrsio.read_FID(SVSTestData['txt'])
    
    data_raw = data_raw.conj()

    # Check that the data from each of these matches - it should they are all the same bit of data.
    datamean = np.mean([data_nifti,data_raw,data_txt],axis=0)

    assert np.isclose(data_nifti,datamean).all()
    assert np.isclose(data_raw,datamean).all()
    assert np.isclose(data_txt,datamean).all()

    # Check that the headers each contain the required fields
    for r in headerReqFields:
        assert r in header_nifti
        assert r in header_raw
        assert r in header_txt
        
        headerMean = np.mean([header_nifti[r],header_raw[r],header_txt[r]])
        assert np.isclose(header_nifti[r],headerMean)
        assert np.isclose(header_raw[r],headerMean)
        assert np.isclose(header_txt[r],headerMean)

# TODO: Make MRSI test function (and find data)
# def test_read_FID_MRSI()

BasisTestData = {'fsl':op.join(testsPath,'testdata/mrs_io/basisset_FSL'),
                'raw':op.join(testsPath,'testdata/mrs_io/basisset_LCModel_raw'),
                'txt':op.join(testsPath,'testdata/mrs_io/basisset_JMRUI'),
                'lcm':op.join(testsPath,'testdata/mrs_io/basisset_LCModel.BASIS')}
def test_read_Basis():
    # Test the loading of the four types of data we handle for basis specta
    # fsl_mrs - folder of json
    # lcmodel - .basis file
    # lcmodel - folder of .raw
    # jmrui - folder of .txt

    basis_fsl,names_fsl,headers_fsl = mrsio.read_basis(BasisTestData['fsl'])
    basis_raw,names_raw,headers_raw = mrsio.read_basis(BasisTestData['raw'])
    basis_txt,names_txt,headers_txt = mrsio.read_basis(BasisTestData['txt'])
    basis_lcm,names_lcm,headers_lcm = mrsio.read_basis(BasisTestData['lcm'])

    #lcm basis file is zeropadded by a factor of 2 remove
    basis_lcm = basis_lcm[:2048,:]

    # Test that all contain the same amount of data.
    expectedDataSize = (2048, 21)
    assert basis_fsl.shape == expectedDataSize
    assert basis_raw.shape == expectedDataSize
    assert basis_txt.shape == expectedDataSize
    assert basis_lcm.shape == expectedDataSize
    
    # Test that the number of names match the amount of data
    numNames = 21
    assert len(names_fsl)==numNames
    assert len(names_raw)==numNames
    assert len(names_txt)==numNames
    assert len(names_lcm)==numNames
    
    # Check that the headers each contain the required fields
    # Exclude raw, we know it doesn't contain everything
    for r in headerReqFields:
        assert r in headers_fsl[0]        
        assert r in headers_txt[0]
        assert r in headers_lcm[0]
        
        headerMean = np.mean([headers_fsl[0][r],headers_txt[0][r],headers_lcm[0][r]])
        if r == 'centralFrequency':
            assert np.isclose(headers_fsl[0][r],headerMean,rtol=2e-01, atol=1e05)
            assert np.isclose(headers_txt[0][r],headerMean,rtol=2e-01, atol=1e05)
            assert np.isclose(headers_lcm[0][r],headerMean,rtol=2e-01, atol=1e05)
        else:
            assert np.isclose(headers_fsl[0][r],headerMean)
            assert np.isclose(headers_txt[0][r],headerMean)
            assert np.isclose(headers_lcm[0][r],headerMean)
    
    
    # Test that all contain roughly the same data when scaled.
    metabToCheck = 'Cr'
    checkIdx = names_raw.index('Cr')
    normAbsSpec = lambda spec:np.abs(spec)/np.max(np.abs(spec))
    convertToLimitedSpec = lambda fid: normAbsSpec(plotting.FID2Spec(fid)[900:1000])
    meanSpec = np.mean([convertToLimitedSpec(basis_fsl[:,checkIdx]),
                       convertToLimitedSpec(basis_raw[:,checkIdx]),
                       convertToLimitedSpec(basis_txt[:,checkIdx]),
                       convertToLimitedSpec(basis_lcm[:,checkIdx])],axis=0)
    
    assert np.isclose(convertToLimitedSpec(basis_fsl[:,checkIdx]),meanSpec,rtol=2e-01, atol=1e-03).all()
    assert np.isclose(convertToLimitedSpec(basis_raw[:,checkIdx]),meanSpec,rtol=2e-01, atol=1e-03).all
    assert np.isclose(convertToLimitedSpec(basis_txt[:,checkIdx]),meanSpec,rtol=2e-01, atol=1e-03).all
    assert np.isclose(convertToLimitedSpec(basis_lcm[:,checkIdx]),meanSpec,rtol=2e-01, atol=1e-03).all

def test_fslBasisRegen():
    basis_fsl,names_fsl,headers_fsl = mrsio.read_basis(BasisTestData['fsl'])
    basis_fsl2,names_fsl2,headers_fsl2 = mrsio.read_basis(BasisTestData['fsl'])
