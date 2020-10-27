from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.utils.synthetic.synthetic_from_basis import syntheticFromBasisFile  
from fsl_mrs.core import MRS
from fsl_mrs.utils.fitting import fit_FSLModel
from fsl_mrs.utils import mrs_io
from pytest import fixture
import numpy as np

from pathlib import Path

testsPath = Path(__file__).parent
basis_path = testsPath / 'testdata/fsl_mrs/steam_basis'

@fixture
def data():
    noiseCov = 0.01
    amplitude = np.asarray([0.5,0.5,1.0])*10
    chemshift = np.asarray([3.0,3.05,2.0])-4.65
    lw = [10,10,10]
    phases = [0,0,0]
    g = [0,0,0]
    basisNames = ['Cr','PCr','NAA']

    basisFIDs = []
    for idx,_ in enumerate(amplitude):
        tmp,basisHdr = syntheticFID(noisecovariance=[[0.0]],
                                        chemicalshift=[chemshift[idx]],
                                        amplitude=[1.0],
                                        linewidth=[lw[idx]/5],
                                        phase=[phases[idx]],
                                        g=[g[idx]])
        basisFIDs.append(tmp[0])
    basisFIDs = np.asarray(basisFIDs)

    synFID,synHdr =  syntheticFID(noisecovariance=[[noiseCov]],
                                     chemicalshift=chemshift,
                                     amplitude=amplitude,
                                     linewidth=lw,
                                     phase=phases,
                                     g=g)

    synMRS = MRS(FID =synFID[0],header=synHdr,basis =basisFIDs,basis_hdr=basisHdr,names=basisNames)

    return synMRS,amplitude

def test_fit_FSLModel_Newton(data):

    mrs = data[0]
    amplitudes = data[1]

    metab_groups = [0]*mrs.numBasis
    Fitargs = {'ppmlim':[0.2,4.2],
            'method':'Newton','baseline_order':-1,
            'metab_groups':metab_groups}
    
    res = fit_FSLModel(mrs,**Fitargs)

    fittedconcs = res.getConc(metab = mrs.names)
    fittedRelconcs = res.getConc(scaling='internal',metab = mrs.names)
    assert np.allclose(fittedconcs,amplitudes,atol=1E-1)
    assert np.allclose(fittedRelconcs,amplitudes/(amplitudes[0]+amplitudes[1]),atol=1E-1)

def test_fit_FSLModel_lorentzian_Newton(data):

    mrs = data[0]
    amplitudes = data[1]

    metab_groups = [0]*mrs.numBasis
    Fitargs = {'ppmlim': [0.2,4.2],
               'method': 'Newton',
               'baseline_order': -1,
               'metab_groups': metab_groups,
               'model': 'lorentzian'}
    
    res = fit_FSLModel(mrs,**Fitargs)

    fittedconcs = res.getConc(metab = mrs.names)
    fittedRelconcs = res.getConc(scaling='internal',metab = mrs.names)
    assert np.allclose(fittedconcs,amplitudes,atol=1E-1)
    assert np.allclose(fittedRelconcs,amplitudes/(amplitudes[0]+amplitudes[1]),atol=1E-1)

def test_fit_FSLModel_MH(data):

    mrs = data[0]
    amplitudes = data[1]

    metab_groups = [0]*mrs.numBasis
    Fitargs = {'ppmlim':[0.2,4.2],
            'method':'MH','baseline_order':-1,
            'metab_groups':metab_groups,
            'MHSamples':100}
    
    res = fit_FSLModel(mrs,**Fitargs)

    fittedconcs = res.getConc(metab = mrs.names)
    fittedRelconcs = res.getConc(scaling='internal',metab = mrs.names)

    assert np.allclose(fittedconcs,amplitudes,atol=2E-1)
    assert np.allclose(fittedRelconcs,amplitudes/(amplitudes[0]+amplitudes[1]),atol=1E-1)

def test_fit_FSLModel_lorentzian_MH(data):

    mrs = data[0]
    amplitudes = data[1]

    metab_groups = [0]*mrs.numBasis
    Fitargs = {'ppmlim':[0.2,4.2],
               'method':'MH',
               'baseline_order':-1,
               'metab_groups':metab_groups,
               'MHSamples':100,
               'model': 'lorentzian'}
    
    res = fit_FSLModel(mrs,**Fitargs)

    fittedconcs = res.getConc(metab = mrs.names)
    fittedRelconcs = res.getConc(scaling='internal',metab = mrs.names)

    assert np.allclose(fittedconcs,amplitudes,atol=2E-1)
    assert np.allclose(fittedRelconcs,amplitudes/(amplitudes[0]+amplitudes[1]),atol=1E-1)

def test_fit_FSLModel_on_invivo_sim():

    FIDs,hdr,trueconcs = syntheticFromBasisFile(basis_path,
                                                noisecovariance=[[1E-3]],
                                                broadening=(9.0, 9.0),
                                                concentrations={'Mac':2.0})

    basis,names,header = mrs_io.read_basis(basis_path)

    mrs = MRS(FID=FIDs,header=hdr,basis=basis,basis_hdr=header[0],names=names)
    mrs.processForFitting()

    metab_groups = [0]*mrs.numBasis
    Fitargs = {'ppmlim':[0.2,4.2],
               'method':'MH',
               'baseline_order':-1,
               'metab_groups':metab_groups,
               'MHSamples':100}
    
    res = fit_FSLModel(mrs,**Fitargs)

    fittedRelconcs = res.getConc(scaling='internal',metab = mrs.names)

    answers = np.asarray(trueconcs)
    answers /= (answers[names.index('Cr')]+trueconcs[names.index('PCr')])

    assert np.allclose(fittedRelconcs,answers,atol=5E-2)