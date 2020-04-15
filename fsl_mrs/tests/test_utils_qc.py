from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.utils.qc import specApodise,calcQC
from fsl_mrs.utils.fitting import fit_FSLModel
from fsl_mrs.core import MRS
import numpy as np

def test_calcQC():
    # Syntetic data
    synFID,synHdr = syntheticFID(noisecovariance=[[0.1]],points=2*2048,chemicalshift=[0],amplitude=[6.0],linewidth=[10])
    synFIDNoise,synHdrNoise = syntheticFID(noisecovariance=[[0.1]],points=2*2048,chemicalshift=[0],amplitude=[0],linewidth=[10])
    basisFID,basisHdr = syntheticFID(noisecovariance=[[0.0]],points=2*2048,chemicalshift=[0],amplitude=[0.1],linewidth=[2])

    synMRS = MRS(FID =synFID[0],header=synHdr)
    synMRSNoise = MRS(FID =synFIDNoise[0],header=synHdrNoise)
    synMRSNoNoise = MRS(FID =synHdr['noiseless'],header=synHdr)
    synMRS_basis = MRS(FID =synFID[0],header=synHdr,basis =basisFID[0] ,basis_hdr=basisHdr,names=['Peak1'])

    truenoiseSD = np.sqrt(synHdrNoise['cov'][0,0])
    pureNoiseMeasured = np.std(synMRSNoise.getSpectrum())
    realnoise = np.std(np.real(synMRSNoise.getSpectrum()))
    imagNoise = np.std(np.imag(synMRSNoise.getSpectrum()))
    print(f'True cmplx noise = {truenoiseSD:0.3f}, pure noise measured = {pureNoiseMeasured:0.3f} (real/imag = {realnoise:0.3f}/{imagNoise:0.3f})')

    # Calc SNR without apodisation from the no noise and pure noise spectra
    truePeakHeight = np.max(np.real(synMRSNoNoise.getSpectrum()))
    SNR_noApod = truePeakHeight/pureNoiseMeasured
    print(f'SNR no apod: {SNR_noApod:0.1f} ({truePeakHeight:0.2e}/{pureNoiseMeasured:0.2e})')


    # Calc SNR with apodisation from the no noise and pure noise spectra
    trueLW = synHdr['inputopts']['linewidth'][0]
    trueApodSpec_Noise = specApodise(synMRSNoise,trueLW)
    apodNoise = np.std(trueApodSpec_Noise)
    trueApodSpec_noNoise = specApodise(synMRSNoNoise,trueLW)
    peakHeigtApod = np.max(np.real(trueApodSpec_noNoise))
    SNR = peakHeigtApod/apodNoise
    print(f'SNR w. apod: {SNR:0.1f} ({peakHeigtApod:0.2e}/{apodNoise:0.2e})')

    metab_groups = [0]
    Fitargs = {'ppmlim':[2.65,6.65],
            'method':'Newton','baseline_order':-1,
            'metab_groups':[0]}

    res = fit_FSLModel(synMRS_basis,**Fitargs)
    fwhm_test,SNRObj = calcQC(synMRS_basis,res,ppmlim=[2.65,6.65])
    print(f'Measured FWHM: {fwhm_test.mean().to_numpy()[0]:0.1f}')
    print(f'Measured spec SNR: {SNRObj.spectrum:0.1f}')
    print(f'Measured peak SNR: {SNRObj.peaks.mean().to_numpy()[0]:0.1f}')
    assert np.isclose(fwhm_test.mean().to_numpy(),trueLW,atol=1E0)
    assert np.isclose(SNRObj.spectrum,SNR_noApod,atol=1E1)
    assert np.isclose(SNRObj.peaks.mean().to_numpy(),SNR,atol=2E1)
