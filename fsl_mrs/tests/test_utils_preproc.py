import fsl_mrs.utils.preproc as preproc
import fsl_mrs.utils.synthetic as syn
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.core import MRS
import numpy as np

# Test frequency shift by finding max on freq/ppm axis after a shift
def test_freqshift():
    testFID,testHdrs = syn.syntheticFID(amplitude=[0.0,1.0]) # Single peak at 3 ppm
    # Shift to 0 ppm
    dt = 1/testHdrs['inputopts']['bandwidth']
    shift = testHdrs['inputopts']['centralfrequency']*-3.0
    shiftedFID = preproc.freqshift(testFID[0],dt,shift)

    maxindex = np.argmax(np.abs(FIDToSpec(shiftedFID)))
    freqOfMax = testHdrs['faxis'][maxindex]

    assert freqOfMax < 5 and freqOfMax > -5

# Test timeshift by 1) counting points, 2) undoing 1st order phase of fid with begin time.
def test_timeshift():
    # Create data with lots of points and some begin time delay
    testFID,testHdrs = syn.syntheticFID(begintime=0.001,points=4096,noisecovariance=[[0.0]])

    assert ~(np.real(FIDToSpec(testFID))>0.0).all() # Check starting conditions

    # Reduce points and pad to remove first order phase 
    shiftedFID,_ = preproc.timeshift(testFID[0],1/testHdrs['inputopts']['bandwidth'],-0.001,0.0,samples=2048)

    assert shiftedFID.size == 2048
    assert (np.real(FIDToSpec(shiftedFID)+0.005)>0.0).all()

# Test combine_FIDs:
# Test mean by calculating mean of anti phase signals
# Test averaging using weights to zero signal
# Test coil combination against analytical Roemer eqn given known coil complex weights.
def test_combine_FIDs():
    testFIDs,testHdrs = syn.syntheticFID(noisecovariance=np.zeros((2,2)),coilamps=[1.0,1.0],coilphase=[0.0,np.pi])

    combfids = preproc.combine_FIDs(testFIDs,'mean')

    assert np.isclose(combfids,0).all()

    testFIDs,testHdrs = syn.syntheticFID(noisecovariance=np.zeros((4,4)),coilamps=[1.0,1.0,1.0,1.0],coilphase=[0.0,0.0,0.0,0.0])

    weigths = [np.exp(1j*0),np.exp(1j*np.pi/2),np.exp(1j*np.pi),np.exp(1j*3*np.pi/2)]
    combfids = preproc.combine_FIDs(testFIDs,'weighted',weights=weigths)

    assert np.isclose(combfids,0).all()

    #Generate high SNR data
    coilvar = 0.001
    noiseCov = [[coilvar,0],[0,coilvar]]
    coilamps= 0.5+np.random.rand(2)*0.4
    coilphs =  np.random.rand(2)*np.pi*2

    testFIDs,testHdrs = syn.syntheticFID(noisecovariance=noiseCov,coilamps=coilamps,coilphase=coilphs,points =4096)

    invcovMat = coilvar*np.linalg.inv(noiseCov)
    analyticalRoemer = []
    testFIDs = np.asarray(testFIDs).T
    cmplxW = coilamps * np.exp(1j*coilphs)
    for f in testFIDs:
        tmp = (f@invcovMat@cmplxW.conj())/np.sqrt(cmplxW.conj()@invcovMat@cmplxW)
        analyticalRoemer.append(tmp)
    analyticalRoemer = np.asarray(analyticalRoemer)
    
    combfid = preproc.combine_FIDs(testFIDs,'svd',do_prewhiten=True)
    # Check only the first few points otherwise the relative tolarence has to be very high.
    assert np.isclose(np.abs(combfid[:200]),np.abs(analyticalRoemer[:200]),atol=1E-4,rtol=1E-1).all()


# Test the alignment by aligning based on two sets of offset peaks with different offsets and measuring combined peak height
def test_phase_freq_align():

    peak1Shift = np.random.rand(10)*0.1
    peak1Phs = np.random.randn(10)*2*np.pi
    shiftedFIDs = []
    for s,p in zip(peak1Shift,peak1Phs):
        testFIDs,testHdrs = syn.syntheticFID(amplitude = [1,1],chemicalshift=[-2+s,3],phase=[p,0.0],points=2048,noisecovariance=[[1E-1]])
        shiftedFIDs.append(testFIDs[0])

    MRSargs = {'FID':shiftedFIDs[0],'bw':testHdrs['inputopts']['bandwidth'],'cf':testHdrs['inputopts']['centralfrequency']*1E6}
    mrs = MRS(**MRSargs)

    # Align across shifted peak
    alignedFIDs = preproc.phase_freq_align(shiftedFIDs,testHdrs['inputopts']['bandwidth'],
                                        testHdrs['inputopts']['centralfrequency']*1E6,
                                        niter=2,verbose=False,ppmlim=(-2.2,-1.7),shift=False)
 
    meanFID = preproc.combine_FIDs(alignedFIDs,'mean')
    assert np.max(np.abs(FIDToSpec(meanFID)))>0.09

    # Align across fixed peak
    alignedFIDs = preproc.phase_freq_align(shiftedFIDs,testHdrs['inputopts']['bandwidth'],
                                        testHdrs['inputopts']['centralfrequency']*1E6,
                                        niter=2,verbose=False,ppmlim=(2,4),shift=False)
 
    meanFID = preproc.combine_FIDs(alignedFIDs,'mean')
    assert np.max(np.abs(FIDToSpec(meanFID)))>0.09


# Test hlsvd by deleting the two default lorentzian peaks of the synthetic function