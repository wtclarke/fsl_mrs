from fsl_mrs.utils import misc
from fsl_mrs.utils import synthetic as synth
import numpy as np

def test_ppm2hz_hz2ppm():
    cf = 300E6
    ppm = 1.0
    shift = 4.65
    assert misc.ppm2hz(cf,ppm,shift=False)==(1.0*300)
    assert misc.ppm2hz(cf,ppm,shift=True)==((1.0-shift)*300)

    hz = 300
    assert misc.hz2ppm(cf,hz,shift=False)==1.0
    assert misc.hz2ppm(cf,hz,shift=True)==(1.0+shift)

def test_FIDToSpec_SpecToFID():
    testFID,hdr = synth.syntheticFID(amplitude=[10],chemicalshift=[0],phase=[0],damping=[20])
    
    # SVS case
    spec = misc.FIDToSpec(testFID[0])
    reformedFID = misc.SpecToFID(spec)
    assert np.allclose(reformedFID,testFID)

    testMRSI = np.tile(testFID,(4,4,4,1)).T
    testspec = misc.FIDToSpec(testMRSI)
    assert np.argmax(np.abs(testspec[:,2,2,2]))==1024

    testspec = misc.FIDToSpec(testMRSI.T,axis=3)
    assert np.argmax(np.abs(testspec[2,2,2,:]))==1024

    reformedFID = misc.SpecToFID(testspec,axis=3)
    assert np.allclose(reformedFID,testMRSI.T)

    reformedFID = misc.SpecToFID(testspec.T)
    assert np.allclose(reformedFID,testMRSI)

    # Odd number of points - guard against fftshift/ifftshift errors
    testFID,hdr = synth.syntheticFID(amplitude=[1],chemicalshift=[0],phase=[0],damping=[20],points=1025)
    assert np.allclose(misc.SpecToFID(misc.FIDToSpec(testFID[0])),testFID)
