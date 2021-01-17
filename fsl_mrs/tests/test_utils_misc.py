from fsl_mrs.utils.mrs_io.main import read_FID
from fsl_mrs.utils import misc
from fsl_mrs.utils import synthetic as synth
import numpy as np
from pathlib import Path

testsPath = Path(__file__).parent
basis_path = testsPath / 'testdata/fsl_mrs/steam_basis'
data_path = testsPath / 'testdata/fsl_mrs/metab.nii.gz'


def test_ppm2hz_hz2ppm():
    cf = 300E6
    ppm = 1.0
    shift = 4.65
    assert misc.ppm2hz(cf, ppm, shift=False) == (1.0 * 300)
    assert misc.ppm2hz(cf, ppm, shift=True) == ((1.0 - shift) * 300)

    hz = 300
    assert misc.hz2ppm(cf, hz, shift=False) == 1.0
    assert misc.hz2ppm(cf, hz, shift=True) == (1.0 + shift)


def test_FIDToSpec_SpecToFID():
    testFID, hdr = synth.syntheticFID(amplitude=[10], chemicalshift=[0], phase=[0], damping=[20])

    # SVS case
    spec = misc.FIDToSpec(testFID[0])
    reformedFID = misc.SpecToFID(spec)
    assert np.allclose(reformedFID, testFID)

    testMRSI = np.tile(testFID, (4, 4, 4, 1)).T
    testspec = misc.FIDToSpec(testMRSI)
    assert np.argmax(np.abs(testspec[:, 2, 2, 2])) == 1024

    testspec = misc.FIDToSpec(testMRSI.T, axis=3)
    assert np.argmax(np.abs(testspec[2, 2, 2, :])) == 1024

    reformedFID = misc.SpecToFID(testspec, axis=3)
    assert np.allclose(reformedFID, testMRSI.T)

    reformedFID = misc.SpecToFID(testspec.T)
    assert np.allclose(reformedFID, testMRSI)

    # Odd number of points - guard against fftshift/ifftshift errors
    testFID, hdr = synth.syntheticFID(amplitude=[1], chemicalshift=[0], phase=[0], damping=[20], points=1025)
    assert np.allclose(misc.SpecToFID(misc.FIDToSpec(testFID[0])), testFID)


def test_parse_metab_groups():

    mrs = read_FID(str(data_path)).mrs(basis_file=str(basis_path))

    # combine all
    assert misc.parse_metab_groups(mrs, 'combine_all')\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # separate all
    assert misc.parse_metab_groups(mrs, 'separate_all')\
        == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # One metabolite
    assert misc.parse_metab_groups(mrs, 'Mac')\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # Two metabolites
    assert misc.parse_metab_groups(mrs, ['Mac', 'NAA'])\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]

    # Combine two
    assert misc.parse_metab_groups(mrs, 'NAA+NAAG')\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    # Combine three
    assert misc.parse_metab_groups(mrs, ['NAA+NAAG+Cr'])\
        == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    # Combine three and separate
    assert misc.parse_metab_groups(mrs, ['Mac', 'NAA+NAAG+Cr'])\
        == [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0]

    # Single integer
    assert misc.parse_metab_groups(mrs, 1)\
        == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # List of integers
    assert misc.parse_metab_groups(mrs, [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])\
        == [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
