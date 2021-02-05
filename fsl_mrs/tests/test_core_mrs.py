'''FSL-MRS test script

Test core MRS class.

Copyright Will Clarke, University of Oxford, 2021'''


from pathlib import Path
from fsl_mrs.core import MRS, mrs_from_files
import pytest
from fsl_mrs.utils import synthetic as syn
import numpy as np
from fsl_mrs.utils.misc import FIDToSpec, hz2ppm
from fsl_mrs.utils.constants import GYRO_MAG_RATIO

# Files
testsPath = Path(__file__).parent
svs_metab = testsPath / 'testdata/fsl_mrs/metab.nii.gz'
svs_water = testsPath / 'testdata/fsl_mrs/wref.nii.gz'
svs_basis = testsPath / 'testdata/fsl_mrs/steam_basis'


@pytest.fixture
def synth_data():

    fid, hdr = syn.syntheticFID()
    hdr['json'] = {'ResonantNucleus': '1H'}

    basis_1, bhdr_1 = syn.syntheticFID(noisecovariance=[[0.0]],
                                       chemicalshift=[-2, ],
                                       amplitude=[0.1, ],
                                       damping=[5, ])

    basis_2, bhdr_2 = syn.syntheticFID(noisecovariance=[[0.0]],
                                       chemicalshift=[3, ],
                                       amplitude=[0.1, ],
                                       damping=[5, ])

    basis = np.concatenate((basis_1, basis_2))
    bheader = [bhdr_1, bhdr_2]
    names = ['ppm_2', 'ppm3']

    timeAxis = np.linspace(hdr['dwelltime'],
                           hdr['dwelltime'] * 2048,
                           2048)
    frequencyAxis = np.linspace(-hdr['bandwidth'] / 2,
                                hdr['bandwidth'] / 2,
                                2048)
    ppmAxis = hz2ppm(hdr['centralFrequency'] * 1E6,
                     frequencyAxis,
                     shift=False)
    ppmAxisShift = hz2ppm(hdr['centralFrequency'] * 1E6,
                          frequencyAxis,
                          shift=True)

    axes = {'time': timeAxis,
            'freq': frequencyAxis,
            'ppm': ppmAxis,
            'ppm_shift': ppmAxisShift}

    return fid[0], hdr, basis, names, bheader, axes


def test_load_from_file():

    mrs = mrs_from_files(str(svs_metab),
                         str(svs_basis),
                         H2O_file=str(svs_water))

    assert mrs.FID.shape == (4095,)
    assert mrs.basis.shape == (4095, 20)
    assert mrs.H2O.shape == (4095,)


def test_load(synth_data):

    fid, hdr, basis, names, bheader, axes = synth_data

    mrs = MRS(FID=fid,
              header=hdr,
              basis=basis,
              names=names,
              basis_hdr=bheader[0])

    assert mrs.FID.shape == (2048,)
    assert mrs.basis.shape == (2048, 2)
    assert mrs.numBasis == 2
    assert mrs.dwellTime == 1 / 4000
    assert mrs.centralFrequency == 123.2E6
    assert mrs.nucleus == '1H'


def test_access(synth_data):

    fid, hdr, basis, names, bheader, axes = synth_data

    mrs = MRS(FID=fid,
              header=hdr,
              basis=basis,
              names=names,
              basis_hdr=bheader[0])

    assert np.allclose(mrs.FID, fid)
    assert np.allclose(mrs.get_spec(), FIDToSpec(fid))
    assert np.allclose(mrs.basis.T, basis)

    assert np.allclose(mrs.getAxes(axis='ppmshift'), axes['ppm_shift'])
    assert np.allclose(mrs.getAxes(axis='ppm'), axes['ppm'])
    assert np.allclose(mrs.getAxes(axis='freq'), axes['freq'])
    assert np.allclose(mrs.getAxes(axis='time'), axes['time'])

    mrs.rescaleForFitting()
    assert np.allclose(mrs.get_spec() / mrs.scaling['FID'], FIDToSpec(fid))
    assert np.allclose(mrs.basis.T / mrs.scaling['basis'], basis)

    mrs.conj_Basis()
    mrs.conj_FID()
    assert np.allclose(mrs.get_spec() / mrs.scaling['FID'],
                       FIDToSpec(fid.conj()))
    assert np.allclose(mrs.basis.T / mrs.scaling['basis'], basis.conj())


def test_basis_manipulations(synth_data):

    fid, hdr, basis, names, bheader, axes = synth_data

    mrs = MRS(FID=fid,
              header=hdr,
              basis=basis,
              names=names,
              basis_hdr=bheader[0])

    assert mrs.basis.shape == (2048, 2)
    assert mrs.numBasis == 2

    mrs.keep(['ppm_2'])

    assert mrs.basis.shape == (2048, 1)
    assert mrs.numBasis == 1

    mrs.add_peak(0, 1, 'test', gamma=10, sigma=10)

    assert mrs.basis.shape == (2048, 2)
    assert mrs.numBasis == 2
    assert mrs.names == ['ppm_2', 'test']

    mrs.ignore(['test'])

    assert mrs.basis.shape == (2048, 1)
    assert mrs.numBasis == 1

    mrs.add_MM_peaks(gamma=10, sigma=10)
    assert mrs.basis.shape == (2048, 6)
    assert mrs.numBasis == 6


def test_nucleus_identification():
    rng = np.random.default_rng()
    fid = rng.standard_normal(512) + 1j * rng.standard_normal(512)

    hdr = {'centralFrequency': GYRO_MAG_RATIO['1H'] * 2.9,
           'bandwidth': 4000.0}

    mrs = MRS(FID=fid,
              header=hdr,
              nucleus='1H')

    assert mrs.nucleus == '1H'

    hdr = {'ResonantNucleus': '1H',
           'centralFrequency': GYRO_MAG_RATIO['1H'] * 2.9,
           'bandwidth': 4000.0}

    mrs = MRS(FID=fid,
              header=hdr)

    assert mrs.nucleus == '1H'

    hdr = {'json': {'ResonantNucleus': '1H'},
           'centralFrequency': GYRO_MAG_RATIO['1H'] * 2.9,
           'bandwidth': 4000.0}

    mrs = MRS(FID=fid,
              header=hdr)

    assert mrs.nucleus == '1H'

    # Test automatic identification
    for nuc in ['1H', '13C', '31P']:
        for field in [1.5, 2.894, 3.0, 7.0, 9.4, 11.755]:
            hdr = {'centralFrequency': GYRO_MAG_RATIO[nuc] * field,
                   'bandwidth': 4000.0}

            mrs = MRS(FID=fid,
                      header=hdr)

            assert mrs.nucleus == nuc
