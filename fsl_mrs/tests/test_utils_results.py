'''FSL-MRS test script

Test features of the results class

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
from fsl_mrs.utils.synthetic import syntheticFID
from fsl_mrs.utils import synthetic as syn
from fsl_mrs.core import MRS
from fsl_mrs.utils.fitting import fit_FSLModel
from pytest import fixture
import numpy as np
import json
from pathlib import Path


# Set up some synthetic data to use
@fixture(scope='module')
def data():
    noiseCov = 0.001
    amplitude = np.asarray([1.0, 0.5, 0.5, 1.0]) * 10
    chemshift = np.asarray([4.65, 3.0, 3.05, 2.0]) - 4.65
    lw = [10, 10, 10, 10]
    phases = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    basisNames = ['h2o', 'Cr', 'PCr', 'NAA']
    begintime = 0.00005

    basisFIDs = []
    basisHdr = []
    for idx, _ in enumerate(amplitude):
        tmp, hdr = syntheticFID(noisecovariance=[[0.0]],
                                chemicalshift=[chemshift[idx] + 0.1],
                                amplitude=[1.0],
                                linewidth=[lw[idx] / 5],
                                phase=[phases[idx]],
                                g=[g[idx]],
                                begintime=0)
        hdr['fwhm'] = lw[idx] / 5
        basisFIDs.append(tmp[0])
        basisHdr.append(hdr)
    basisFIDs = np.asarray(basisFIDs)

    synFID, synHdr = syntheticFID(noisecovariance=[[noiseCov]],
                                  chemicalshift=chemshift,
                                  amplitude=amplitude,
                                  linewidth=lw,
                                  phase=phases,
                                  g=g,
                                  begintime=begintime)

    synMRS = MRS(FID=synFID[0],
                 header=synHdr,
                 basis=basisFIDs,
                 basis_hdr=basisHdr,
                 names=basisNames)

    metab_groups = [0] * synMRS.numBasis
    Fitargs = {'ppmlim': [0.2, 5.2],
               'method': 'MH',
               'baseline_order': -1,
               'metab_groups': metab_groups,
               'MHSamples': 100,
               'disable_mh_priors': True}

    res = fit_FSLModel(synMRS, **Fitargs)

    return res, amplitude, synMRS


def test_peakcombination(data):

    res = data[0]
    amplitudes = data[1]

    res.combine([['Cr', 'PCr']])

    fittedconcs = res.getConc()
    fittedRelconcs = res.getConc(scaling='internal')

    amplitudes = np.append(amplitudes, amplitudes[1] + amplitudes[2])

    assert 'Cr+PCr' in res.metabs
    assert np.allclose(fittedconcs, amplitudes, atol=2E-1)
    assert np.allclose(fittedRelconcs,
                       amplitudes / (amplitudes[1] + amplitudes[2]),
                       atol=2E-1)


def test_units(data):
    res = data[0]

    # Phase
    p0, p1 = res.getPhaseParams(phi0='degrees', phi1='seconds')
    assert np.isclose(p0, 0.0, atol=1E-1)
    assert np.isclose(p1, 0.00005, atol=3E-5)

    # Shift
    shift = res.getShiftParams(units='ppm')
    shift_hz = res.getShiftParams(units='Hz')
    assert np.isclose(shift, 0.1, atol=1E-2)
    assert np.isclose(shift_hz, 0.1 * 123.0, atol=1E-1)

    # Linewidth
    lw = res.getLineShapeParams(units='Hz')[0]
    lw_ppm = res.getLineShapeParams(units='ppm')[0]
    assert np.isclose(lw, 8.0, atol=1E-1)  # 10-2
    assert np.isclose(lw_ppm, 8.0 / 123.0, atol=1E-1)


def test_qcOutput(data):
    res = data[0]
    SNR, FWHM = res.getQCParams()

    assert np.allclose(FWHM, 10.0, atol=1E0)
    assert SNR.size == 4


def test_metabs_in_groups(data):
    res = data[0]
    met_g = res.metabs_in_groups()

    assert met_g == [['h2o', 'Cr', 'PCr', 'NAA']]


def test_metabs_in_group(data):
    res = data[0]
    met_g = res.metabs_in_group(0)

    assert met_g == ['h2o', 'Cr', 'PCr', 'NAA']


def test_metab_in_group_json(data, tmp_path):
    res = data[0]
    met_g = res.metab_in_group_json()
    assert json.loads(met_g) == {'0': ['h2o', 'Cr', 'PCr', 'NAA']}

    met_g2 = res.metab_in_group_json(tmp_path / 'test.json')
    assert json.loads(met_g2) == {'0': ['h2o', 'Cr', 'PCr', 'NAA']}
    assert (tmp_path / 'test.json').is_file()
    with open(tmp_path / 'test.json') as fp:
        assert json.load(fp) == {'0': ['h2o', 'Cr', 'PCr', 'NAA']}


def test_fit_parameters_json(data, tmp_path):
    res = data[0]
    res.fit_parameters_json(tmp_path / 'params.json')
    with open(tmp_path / 'params.json') as fp:
        saved_dict = json.load(fp)
    assert saved_dict['parameters'] ==\
        ['h2o', 'Cr', 'PCr', 'NAA', 'gamma_0', 'sigma_0', 'eps_0', 'Phi0', 'Phi1', 'B_real_0', 'B_imag_0']
    assert saved_dict['parameters_inc_comb'] ==\
        ['h2o', 'Cr', 'PCr', 'NAA', 'gamma_0', 'sigma_0', 'eps_0', 'Phi0', 'Phi1', 'B_real_0', 'B_imag_0', 'Cr+PCr']
    assert saved_dict['metabolites'] == ['h2o', 'Cr', 'PCr', 'NAA']
    assert saved_dict['metabolites_inc_comb'] == ['h2o', 'Cr', 'PCr', 'NAA', 'Cr+PCr']


def test_plot_utility_method(data):
    res, _, mrs = data
    fig = res.plot(mrs)
    import matplotlib.pyplot
    assert isinstance(fig, matplotlib.pyplot.Figure)


testsPath = Path(__file__).parent
basis = testsPath / 'testdata/results/no_lw_basis'


def gen_one_spec(broadening):
    data, mrs, _ = syn.syntheticFromBasisFile(
        basis,
        concentrations={'MM20': 1},
        broadening=broadening,
        noisecovariance=[[1E-5]],
        baseline=(0, 0),
    )

    mrs.FID = data
    return mrs


def test_lorentzian_lw_estimates():
    qc_out = []
    combined_out = []
    single_out = []
    gamma_vec = np.arange(3, 60, 6, dtype=float)

    for gamma in gamma_vec:
        mrs_out = gen_one_spec((gamma, 0))
        res = mrs_out.fit(baseline_order=-1)
        single_out.append(res.getLineShapeParams()[1][0])
        combined_out.append(res.getLineShapeParams()[0][0])
        qc_out.append(res.getQCParams()[1]['fwhm_MM20'])

    gamma_vec /= np.pi

    assert np.allclose(single_out, gamma_vec, atol=1E-1)
    assert np.allclose(single_out, qc_out, atol=2E1)
    assert np.allclose(combined_out, qc_out, atol=2E1)


def test_gaussian_lw_estimates():
    qc_out = []
    combined_out = []
    single_out = []
    sigma_vec = np.arange(3, 60, 6, dtype=float)

    for sigma in sigma_vec:
        mrs_out = gen_one_spec((0, sigma))
        res = mrs_out.fit(baseline_order=-1)
        single_out.append(res.getLineShapeParams()[2][0])
        combined_out.append(res.getLineShapeParams()[0][0])
        qc_out.append(res.getQCParams()[1]['fwhm_MM20'])

    sigma_vec = 2.335 / (2 * np.pi * (np.sqrt(0.5) / sigma_vec))

    assert np.allclose(single_out, sigma_vec, atol=1E-1)
    assert np.allclose(single_out, qc_out, atol=2E1)
    assert np.allclose(combined_out, qc_out, atol=2E1)


def test_combined_lw_estimates():

    qc_out = []
    combined_out = []
    rng = np.random.default_rng(1)

    for _ in range(0, 10):
        mrs_out = gen_one_spec(rng.random(2) * 100)
        res = mrs_out.fit(baseline_order=-1)
        combined_out.append(res.getLineShapeParams()[0][0])
        qc_out.append(res.getQCParams()[1]['fwhm_MM20'])

    assert np.allclose(combined_out, qc_out, atol=2E1)
