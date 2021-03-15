'''FSL-MRS test script

Tests for the quantify module.
Utilise the independently constructed MRS fitting challenge data to test against

Copyright Will Clarke, University of Oxford, 2021'''


import os.path as op
import numpy as np

import fsl_mrs.utils.mrs_io as mrsio
from fsl_mrs.utils.fitting import fit_FSLModel
import fsl_mrs.utils.quantify as quant
from fsl_mrs.utils.constants import STANDARD_T1, STANDARD_T2

metabfile = op.join(op.dirname(__file__), 'testdata/quantify/Cr_10mM_test_water_scaling_WS.txt')
h2ofile = op.join(op.dirname(__file__), 'testdata/quantify/Cr_10mM_test_water_scaling_nWS.txt')
basisfile = op.join(op.dirname(__file__), 'testdata/quantify/basisset_JMRUI')


def test_QuantificationInfo():
    qci = quant.QuantificationInfo(0.000, 40, ['Cr', 'NAA'], 298)
    assert qci.relax_corr_water_molal > 55500
    assert qci.relax_corr_water_molar > 55500

    qci = quant.QuantificationInfo(0.000, 40, ['Cr', 'NAA'], 127)
    assert qci.relax_corr_water_molal > 55500
    assert qci.relax_corr_water_molar > 55500

    qci = quant.QuantificationInfo(0.010, 3, ['Cr', 'NAA'], 127)
    t2s = STANDARD_T2['3T']
    t1s = STANDARD_T1['3T']
    assert np.isclose(qci.R_H2O_WM, np.exp(-0.010 / t2s['H2O_WM']) * (1 - np.exp(-3 / t1s['H2O_WM'])))
    assert np.isclose(qci.R_H2O_GM, np.exp(-0.010 / t2s['H2O_GM']) * (1 - np.exp(-3 / t1s['H2O_GM'])))
    assert np.isclose(qci.R_H2O_CSF, np.exp(-0.010 / t2s['H2O_CSF']) * (1 - np.exp(-3 / t1s['H2O_CSF'])))

    qci = quant.QuantificationInfo(0.010, 3, ['Cr', 'NAA'], 298)
    t2s = STANDARD_T2['7T']
    t1s = STANDARD_T1['7T']
    assert np.isclose(qci.R_H2O_WM, np.exp(-0.010 / t2s['H2O_WM']) * (1 - np.exp(-3 / t1s['H2O_WM'])))
    assert np.isclose(qci.R_H2O_GM, np.exp(-0.010 / t2s['H2O_GM']) * (1 - np.exp(-3 / t1s['H2O_GM'])))
    assert np.isclose(qci.R_H2O_CSF, np.exp(-0.010 / t2s['H2O_CSF']) * (1 - np.exp(-3 / t1s['H2O_CSF'])))

    assert qci.ref_metab == 'Cr'
    assert qci.ref_protons == 5
    assert qci.ref_limits == (2, 5)

    qci = quant.QuantificationInfo(0.010, 3, ['NAA'], 298)
    assert qci.ref_metab == 'NAA'
    assert qci.ref_protons == 3
    assert qci.ref_limits == (1.8, 2.2)

    qci.set_fractions({'GM': 0.45, 'WM': 0.45, 'CSF': 0.1})
    assert qci._fractions is not None

    assert np.isclose(qci.csf_corr, 1 / 0.9)

    qci.add_corr = 5.0
    assert qci.add_corr == 5.0


def test_quantifyWater():
    basis, names, headerb = mrsio.read_basis(basisfile)
    crIndex = names.index('Cr')
    data = mrsio.read_FID(metabfile)
    dataw = mrsio.read_FID(h2ofile)

    mrs = data.mrs(basis=basis[:, crIndex],
                   names=['Cr'],
                   basis_hdr=headerb[crIndex],
                   ref_data=dataw)
    mrs.check_FID(repair=True)
    mrs.check_Basis(repair=True)

    Fitargs = {'ppmlim': [0.2, 5.2],
               'method': 'MH',
               'baseline_order': 0,
               'metab_groups': [0]}

    res = fit_FSLModel(mrs, **Fitargs)

    tissueFractions = {'GM': 0.6, 'WM': 0.4, 'CSF': 0.0}
    TE = 0.03
    TR = 20
    T2dict = {'H2O_GM': 0.110,
              'H2O_WM': 0.080,
              'H2O_CSF': 2.55,
              'METAB': 0.160}

    q_info = quant.QuantificationInfo(
        TE,
        TR,
        mrs.names,
        mrs.centralFrequency / 1E6,
        t2=T2dict)

    q_info.set_fractions(tissueFractions)

    res.calculateConcScaling(mrs,
                             q_info,
                             internal_reference=['Cr'],
                             verbose=True)

    print(res.getConc(scaling='raw'))
    print(res.getConc(scaling='internal'))
    print(res.getConc(scaling='molality'))
    print(res.getConc(scaling='molarity'))

    assert np.allclose(res.getConc(scaling='internal'), 1.0)
    assert np.allclose(res.getConc(scaling='molarity'), 10.78, atol=1E-1)
