"""Run the validation tests on the MRS fitting challenge data.

Output results to named directories

Copyright William Clarke, University of Oxford, 2022
"""

import argparse
from pathlib import Path

import pandas as pd

import fsl_mrs
from fsl_mrs.utils.example import simulated
from fsl_mrs.utils.quantify import QuantificationInfo


def _run_fit(idx, fitargs):
    mrs = simulated(idx)
    mrs.ignore = ['Ace']
    metab_groups = mrs.parse_metab_groups('Mac')
    fitargs['metab_groups'] = metab_groups
    res = mrs.fit(**fitargs)

    combinationList = [['NAA', 'NAAG'],
                       ['Glu', 'Gln'],
                       ['GPC', 'PCho'],
                       ['Cr', 'PCr'],
                       ['Glc', 'Tau'],
                       ['Ins', 'Gly']]
    res.combine(combinationList)

    TE = 0.03
    TR = 15.0
    T2dict = {
        'H2O_GM': 0.110,
        'H2O_WM': 0.080,
        'H2O_CSF': 2.55,
        'METAB': 0.160}
    tissueFractions = {'GM': 0.6, 'WM': 0.4, 'CSF': 0.0}

    qinfo = QuantificationInfo(
        TE,
        TR,
        mrs.names,
        mrs.centralFrequency / 1E6,
        t2=T2dict,
        water_ref_metab='Cr',
        water_ref_metab_protons=5,
        water_ref_metab_limits=(2, 5))
    qinfo.set_fractions(tissueFractions)

    res.calculateConcScaling(
        mrs,
        quant_info=qinfo,
        internal_reference=['Cr', 'PCr'])
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Run FSL-MRS MRS fitting challenge validation tests")

    parser.add_argument('output', type=Path, metavar='DIR',
                        help='Output dir')
    parser.add_argument('--mh_samples', type=int, default=500,
                        help='Number of MH samples (default = 500')

    args = parser.parse_args()
    to_run = 22
    results_n = []
    results_mh = []
    for idx in range(1, to_run):

        fitargs_newton = {
            'ppmlim': [0.2, 4.2],
            'method': 'Newton',
            'baseline_order': -1,
            'model': 'voigt'}

        fitargs_MH = {
            'ppmlim': [0.2, 4.2],
            'method': 'MH',
            'baseline_order': -1,
            'model': 'voigt',
            'MHSamples': args.mh_samples}

        res_n = _run_fit(idx, fitargs_newton)
        results_n.append(res_n.getConc(scaling='molarity', function=None).mean().T)

        res_mh = _run_fit(idx, fitargs_MH)
        results_mh.append(res_mh.getConc(scaling='molarity', function=None).mean().T)

    all_res_n = pd.concat(results_n, keys=range(1, to_run), axis=1)
    all_res_mh = pd.concat(results_mh, keys=range(1, to_run), axis=1)

    out_str = fsl_mrs.__version__.replace('.', '_')
    all_res_n.to_csv(args.output / 'fsl_mrs_newton' / f'{out_str}.gz')
    all_res_mh.to_csv(args.output / 'fsl_mrs_mh' / f'{out_str}.gz')

if __name__ == '__main__':
    main()
