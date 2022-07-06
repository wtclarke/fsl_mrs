#!/usr/bin/env python

# results_to_spectrum - script to convert result to a full spectrum
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Carke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2021 University of Oxford
# SHBASECOPYRIGHT

import argparse
from pathlib import Path


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="FSL Magnetic Resonance Spectroscopy - convert fsl_mrs results to spectrum.")

    parser.add_argument('results_dir', type=Path,
                        help='Directory containing fsl_mrs results.')
    parser.add_argument('--export_baseline', action="store_true",
                        help="Output just baseline")
    parser.add_argument('--export_no_baseline', action="store_true",
                        help="Output fit without baseline")
    parser.add_argument('--export_separate', action="store_true",
                        help="Output individual metabolites")
    parser.add_argument('-o', '--output', type=Path,
                        help='Output directory', default='.')
    parser.add_argument('-f', '--filename', type=str,
                        help='Output file name', default='fit')
    args = parser.parse_args()

    import pandas as pd
    import json

    from fsl_mrs.utils import results, mrs_io, misc
    from fsl_mrs.utils.baseline import prepare_baseline_regressor
    from fsl_mrs.core.nifti_mrs import gen_new_nifti_mrs

    # output dir - make if it doesn't exist
    if not args.output.exists():
        args.output.mkdir(parents=True)

    # Read all_parameters.csv into pandas DF
    param_df = pd.read_csv(args.results_dir / 'all_parameters.csv')

    # Read options.txt
    with open(args.results_dir / 'options.txt', "r") as f:
        orig_args = json.loads(f.readline())

    # Load data into mrs object
    FID = mrs_io.read_FID(orig_args['data'])
    basis = mrs_io.read_basis(orig_args['basis'])

    # Instantiate MRS object
    mrs = FID.mrs(basis=basis)
    if orig_args['conjfid'] is None:
        mrs.check_FID(repair=True)
    elif orig_args['conjfid']:
        mrs.conj_FID = True
    else:
        mrs.conj_FID = False

    if orig_args['conjbasis'] is None:
        mrs.check_Basis(repair=True)
    elif orig_args['conjbasis']:
        mrs.conj_Basis = True
    else:
        mrs.conj_Basis = False

    if not orig_args['no_rescale']:
        mrs.rescaleForFitting(ind_scaling=orig_args['ind_scale'])
    mrs.keep = orig_args['keep']
    mrs.ignore = orig_args['ignore']

    if orig_args['lorentzian']:
        model = 'lorentzian'
    else:
        model = 'voigt'

    method = orig_args['algo']
    # Generate metabolite groups
    metab_groups = misc.parse_metab_groups(mrs, orig_args['metab_groups'])
    baseline_order = orig_args['baseline_order']
    if baseline_order < 0:
        baseline_order = 0  # Required to make prepare_baseline_regressor run.
    ppmlim = orig_args['ppmlim']
    # Generate baseline polynomials (B)
    B = prepare_baseline_regressor(mrs, baseline_order, ppmlim)

    # Generate results object
    print(metab_groups)
    res = results.FitRes(
        mrs,
        param_df['mean'].to_numpy(),
        model,
        method,
        metab_groups,
        baseline_order,
        B,
        ppmlim)

    if orig_args['combine'] is not None:
        res.combine(orig_args['combine'])

    data_out = res.predictedFID(mrs, mode='Full')
    data_out /= mrs.scaling['FID']
    data_out = data_out.reshape((1, 1, 1) + data_out.shape)
    out = gen_new_nifti_mrs(data_out,
                            mrs.dwellTime,
                            mrs.centralFrequency,
                            nucleus=mrs.nucleus,
                            affine=FID.voxToWorldMat)
    out.save(args.output / args.filename)

    if args.export_no_baseline:
        data_out = res.predictedFID(mrs, mode='Full', noBaseline=True)
        data_out /= mrs.scaling['FID']
        data_out = data_out.reshape((1, 1, 1) + data_out.shape)
        out = gen_new_nifti_mrs(data_out,
                                mrs.dwellTime,
                                mrs.centralFrequency,
                                nucleus=mrs.nucleus,
                                affine=FID.voxToWorldMat)
        out.save(args.output / (args.filename + '_no_baseline'))

    if args.export_baseline:
        data_out = res.predictedFID(mrs, mode='baseline')
        data_out /= mrs.scaling['FID']
        data_out = data_out.reshape((1, 1, 1) + data_out.shape)
        out = gen_new_nifti_mrs(data_out,
                                mrs.dwellTime,
                                mrs.centralFrequency,
                                nucleus=mrs.nucleus,
                                affine=FID.voxToWorldMat)
        out.save(args.output / (args.filename + '_baseline'))

    if args.export_separate:
        for metab in res.original_metabs:
            data_out = res.predictedFID(mrs, mode=metab)
            data_out /= mrs.scaling['FID']
            data_out = data_out.reshape((1, 1, 1) + data_out.shape)
            out = gen_new_nifti_mrs(data_out,
                                    mrs.dwellTime,
                                    mrs.centralFrequency,
                                    nucleus=mrs.nucleus,
                                    affine=FID.voxToWorldMat)
            out.save(args.output / (args.filename + f'_{metab}'))


if __name__ == '__main__':
    main()
