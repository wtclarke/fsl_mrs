#!/usr/bin/env python

# fmrs_stats - wrapper script to carry out group analysis of fMRS results
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
#
# Copyright (C) 2022 University of Oxford
# SHBASECOPYRIGHT

# Quick imports
from pathlib import Path

from fsl_mrs.auxiliary import configargparse
from fsl_mrs import __version__


def main():
    # Parse command-line arguments
    p = configargparse.ArgParser(
        add_config_file_help=False,
        description="FSL-MRS fMRS group level analysis Wrapper Script")

    p.add_argument('--version', action='version', version=__version__)

    required = p.add_argument_group('required arguments')
    fl_contrast_args = p.add_argument_group('First-level (single subject) contrast options')
    ga_contrast_args = p.add_argument_group('Higher-level (group) contrast options')

    optional = p.add_argument_group('additional options')

    # REQUIRED ARGUMENTS
    required.add_argument('--data',
                          required=True, nargs='+', type=Path, metavar='<FILE> or <DIRS>',
                          action=DataAction,
                          help='File containing list of results direcotries, or list of result directories.')
    required.add_argument('--output',
                          required=True, type=Path, metavar='<str>',
                          help='output folder')

    # FIRST-LEVEL CONTRAST ARGUMENTS
    fl_contrast_args.add_argument('--combine', type=str, nargs='+',
                                  action='append', metavar='METAB', default=[],
                                  help='combine listed metabolites [repeatable]')
    fl_contrast_args.add_argument('--fl-contrasts', type=Path, metavar='<FILE>',
                                  help='File defining first-level single-subject contrasts in json format.')
    fl_contrast_args.add_argument('--mean-contrasts', type=str, nargs='+',
                                  action='append', metavar='betas',
                                  help='Shortcut to specify betas to average into first-level contrasts [repeatable]')
    fl_contrast_args.add_argument('--reference-contrast', type=str,
                                  help='Divide all concentration betas and contrasts by this contrast. '
                                       'Applied after custom first level contrasts created.')

    # HIGHER-LEVEL / GROUP CONTRAST ARGUMENTS
    ga_contrast_args.add_argument('--hl-design', type=Path, metavar='<FILE>',
                                  help='FSL VEST file defining higher-level/group design matrix')
    ga_contrast_args.add_argument('--hl-contrasts', type=Path, metavar='<FILE>',
                                  help='FSL VEST file defining higher-level/group contrasts')
    ga_contrast_args.add_argument('--hl-covariance', type=Path, metavar='<FILE>',
                                  help='FSL VEST file defining higher-level/group covariance groups')
    ga_contrast_args.add_argument('--hl-contrast-names', type=str, nargs='+', default=None,
                                  help='Names assigned to each contrast specified by hl-contrasts option')
    ga_contrast_args.add_argument('--hl-ftests', type=Path, metavar='<FILE>',
                                  help='FSL VEST file defining higher-level/group f-tests')

    # ADDITIONAL OPTIONAL ARGUMENTS
    optional.add_argument('--report', action="store_true",
                          help='output html report')
    optional.add_argument('--verbose', action="store_true",
                          help='spit out verbose info')
    optional.add_argument('--overwrite', action="store_true",
                          help='overwrite existing output folder')
    optional.add('--config', required=False, is_config_file=True,
                 help='configuration file')

    # Parse command-line arguments
    args = p.parse_args()

    # ######################################################
    # DO THE IMPORTS AFTER PARSING TO SPEED UP HELP DISPLAY
    import shutil
    import warnings
    import json

    import numpy as np
    import pandas as pd

    import fsl_mrs.utils.fmrs_tools as fmrs
    from fsl.data.vest import loadVestFile
    # import datetime
    # ######################################################
    if not args.verbose:
        warnings.filterwarnings("ignore")

    # Check if output folder exists
    overwrite = args.overwrite
    if args.output.is_dir():
        if not overwrite:
            print(f"Folder '{args.output}' exists."
                  " Are you sure you want to delete it? [Y,N]")
            response = input()
            overwrite = response.upper() == "Y"

        if not overwrite:
            print('Please specify a different output folder name.')
            exit()
        else:
            shutil.rmtree(args.output)
            args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)

    # Interpret contrast arguments
    contrasts = []

    if args.fl_contrasts:
        with open(args.fl_contrasts) as fp:
            json_con = json.load(fp)
        for con in json_con:
            contrasts.append(
                fmrs.Contrast(
                    con['name'],
                    con['betas'],
                    con['scale']
                ))

    # Add any of the 'shortcut' options
    if args.mean_contrasts:
        for idx, mcon in enumerate(args.mean_contrasts):
            contrasts.append(
                fmrs.Contrast(
                    f'mean_{idx}_{"_".join(mcon)}',
                    mcon,
                    (np.ones(len(mcon)) / len(mcon)).tolist()
                ))

    # fmrs.Contrast
    if args.verbose:
        print('Identified contrasts:')
        for con in contrasts:
            print(con.name)

    # Do the work
    copes = []
    varcopes = []
    for idx, dd in enumerate(args.data):
        if args.verbose:
            print(f'Processing {str(dd)}')
        current_output = args.output / f'{idx}_{dd.name}'
        current_output.mkdir(parents=True, exist_ok=False)

        # 1. Form contrasts and peak combination
        _, _, df, _ = fmrs.create_contrasts(
            dd,
            contrasts=contrasts,
            metabolites_to_combine=args.combine,
            output_dir=current_output)

        # 2. Concentration parameter scaling
        if args.reference_contrast is not None:
            _, _, df, _ = fmrs.fmrs_internal_reference(
                current_output,
                args.reference_contrast,
                output_dir=current_output)

        copes.append(df['mean'].to_numpy())
        varcopes.append(df['sd'].pow(2).to_numpy())

    parameters = df.index
    copes = np.stack(copes)
    varcopes = np.stack(varcopes)

    # 3. Pass to the FLAMEO wrapper

    # Process second level matrices.
    if args.hl_design is not None:
        design_mat = loadVestFile(args.hl_design)
    else:
        design_mat = None

    if args.hl_contrasts is not None:
        contrast_mat = loadVestFile(args.hl_contrasts)
    else:
        contrast_mat = None

    if args.hl_covariance is not None:
        covariance_mat = loadVestFile(args.hl_covariance)
    else:
        covariance_mat = None

    if args.hl_ftests is not None:
        ftests_mat = loadVestFile(args.hl_ftests)
    else:
        ftests_mat = None

    p, z, out_cope, out_varcope, f = fmrs.flameo_wrapper(
        copes,
        varcopes,
        design_mat=design_mat,
        contrast_mat=contrast_mat,
        covariance_mat=covariance_mat,
        ftests=ftests_mat,
        verbose=args.verbose)

    # Save main results
    # 1. Form output dataframe
    # Squeeze if only one contrast, otherwise form a multi-level index on the columns
    out_cope = out_cope.squeeze()
    out_varcope = out_varcope.squeeze()
    z = z.squeeze()
    p = p.squeeze()
    if out_cope.ndim == 1:
        group_stats = pd.DataFrame(
            [out_cope, out_varcope, z, p],
            index=['COPE', 'VARCOPE', 'z', 'p'],
            columns=parameters).T
    else:
        all_stats = np.stack([out_cope, out_varcope, z, p])
        if args.hl_contrast_names is None:
            columns = pd.MultiIndex.from_product(
                (['COPE', 'VARCOPE', 'z', 'p'],
                 list(range(all_stats.shape[1]))),
                names=['Statistics', 'Contrast'])
        else:
            columns = pd.MultiIndex.from_product(
                (['COPE', 'VARCOPE', 'z', 'p'],
                 args.hl_contrast_names),
                names=['Statistics', 'Contrast'])
        group_stats = pd.DataFrame(all_stats.reshape(-1, all_stats.shape[-1]), index=columns, columns=parameters).T

    # 2. Save to output folder
    group_stats.to_csv(args.output / 'group_stats.csv')

    # Create interactive HTML report
    if args.report:
        print('TO DO!')

    if args.verbose:
        print('\n\n\nDone.')


class DataAction(configargparse.Action):
    """Sort out data argument. Should return list of directories as pathlib Path objs"""
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1:
            dirs = []
            with open(values[0], 'r') as file:
                for line in file:
                    dirs.append(Path(line.rstrip()))
            setattr(namespace, self.dest, dirs)
        else:
            setattr(namespace, self.dest, values[0])


if __name__ == '__main__':
    main()
