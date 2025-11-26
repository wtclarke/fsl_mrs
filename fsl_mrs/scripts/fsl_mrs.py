#!/usr/bin/env python

# fsl_mrs - wrapper script for MRS fitting
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

# Quick imports
# NOTE!!!! THERE ARE MORE IMPORTS IN THE CODE BELOW (AFTER ARGPARSING)
from pathlib import Path

from fsl_mrs.auxiliary import configargparse

from fsl_mrs import __version__
from fsl_mrs.utils.splash import splash


class FSLMRSException(Exception):
    """Exception class for issues in the FSL-MRS script"""
    pass


class QuantificationException(FSLMRSException):
    """Exception class for issues arising from the quantification steps"""
    def __init__(self, in_msg, *args, **kwargs):
        msg = f'There has been an error in the FSL-MRS quantification step: {in_msg}'
        super().__init__(msg, *args, **kwargs)


def main():
    # Parse command-line arguments
    p = configargparse.ArgParser(
        add_config_file_help=False,
        description="FSL Magnetic Resonance Spectroscopy Wrapper Script")

    # utility for hiding certain arguments
    def hide_args(arglist):
        for action in arglist:
            action.help = p.SUPPRESS

    p.add_argument('-v', '--version', action='version', version=__version__)

    required = p.add_argument_group('required arguments')
    fitting_args = p.add_argument_group('fitting options')
    optional = p.add_argument_group('additional options')

    # REQUIRED ARGUMENTS
    required.add_argument('--data',
                          required=True, type=str, metavar='<str>',
                          help='input FID file')
    required.add_argument('--basis',
                          required=True, type=str, metavar='<str>',
                          help='.BASIS file or folder containing basis spectra'
                               '(will read all files within)')
    required.add_argument('--output',
                          required=True, type=Path, metavar='<str>',
                          help='output folder')

    # FITTING ARGUMENTS
    fitting_args.add_argument('--algo', default='Newton', type=str,
                              help='algorithm [Newton (fast, default)'
                                   ' or MH (slow)]')
    fitting_args.add_argument('--ignore', type=str, nargs='+',
                              metavar='METAB',
                              help='ignore certain metabolites [repeatable]')
    fitting_args.add_argument('--keep', type=str, nargs='+', metavar='METAB',
                              help='only keep these metabolites')
    fitting_args.add_argument('--combine', type=str, nargs='+',
                              action='append', metavar='METAB',
                              help='combine certain metabolites [repeatable]')
    fitting_args.add_argument('--ppmlim', default=None, type=float,
                              nargs=2, metavar=('LOW', 'HIGH'),
                              help='limit the fit optimisation to a chemical shift range. '
                                   'Defaults to a nucleus-specific range. '
                                   'For 1H default=(.2,4.2).')
    fitting_args.add_argument('--h2o', default=None, type=str, metavar='H2O',
                              help='input .H2O file for quantification')
    fitting_args.add_argument('--baseline',
                              type=str,
                              default='poly, 2',
                              help='Select baseline type. '
                                   'Options: "off", "polynomial", or "spline". '
                                   'With "polynomial" specify an order, e.g. "polynomial, 2". '
                                   'With spline specify a stiffness: '
                                   "'very-stiff', 'stiff', 'moderate', 'flexible', and 'very-flexible', "
                                   "or an effective dimension (2 -> inf) where 2 is the stiffest. "
                                   "The default is 'polynomial, 2'.")
    #     'LEGACY OPTION use --baseline instead:'
    #    ' order of baseline polynomial'
    #    ' (default=2, -1 disables)'
    fitting_args.add_argument('--baseline_order',
                              type=int,
                              default=None,
                              metavar=('ORDER'),
                              help=configargparse.SUPPRESS)
    fitting_args.add_argument('--metab_groups', default=0, nargs='+',
                              type=str_or_int_arg,
                              help='metabolite groups: list of groups'
                                   ' or list of names for indept groups.')
    fitting_args.add_argument('--lorentzian', action="store_true",
                              help='Enable purely lorentzian broadening'
                                   ' (default is Voigt)')
    fitting_args.add_argument('--free_shift', action="store_true",
                              help='Enable free frequency shifting of all metabolites.')
    fitting_args.add_argument('--ind_scale', default=None, type=str,
                              nargs='+',
                              help='List of basis spectra to scale'
                                   ' independently of other basis spectra.')
    fitting_args.add_argument('--disable_MH_priors', action="store_true",
                              help="Disable MH priors.")
    fitting_args.add_argument('--mh_samples', type=int, default=500,
                              help="Number of Metropolis Hastings samples,"
                                   " every tenth sample is kept."
                                   " Default = 500")

    # ADDITIONAL OPTIONAL ARGUMENTS
    optional.add_argument('--t1', type=str, default=None, metavar='IMAGE',
                          help='structural image (for report)')
    optional.add_argument('--TE', type=float, default=None, metavar='TE',
                          help='Echo time for relaxation correction (ms)')
    optional.add_argument('--TR', type=float, default=None, metavar='TR',
                          help='Repetition time for relaxation correction (s)')
    optional.add_argument('--tissue_frac', type=tissue_frac_arg,
                          action=TissueFracAction, nargs='+',
                          default=None, metavar='WM GM CSF OR .json OR quantification.csv',
                          help='Fractional tissue volumes for WM, GM, CSF '
                               'or path to json segmentation file '
                               'or path to a previously generated '
                               'quantification_info.csv file. '
                               'Defaults to pure water scaling.')
    optional.add_argument('--t1-values',
                          type=t1_arg,
                          default=None,
                          help='Manually specify T1 values. '
                               'Must be a path to a .json formatted file '
                               'containing T1 values with fields H2O_WM, H2O_GM, '
                               'H2O_CSF and METAB; or path to a previously generated '
                               'quantification_info.csv file.')
    optional.add_argument('--t2-values',
                          type=t2_arg,
                          default=None,
                          help='Manually specify T2 values. '
                               'Must be a path to a .json formatted file '
                               'containing T2 values with fields H2O_WM, H2O_GM, '
                               'H2O_CSF and METAB; or path to a previously generated '
                               'quantification_info.csv file.')
    optional.add_argument('--internal_ref', type=str, default=['Cr', 'PCr'],
                          nargs='+',
                          help='Metabolite(s) used as an internal reference.'
                               ' Defaults to tCr (Cr+PCr).')
    optional.add_argument('--wref_metabolite', type=str, default=None,
                          nargs='+',
                          help='Metabolite(s) used as an the reference for water scaling.'
                               ' Uses internal defaults otherwise.')
    optional.add_argument('--ref_protons', type=int, default=None,
                          help='Number of protons that reference metabolite is equivalent to.'
                               ' No effect without setting --wref_metabolite.')
    optional.add_argument('--ref_int_limits', type=float, default=None, nargs=2,
                          help='Reference spectrum integration limits (low, high).'
                               ' No effect without setting --wref_metabolite.')
    optional.add_argument('--h2o_scale', type=float, default=1.0,
                          help='Additional scaling modifier for external water referencing.')
    optional.add_argument('--report', action="store_true",
                          help='output html report')
    optional.add_argument('--verbose', action="store_true",
                          help='spit out verbose info')
    optional.add_argument('--overwrite', action="store_true",
                          help='overwrite existing output folder')
    optional.add_argument('--conj_fid', dest='conjfid', action="store_true",
                          help='Force conjugation of FID')
    optional.add_argument('--no_conj_fid', dest='conjfid',
                          action="store_false",
                          help='Forbid automatic conjugation of FID')
    optional.add_argument('--conj_basis', dest='conjbasis',
                          action="store_true",
                          help='Force conjugation of basis')
    optional.add_argument('--no_conj_basis', dest='conjbasis',
                          action="store_false",
                          help='Forbid automatic conjugation of basis')
    optional.set_defaults(conjfid=None, conjbasis=None)
    optional.add_argument('--no_rescale', action="store_true",
                          help='Forbid rescaling of FID/basis/H2O.')
    optional.add_argument('--export_baseline', action="store_true",
                          help="Output just baseline")
    optional.add_argument('--export_no_baseline', action="store_true",
                          help="Output fit without baseline")
    optional.add_argument('--export_separate', action="store_true",
                          help="Output individual metabolites")
    optional.add_argument('-f', '--filename', type=str,
                          help='Output file name', default='fit')
    optional.add('--config', required=False, is_config_file=True,
                 help='configuration file')

    # Parse command-line arguments
    args = p.parse_args()

    # Output kickass splash screen
    if args.verbose:
        splash(logo='mrs')

    # ######################################################
    # DO THE IMPORTS AFTER PARSING TO SPEED UP HELP DISPLAY
    import time
    import json
    import warnings
    import re
    import matplotlib
    matplotlib.use('agg')
    from fsl_mrs.utils import mrs_io
    from fsl_mrs.utils import report
    from fsl_mrs.utils import fitting
    from fsl_mrs.utils import plotting
    from fsl_mrs.utils import misc
    from fsl_mrs.utils import quantify
    from fsl_mrs.scripts import make_output_folder
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
    import datetime
    # ######################################################
    # Output
    if not args.verbose:
        warnings.filterwarnings("ignore")

    def verboseprint(x: str):
        if args.verbose:
            print(x)

    # Check if output folder exists
    make_output_folder(args.output, args.overwrite)

    # Create symlinks to original data (data, reference) and basis in output location
    '''Links are relative and should provide a route back to the data not relying on the
    possible relative, possible absolute paths stored from the command line.
    '''
    misc.create_rel_symlink(args.data, args.output, 'data')
    misc.create_rel_symlink(args.basis, args.output, 'basis')
    if args.h2o is not None:
        misc.create_rel_symlink(args.h2o, args.output, 'h2o')

    # Save chosen arguments
    with open(args.output / "options.txt", "w") as f:
        # Deal with any path objects
        f.write(json.dumps(vars(args), default=str))
        f.write("\n--------\n")
        f.write(p.format_values())

    # Do the work

    # Read data/h2o/basis
    verboseprint('--->> Read input data and basis\n')
    verboseprint(f'  {args.data}')
    verboseprint(f'  {args.basis}\n')

    FID = mrs_io.read_FID(args.data)
    basis = mrs_io.read_basis(args.basis)

    if args.h2o is not None:
        H2O = mrs_io.read_FID(args.h2o)
    else:
        H2O = None

    # Check for default MM and appropriate use of metabolite groups
    default_mm_name = re.compile(r'MM\d{2}')
    default_mm_matches = list(filter(default_mm_name.match, basis.names))
    if args.metab_groups == 0:
        default_mm_mgroups = []
    else:
        default_mm_mgroups = list(filter(default_mm_name.match, args.metab_groups))
    if len(default_mm_matches) > 0\
            and len(default_mm_mgroups) != len(default_mm_matches):
        print(
            f'Default macromolecules ({", ".join(default_mm_matches)}) are present in the basis set. '
            'However they are not all listed in the --metab_groups. '
            'It is recommended that all default MM are assigned their own group. '
            f'E.g. Use --metab_groups {" ".join(default_mm_matches)}')

    # Instantiate MRS object
    mrs = FID.mrs(basis=basis,
                  ref_data=H2O)

    if isinstance(mrs, list):
        raise FSLMRSException(
            'fsl_mrs only handles a single FID at a time. '
            'Please preprocess data first.')

    # Check the FID and basis / conjugate
    if args.conjfid is not None:
        if args.conjfid:
            mrs.conj_FID = True
    else:
        conjugated = mrs.check_FID(repair=True)
        if args.verbose and conjugated == 1:
            warnings.warn(
                'FID has been checked and conjugated. Please check!',
                UserWarning)

    if args.conjbasis is not None:
        if args.conjbasis:
            mrs.conj_Basis = True
    else:
        conjugated = mrs.check_Basis(repair=True)
        if args.verbose and conjugated == 1:
            warnings.warn(
                'Basis has been checked and conjugated. Please check!',
                UserWarning)

    # Rescale FID, H2O and basis to have nice range
    if not args.no_rescale:
        mrs.rescaleForFitting(ind_scaling=args.ind_scale)

    # Keep/Ignore metabolites
    mrs.keep = args.keep
    mrs.ignore = args.ignore

    # Do the fitting here
    verboseprint('--->> Start fitting\n\n')
    verboseprint('    Algorithm = [{}]\n'.format(args.algo))

    if args.mh_samples >= 1000:
        fit_baseline_mh = True
        verboseprint('Number of mh_samples set to 1000 or more, fitting baseline with MH.')
    else:
        fit_baseline_mh = False
        verboseprint('Number of MH Samples set to 999 or fewer, baseline estimated with Newton init.')

    # Initialise fitting arguments and parse metabolite groups
    Fitargs = {
        'ppmlim': args.ppmlim,
        'method': args.algo,
        'metab_groups': misc.parse_metab_groups(mrs, args.metab_groups),
        'disable_mh_priors': args.disable_MH_priors,
        'MHSamples': args.mh_samples,
        'fit_baseline_mh': fit_baseline_mh}

    if args.baseline_order:
        Fitargs['baseline_order'] = args.baseline_order
    else:
        Fitargs['baseline'] = args.baseline

    # Choose fitting lineshape model
    if args.lorentzian and args.free_shift:
        Fitargs['model'] = 'free_shift_lorentzian'
    elif args.lorentzian:
        Fitargs['model'] = 'lorentzian'
    elif args.free_shift:
        Fitargs['model'] = 'free_shift'
    else:
        Fitargs['model'] = 'voigt'

    verboseprint(mrs)
    verboseprint('Fitting args:')
    verboseprint(Fitargs)

    start = time.time()
    res = fitting.fit_FSLModel(mrs, **Fitargs)

    # Quantification
    # Echo time
    if args.TE is not None:
        echo_time = args.TE * 1E-3
    elif 'EchoTime' in FID.hdr_ext:
        echo_time = FID.hdr_ext['EchoTime']
    else:
        echo_time = None
    # Repetition time
    if args.TR is not None:
        repetition_time = args.TR
    elif 'RepetitionTime' in FID.hdr_ext:
        repetition_time = FID.hdr_ext['RepetitionTime']
    else:
        repetition_time = None

    # Internal and Water quantification if requested
    if (mrs.H2O is not None)\
            and (echo_time is not None)\
            and (repetition_time is not None):
        # All conditions for water referencing achieved
        # Form quantification information
        try:
            q_info = quantify.QuantificationInfo(
                echo_time,
                repetition_time,
                mrs.names,
                mrs.centralFrequency / 1E6,
                water_ref_metab=args.wref_metabolite,
                water_ref_metab_protons=args.ref_protons,
                water_ref_metab_limits=args.ref_int_limits,
                t1=args.t1_values,
                t2=args.t2_values)
        except (ValueError, TypeError) as exc:
            raise QuantificationException("Inappropriate arguments. Check inputs to fsl_mrs.") from exc
        except quantify.FieldStrengthInfoError as exc:
            raise QuantificationException(
                "This field strength is unknown to FSL-MRS and no relaxation constants are stored. "
                "Please specify these manually using the --t1-values and / or --t2-values arguments.") from exc
        except quantify.NoWaterScalingMetabolite as exc:
            raise QuantificationException(
                "No metabolites in FSL's dictionary of water scaling metabolites are in the basis set. "
                "Please specify a water scaling metabolite manually (--wref_metabolite). "
                "Also specify --ref_protons and --ref_int_limits.") from exc

        if args.tissue_frac:
            q_info.set_fractions(args.tissue_frac)
        if args.h2o_scale:
            q_info.add_corr = args.h2o_scale

        res.calculateConcScaling(mrs,
                                 quant_info=q_info,
                                 internal_reference=args.internal_ref,
                                 verbose=args.verbose)

        # Save quantification information as output
        with open(args.output / 'quantification_info.csv', 'w') as qfp:
            q_info.summary_table.to_csv(qfp)
    else:
        # Basic conditions (TR and TE set) are not met.
        # If water reference was provided but other conditions not met
        # then print some helpful warnings, but proceed with output using
        # internal referencing.
        if (mrs.H2O is not None) and (echo_time is None):
            warnings.warn(
                'H2O file provided but could not determine TE: '
                'no absolute quantification will be performed.',
                UserWarning)
        if (mrs.H2O is not None) and (repetition_time is None):
            warnings.warn(
                'H2O file provided but could not determine TR: '
                'no absolute quantification will be performed.',
                UserWarning)
        res.calculateConcScaling(mrs, internal_reference=args.internal_ref, verbose=args.verbose)

    # Combine metabolites.
    if args.combine is not None:
        res.combine(args.combine)
    stop = time.time()

    # Report on the fitting
    duration = stop - start
    verboseprint(f'    Fitting lasted          : {duration:.3f} secs.\n')

    # Save output files
    verboseprint(f'--->> Saving output files to {args.output}\n')

    res.to_file(
        filename=args.output / 'summary.csv',
        what='summary')
    res.to_file(
        filename=args.output / 'concentrations.csv',
        what='concentrations')
    res.to_file(
        filename=args.output / 'qc.csv',
        what='qc')
    res.to_file(
        filename=args.output / 'all_parameters.csv',
        what='parameters')
    if args.algo == 'MH':
        res.to_file(
            filename=args.output / 'concentration_samples.csv',
            what='concentrations-mh')
        res.to_file(
            filename=args.output / 'all_samples.csv',
            what='parameters-mh')

    # Save spectra
    data_out = res.predictedFID(mrs, mode='Full')
    data_out /= mrs.scaling['FID']
    data_out = data_out.reshape((1, 1, 1) + data_out.shape)
    out = gen_nifti_mrs(
        data_out,
        mrs.dwellTime,
        mrs.centralFrequency,
        nucleus=mrs.nucleus,
        affine=FID.voxToWorldMat)
    out.save(args.output / args.filename)

    if args.export_no_baseline:
        data_out = res.predictedFID(mrs, mode='Full', noBaseline=True)
        data_out /= mrs.scaling['FID']
        data_out = data_out.reshape((1, 1, 1) + data_out.shape)
        out = gen_nifti_mrs(
            data_out,
            mrs.dwellTime,
            mrs.centralFrequency,
            nucleus=mrs.nucleus,
            affine=FID.voxToWorldMat)
        out.save(args.output / (args.filename + '_no_baseline'))

    if args.export_baseline:
        data_out = res.predictedFID(mrs, mode='baseline')
        data_out /= mrs.scaling['FID']
        data_out = data_out.reshape((1, 1, 1) + data_out.shape)
        out = gen_nifti_mrs(
            data_out,
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
            out = gen_nifti_mrs(
                data_out,
                mrs.dwellTime,
                mrs.centralFrequency,
                nucleus=mrs.nucleus,
                affine=FID.voxToWorldMat)
            out.save(args.output / (args.filename + f'_{metab}'))

    # Save image of MRS voxel
    location_fig = None
    if args.t1 is not None \
            and FID.image.getXFormCode() > 0:
        fig = plotting.plot_world_orient(args.t1, args.data)
        fig.tight_layout()
        location_fig = args.output / 'voxel_location.png'
        fig.savefig(location_fig, bbox_inches='tight', facecolor='k')

    # Save quick summary figure
    report.fitting_summary_fig(
        mrs,
        res,
        filename=args.output / 'fit_summary.png')

    # Create interactive HTML report
    if args.report:
        report.create_svs_report(
            mrs,
            res,
            filename=args.output / 'report.html',
            fidfile=args.data,
            basisfile=args.basis,
            h2ofile=args.h2o,
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            location_fig=location_fig)

    verboseprint('\n\n\nDone.')


def str_or_int_arg(x):
    try:
        return int(x)
    except ValueError:
        return x


def tissue_frac_arg(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        x = Path(x)
        if x.suffix == '.json':
            import json
            with open(x) as jsonFile:
                return json.load(jsonFile)
        elif x.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(
                x,
                index_col=0)
            return {
                'WM': df.loc['WM', 'Tissue volume fractions'],
                'GM': df.loc['GM', 'Tissue volume fractions'],
                'CSF': df.loc['CSF', 'Tissue volume fractions']}


class TissueFracAction(configargparse.Action):
    """Sort out tissue fraction types. Should return dict"""
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values[0], dict):
            setattr(namespace, self.dest, values[0])
        else:
            setattr(namespace, self.dest,
                    {'WM': values[0], 'GM': values[1], 'CSF': values[2]})


def relaxation_arg(x: Path, mode: str) -> dict:
    try:
        with open(x) as relaxation_file:
            if x.suffix == '.json':
                import json
                relax_dict = json.load(relaxation_file)
            elif x.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(
                    relaxation_file,
                    index_col=0)
                if mode == 't1':
                    relax_dict = {
                        'H2O_GM': df.loc['GM', 'Water T1 (s)'],
                        'H2O_WM': df.loc['WM', 'Water T1 (s)'],
                        'H2O_CSF': df.loc['CSF', 'Water T1 (s)'],
                        'METAB': df.loc['GM', 'Metabolite T1 (s)']}
                elif mode == 't2':
                    relax_dict = {
                        'H2O_GM': df.loc['GM', 'Water T2 (ms)'] / 1E3,
                        'H2O_WM': df.loc['WM', 'Water T2 (ms)'] / 1E3,
                        'H2O_CSF': df.loc['CSF', 'Water T2 (ms)'] / 1E3,
                        'METAB': df.loc['GM', 'Metabolite T2 (ms)'] / 1E3}
    except IOError:
        raise configargparse.ArgumentError(
            "The t1/t2-values argument must be a path to a JSON formatted file "
            "or a previously generated quantification_info.csv.")
    if not relax_dict.keys() >= set(['H2O_GM', 'H2O_WM', 'H2O_CSF', 'METAB']):
        raise FSLMRSException("Relaxation JSON must contain 'H2O_GM', 'H2O_WM', 'H2O_CSF', 'METAB' fields.")
    return relax_dict


def t1_arg(x: str) -> dict:
    return relaxation_arg(Path(x), 't1')


def t2_arg(x: str) -> dict:
    return relaxation_arg(Path(x), 't2')


if __name__ == '__main__':
    main()
