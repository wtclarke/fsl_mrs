#!/usr/bin/env python

# fsl_mrs - wrapper script for MRS fitting
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

# Quick imports
from fsl_mrs.auxiliary import configargparse

from fsl_mrs import __version__
from fsl_mrs.utils.splash import splash


# NOTE!!!! THERE ARE MORE IMPORTS IN THE CODE BELOW (AFTER ARGPARSING)

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
                          required=True, type=str, metavar='<str>',
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
    fitting_args.add_argument('--ppmlim', default=(.2, 4.2), type=float,
                              nargs=2, metavar=('LOW', 'HIGH'),
                              help='limit the fit to a freq range'
                                   ' (default=(.2,4.2))')
    fitting_args.add_argument('--h2o', default=None, type=str, metavar='H2O',
                              help='input .H2O file for quantification')
    fitting_args.add_argument('--baseline_order', default=2, type=int,
                              metavar=('ORDER'),
                              help='order of baseline polynomial'
                                   ' (default=2, -1 disables)')
    fitting_args.add_argument('--metab_groups', default=0, nargs='+',
                              type=str_or_int_arg,
                              help='metabolite groups: list of groups'
                                   ' or list of names for indept groups.')
    fitting_args.add_argument('--add_MM', action="store_true",
                              help="include default macromolecule peaks")
    fitting_args.add_argument('--add_MM_MEGA', action="store_true",
                              help="include default MEGA-PRESS macromolecule peaks. This option is experimental!")
    fitting_args.add_argument('--lorentzian', action="store_true",
                              help='Enable purely lorentzian broadening'
                                   ' (default is Voigt)')
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
                          default=None, metavar='WM GM CSF OR json',
                          help='Fractional tissue volumes for WM, GM, CSF'
                               ' or json segmentation file.'
                               ' Defaults to pure water scaling.')
    optional.add_argument('--internal_ref', type=str, default=['Cr', 'PCr'],
                          nargs='+',
                          help='Metabolite(s) used as an internal reference.'
                               ' Defaults to tCr (Cr+PCr).')
    optional.add_argument('--wref_metabolite', type=str, default=None,
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
    import os
    import shutil
    import json
    import warnings
    import matplotlib
    matplotlib.use('agg')
    from fsl_mrs.utils import mrs_io
    from fsl_mrs.utils import report
    from fsl_mrs.utils import fitting
    from fsl_mrs.utils import plotting
    from fsl_mrs.utils import misc
    from fsl_mrs.utils import quantify
    import datetime
    # ######################################################
    if not args.verbose:
        warnings.filterwarnings("ignore")

    # Check if output folder exists
    overwrite = args.overwrite
    if os.path.exists(args.output):
        if not overwrite:
            print(f"Folder '{args.output}' exists."
                  " Are you sure you want to delete it? [Y,N]")
            response = input()
            overwrite = response.upper() == "Y"
        if not overwrite:
            print('Early stopping...')
            exit()
        else:
            shutil.rmtree(args.output)
            os.makedirs(args.output, exist_ok=True)
    else:
        os.makedirs(args.output, exist_ok=True)

    # Save chosen arguments
    with open(os.path.join(args.output, "options.txt"), "w") as f:
        f.write(json.dumps(vars(args)))
        f.write("\n--------\n")
        f.write(p.format_values())

    # Do the work

    # Read data/h2o/basis
    if args.verbose:
        print('--->> Read input data and basis\n')
        print(f'  {args.data}')
        print(f'  {args.basis}\n')

    FID = mrs_io.read_FID(args.data)
    basis = mrs_io.read_basis(args.basis)

    if args.h2o is not None:
        H2O = mrs_io.read_FID(args.h2o)
    else:
        H2O = None

    # Instantiate MRS object
    mrs = FID.mrs(basis=basis,
                  ref_data=H2O)

    if isinstance(mrs, list):
        raise ValueError('fsl_mrs only handles a single FID at a time.'
                         ' Please preprocess data.')

    # Check the FID and basis / conjugate
    if args.conjfid is not None:
        if args.conjfid:
            mrs.conj_FID = True
    else:
        conjugated = mrs.check_FID(repair=True)
        if args.verbose:
            if conjugated == 1:
                warnings.warn('FID has been checked and conjugated.'
                              ' Please check!', UserWarning)

    if args.conjbasis is not None:
        if args.conjbasis:
            mrs.conj_Basis = True
    else:
        conjugated = mrs.check_Basis(repair=True)
        if args.verbose:
            if conjugated == 1:
                warnings.warn('Basis has been checked and conjugated.'
                              ' Please check!', UserWarning)

    # Rescale FID, H2O and basis to have nice range
    if not args.no_rescale:
        mrs.rescaleForFitting(ind_scaling=args.ind_scale)

    # Keep/Ignore metabolites
    mrs.keep = args.keep
    mrs.ignore = args.ignore

    # Do the fitting here
    if args.verbose:
        print('--->> Start fitting\n\n')
        print('    Algorithm = [{}]\n'.format(args.algo))
    start = time.time()

    ppmlim = args.ppmlim
    if ppmlim is not None:
        ppmlim = (ppmlim[0], ppmlim[1])

    # Parse metabolite groups
    metab_groups = misc.parse_metab_groups(mrs, args.metab_groups)

    # Include Macromolecules? These should have their own metab groups
    if args.add_MM:
        if args.verbose:
            print('Adding macromolecules')
        nMM = mrs.add_default_MM_peaks(gamma=10, sigma=20)
        G = [i + max(metab_groups) + 1 for i in range(nMM)]
        metab_groups += G
    if args.add_MM_MEGA:
        if args.verbose:
            print('Adding MEGA-PRESS macromolecules')
        nMM = mrs.add_default_MEGA_MM_peaks(gamma=10, sigma=0)
        G = [i + max(metab_groups) + 1 for i in range(nMM)]
        metab_groups += G

    # Choose fitting lineshape model.
    if args.lorentzian:
        Fitargs = {'ppmlim': ppmlim,
                   'method': args.algo,
                   'baseline_order': args.baseline_order,
                   'metab_groups': metab_groups,
                   'model': 'lorentzian',
                   'disable_mh_priors': args.disable_MH_priors,
                   'MHSamples': args.mh_samples}
    else:
        Fitargs = {'ppmlim': ppmlim,
                   'method': args.algo,
                   'baseline_order': args.baseline_order,
                   'metab_groups': metab_groups,
                   'model': 'voigt',
                   'disable_mh_priors': args.disable_MH_priors,
                   'MHSamples': args.mh_samples}

    if args.verbose:
        print(mrs)
        print('Fitting args:')
        print(Fitargs)

    res = fitting.fit_FSLModel(mrs, **Fitargs)

    # Quantification
    # Echo time
    if args.TE is not None:
        echotime = args.TE * 1E-3
    elif 'EchoTime' in FID.hdr_ext:
        echotime = FID.hdr_ext['EchoTime']
    else:
        echotime = None
    # Repetition time
    if args.TR is not None:
        repetition_time = args.TR
    elif 'RepetitionTime' in FID.hdr_ext:
        repetition_time = FID.hdr_ext['RepetitionTime']
    else:
        repetition_time = None

    # Internal and Water quantification if requested
    if (mrs.H2O is None) or (echotime is None) or (repetition_time is None):
        if echotime is None:
            warnings.warn('H2O file provided but could not determine TE:'
                          ' no absolute quantification will be performed.',
                          UserWarning)
        if repetition_time is None:
            warnings.warn('H2O file provided but could not determine TR:'
                          ' no absolute quantification will be performed.',
                          UserWarning)
        res.calculateConcScaling(mrs, internal_reference=args.internal_ref, verbose=args.verbose)
    else:
        # Form quantification information
        q_info = quantify.QuantificationInfo(echotime,
                                             repetition_time,
                                             mrs.names,
                                             mrs.centralFrequency / 1E6,
                                             water_ref_metab=args.wref_metabolite,
                                             water_ref_metab_protons=args.ref_protons,
                                             water_ref_metab_limits=args.ref_int_limits)

        if args.tissue_frac:
            q_info.set_fractions(args.tissue_frac)
        if args.h2o_scale:
            q_info.add_corr = args.h2o_scale

        res.calculateConcScaling(mrs,
                                 quant_info=q_info,
                                 internal_reference=args.internal_ref,
                                 verbose=args.verbose)

    # Combine metabolites.
    if args.combine is not None:
        res.combine(args.combine)
    stop = time.time()

    # Report on the fitting
    if args.verbose:
        duration = stop - start
        print(f'    Fitting lasted          : {duration:.3f} secs.\n')
    # Save output files
    if args.verbose:
        print('--->> Saving output files to {}\n'.format(args.output))

    res.to_file(filename=os.path.join(args.output, 'summary.csv'),
                what='summary')
    res.to_file(filename=os.path.join(args.output, 'concentrations.csv'),
                what='concentrations')
    res.to_file(filename=os.path.join(args.output, 'qc.csv'),
                what='qc')
    res.to_file(filename=os.path.join(args.output, 'all_parameters.csv'),
                what='parameters')
    if args.algo == 'MH':
        res.to_file(filename=os.path.join(
                    args.output, 'concentration_samples.csv'),
                    what='concentrations-mh')
        res.to_file(filename=os.path.join(args.output, 'all_samples.csv'),
                    what='parameters-mh')

    # Save image of MRS voxel
    location_fig = None
    if args.t1 is not None \
            and FID.getXFormCode() > 0:
        fig = plotting.plot_world_orient(args.t1, args.data)
        fig.tight_layout()
        location_fig = os.path.join(args.output, 'voxel_location.png')
        fig.savefig(location_fig, bbox_inches='tight', facecolor='k')

    # Save quick summary figure
    report.fitting_summary_fig(mrs, res,
                               filename=os.path.join(args.output,
                                                     'fit_summary.png'))

    # Create interactive HTML report
    if args.report:
        report.create_report(
            mrs,
            res,
            filename=os.path.join(args.output, 'report.html'),
            fidfile=args.data,
            basisfile=args.basis,
            h2ofile=args.h2o,
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            location_fig=location_fig)

    if args.verbose:
        print('\n\n\nDone.')


def str_or_int_arg(x):
    try:
        return int(x)
    except ValueError:
        return x


def tissue_frac_arg(x):
    import json
    try:
        with open(x) as jsonFile:
            jsonString = jsonFile.read()
        return json.loads(jsonString)
    except IOError:
        return float(x)


class TissueFracAction(configargparse.Action):
    """Sort out tissue fraction types. Should return dict"""
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values[0], dict):
            setattr(namespace, self.dest, values[0])
        else:
            setattr(namespace, self.dest,
                    {'WM': values[0], 'GM': values[1], 'CSF': values[2]})


if __name__ == '__main__':
    main()
