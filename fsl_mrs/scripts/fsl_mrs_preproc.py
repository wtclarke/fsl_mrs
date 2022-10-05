#!/usr/bin/env python

# fsl_mrs_preproc - wrapper script for single voxel MRS preprocessing
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
import os.path as op
from os import mkdir
# NOTE!!!! THERE ARE MORE IMPORTS IN THE CODE BELOW (AFTER ARGPARSING)


def main():
    # Parse command-line arguments
    p = configargparse.ArgParser(
        add_config_file_help=False,
        description="FSL Magnetic Resonance Spectroscopy"
                    " - Complete non-edited SVS Preprocessing")

    p.add_argument('-v', '--version', action='version', version=__version__)

    required = p.add_argument_group('required arguments')
    optional = p.add_argument_group('additional options')

    # REQUIRED ARGUMENTS
    required.add_argument('--data',
                          required=True, type=str, metavar='<str>',
                          help='input files')
    required.add_argument('--reference',
                          required=True, type=str, metavar='<str>',
                          help='Reference non-water suppressed file')
    required.add_argument('--output',
                          required=True, type=str, metavar='<str>',
                          help='output folder')

    # ADDITONAL OPTIONAL ARGUMENTS
    optional.add_argument('--quant', type=str,
                          default=None, metavar='<str>',
                          help='Water reference data for'
                               ' quantification (Optional).')
    optional.add_argument('--ecc', type=str,
                          default=None, metavar='<str>',
                          help='Water reference data for eddy'
                               ' current correction (Optional).')
    optional.add_argument('--fmrs', action="store_true",
                          help='Preprocessing for fMRS, automattically sets noremoval and noaverage arguments')
    optional.add_argument('--noremoval', action="store_false", dest='unlike',
                          help='Do not remove unlike averages.')
    optional.add_argument('--noaverage', action="store_false", dest='average',
                          help='Do not average repetitions.')
    optional.add_argument('--hlsvd', action="store_true",
                          help='Apply HLSVD for residual water removal.')
    optional.add_argument('--leftshift', type=int, metavar='POINTS',
                          help='Remove points at the start of the fid.')
    optional.add_argument('--t1', type=str, default=None, metavar='IMAGE',
                          help='structural image (for report)')
    optional.add_argument('--verbose', action="store_true",
                          help='spit out verbose info')
    optional.add_argument('--conjugate', action="store_true",
                          help='apply conjugate to FID')
    optional.add_argument('--overwrite', action="store_true",
                          help='overwrite existing output folder')
    optional.add_argument('--report', action="store_true",
                          help='Generate report in output folder')
    optional.add('--config', required=False, is_config_file=True,
                 help='configuration file')

    # Parse command-line arguments
    args = p.parse_args()

    # Shorthand verbose printing
    def verbose_print(x):
        if args.verbose:
            print(x)

    # Output kickass splash screen
    if args.verbose:
        splash(logo='mrs')

    if args.fmrs:
        verbose_print('Running in fMRS mode:')
        verbose_print('  --noremoval and --noaverage set.')
        args.average = False
        args.unlike = False

    # ######################################################
    # DO THE IMPORTS AFTER PARSING TO SPEED UP HELP DISPLAY
    import shutil
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.utils import plotting
    from fsl_mrs.utils import mrs_io
    # ######################################################

    # Check if output folder exists
    overwrite = args.overwrite
    if op.exists(args.output):
        if not overwrite:
            print(f"Folder '{args.output}' exists."
                  "Are you sure you want to delete it? [Y,N]")
            response = input()
            overwrite = response.upper() == "Y"
        if not overwrite:
            print('Early stopping...')
            exit()
        else:
            shutil.rmtree(args.output)
            mkdir(args.output)
    else:
        mkdir(args.output)

    # Save chosen arguments
    with open(op.join(args.output, "options.txt"), "w") as f:
        f.write(str(args))
        f.write("\n--------\n")
        f.write(p.format_values())

    if args.report:
        report_dir = args.output
    else:
        # Will suppress report generation
        report_dir = None

    # ######  Do the work #######
    verbose_print('Load the data....')

    # Read the data
    # Suppressed data
    supp_data = mrs_io.read_FID(args.data)
    verbose_print(f'.... Found data with shape {supp_data.shape}.\n\n')
    if supp_data.dim_tags == [None, None, None]:
        print(
            'This data contains no unaveraged transients or uncombined coils. '
            'Please carefully ascertain what pre-processing has already taken place, '
            'before running appropriate individual steps using fsl_mrs_proc. '
            'Note, no pre-processing may be necessary.')
        return 1

    # Reference data
    ref_data = mrs_io.read_FID(args.reference)
    verbose_print(f'.... Found reference with shape {ref_data.shape}.\n\n')

    if args.quant is not None:
        # Separate quant data
        quant_data = mrs_io.read_FID(args.quant)
        verbose_print(f'.... Found quant with shape {quant_data.shape}.\n\n')
    else:
        quant_data = None

    if args.ecc is not None:
        # Separate ecc data
        ecc_data = mrs_io.read_FID(args.ecc)
        verbose_print(f'.... Found ecc with shape {ecc_data.shape}.\n\n')
    else:
        ecc_data = None

    # Data conjugation
    if args.conjugate:
        verbose_print('Conjugation explicitly set,'
                      'applying conjugation.')

        supp_data = nifti_mrs_proc.conjugate(supp_data)
        ref_data = nifti_mrs_proc.conjugate(ref_data)
        if args.quant is not None:
            quant_data = nifti_mrs_proc.conjugate(quant_data)
        if args.ecc is not None:
            ecc_data = nifti_mrs_proc.conjugate(ecc_data)

    # Determine if coils have been combined already
    verbose_print('.... Determine if coil combination is needed')
    if 'DIM_COIL' in supp_data.dim_tags:
        do_coil_combine = True
        verbose_print('  ----> YES.\n')
    else:
        do_coil_combine = False
        verbose_print('   ----> NO.\n')

    # Do preproc
    verbose_print('Begin proprocessing.... ')

    if do_coil_combine:
        verbose_print('... Coil Combination ...')

        if 'DIM_DYN' in ref_data.dim_tags:
            avg_ref_data = nifti_mrs_proc.average(ref_data, 'DIM_DYN')
        else:
            avg_ref_data = ref_data

        supp_data = nifti_mrs_proc.coilcombine(supp_data, reference=avg_ref_data, report=report_dir)
        ref_data = nifti_mrs_proc.coilcombine(ref_data, reference=avg_ref_data)

        if args.quant is not None:
            quant_data = nifti_mrs_proc.coilcombine(quant_data, reference=avg_ref_data)
        if args.ecc is not None:
            ecc_data = nifti_mrs_proc.coilcombine(ecc_data, reference=avg_ref_data)

    verbose_print('... Align Dynamics (1st iteration) ...')
    supp_data = nifti_mrs_proc.align(supp_data, 'DIM_DYN', ppmlim=(0, 4.2), report=report_dir)

    # Bad average removal on the suppressed data
    if args.unlike:
        verbose_print('... Removing unlike averages (>2.58\u03C3 from mean) ...')

        supp_data, bd_data = nifti_mrs_proc.remove_unlike(supp_data,
                                                          sdlimit=2.58,
                                                          niter=1,
                                                          ppmlim=(0, 4.2),
                                                          report=report_dir)
        if bd_data is None:
            bd_shape = 0
        else:
            bd_shape = bd_data.shape[4]
        if supp_data is None:
            supp_shape = 0
        else:
            supp_shape = supp_data.shape[4]
        verbose_print(f'{bd_shape}/{supp_shape + bd_shape} '
                      'bad averages identified and removed.')

    # Frequency and phase align the FIDs
    verbose_print('... Align Dynamics (2nd iteration) ...')
    supp_data = nifti_mrs_proc.align(supp_data, 'DIM_DYN', ppmlim=(0, 4.2), report=report_dir)

    if 'DIM_DYN' in ref_data.dim_tags:
        ref_data = nifti_mrs_proc.align(ref_data, 'DIM_DYN', ppmlim=(0, 8))

    if args.quant is not None and ('DIM_DYN' in quant_data.dim_tags):
        quant_data = nifti_mrs_proc.align(quant_data, 'DIM_DYN', ppmlim=(0, 8))
    if args.ecc is not None and ('DIM_DYN' in ecc_data.dim_tags):
        ecc_data = nifti_mrs_proc.align(ecc_data, 'DIM_DYN', ppmlim=(0, 8))

    # Average the data (if asked to do so)
    if args.average:
        verbose_print('... Average FIDs ...')
        supp_data = nifti_mrs_proc.average(supp_data, 'DIM_DYN', report=report_dir)
        if 'DIM_DYN' in ref_data.dim_tags:
            ref_data = nifti_mrs_proc.average(ref_data, 'DIM_DYN')
        if args.quant is not None and ('DIM_DYN' in quant_data.dim_tags):
            quant_data = nifti_mrs_proc.average(quant_data, 'DIM_DYN')

    # Always average ecc if it exists as a separate scan
    if args.ecc is not None and ('DIM_DYN' in ecc_data.dim_tags):
        ecc_data = nifti_mrs_proc.average(ecc_data, 'DIM_DYN')

    # Eddy current correction
    verbose_print('... ECC correction ...')
    if args.ecc is not None:
        eccRef = ecc_data
    else:
        if 'DIM_DYN' in ref_data.dim_tags:
            eccRef = nifti_mrs_proc.average(ref_data, 'DIM_DYN')
        else:
            eccRef = ref_data

    supp_data = nifti_mrs_proc.ecc(supp_data, eccRef, report=report_dir)
    ref_data = nifti_mrs_proc.ecc(ref_data, eccRef)
    # Typically if a separate "quantification" water reference
    #  has been collected it will have most gradients removed
    # (OVS and water suppression), therefore use it as it's own reference.
    if args.quant is not None:
        quant_data = nifti_mrs_proc.ecc(quant_data, quant_data)

    # HLSVD
    if args.hlsvd:
        if not args.average:
            print('Warning: Running HLSVD water removal on each unaveraged dynamic might take a long time, '
                  'and potentially introduce high variance.')
        verbose_print('... Residual water removal ...')
        hlsvdlimits = [-0.25, 0.25]
        supp_data = nifti_mrs_proc.remove_peaks(supp_data, hlsvdlimits, limit_units='ppm', report=report_dir)

    if args.leftshift:
        verbose_print('... Truncation ...')
        supp_data = nifti_mrs_proc.truncate_or_pad(supp_data, -args.leftshift, 'first', report=report_dir)
        ref_data = nifti_mrs_proc.truncate_or_pad(ref_data, -args.leftshift, 'first')
        if args.quant is not None:
            quant_data = nifti_mrs_proc.truncate_or_pad(quant_data, -args.leftshift, 'first')

    # Apply shift to reference
    verbose_print('... Shifting tCr to 3.027 ...')
    supp_data = nifti_mrs_proc.shift_to_reference(
        supp_data,
        3.027,
        (2.9, 3.1),
        use_avg=not args.average,
        report=report_dir)

    # Apply phasing based on a single peak (tCr)
    verbose_print('... Phasing on tCr ...')
    supp_data = nifti_mrs_proc.phase_correct(
        supp_data,
        (2.9, 3.1),
        use_avg=not args.average,
        report=report_dir)
    if args.quant is not None:
        final_wref = nifti_mrs_proc.phase_correct(quant_data, (4.55, 4.7), hlsvd=False, report=report_dir)
    else:
        final_wref = nifti_mrs_proc.phase_correct(ref_data, (4.55, 4.7), hlsvd=False, report=report_dir)

    # Save the data
    verbose_print('... Saving data ...')
    supp_data.save(op.join(args.output, 'metab'))
    final_wref.save(op.join(args.output, 'wref'))

    # Produce full html report
    if args.report:
        import subprocess
        import glob
        verbose_print('Create report')
        htmlfiles = glob.glob(op.join(args.output, '*.html'))
        subprocess.call(['merge_mrs_reports', '-d',
                         op.join(args.output, 'metab'),
                         '-o', args.output,
                         '--delete'] + htmlfiles)

        if args.t1 is not None:
            fig = plotting.plot_world_orient(args.t1, args.data)
            fig.savefig(op.join(args.output, 'voxel_location.png'))


if __name__ == '__main__':
    main()
