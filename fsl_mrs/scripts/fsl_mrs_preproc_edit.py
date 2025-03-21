#!/usr/bin/env python

# fsl_mrs_preproc_edit - wrapper script for (MEGA) edited single voxel MRS preprocessing
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2021 University of Oxford
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
                    " - Complete edited SVS Preprocessing")

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
    optional.add_argument('--noise', type=str,
                          default=None, metavar='<str>',
                          help='Noise data data for estimating coil covariance'
                               ' (used during coil combination, optional).')
    # option to not average, this will toggle the average property to False
    optional.add_argument('--noaverage', action="store_false", dest='average',
                          help='Do not average repetitions.')
    optional.add_argument('--noalign', action="store_false", dest='align',
                          help='Do not align dynamics.')
    optional.add_argument('--align_ppm_dynamic', type=float, nargs=2, default=(1.9, 4.2),
                          help='PPM limit for dynamics dimension alignment. Default=(0.0, 4.2).')
    optional.add_argument('--align_window_dynamic', type=int,
                          help='Window for iterative windowed alignment, defaults to off.')
    optional.add_argument('--align_ppm_edit', type=float, nargs=2, default=None,
                          help='PPM limit for edit dimension alignment. Default=full spectrum.')
    optional.add_argument('--dynamic_align', action="store_true",
                          help='Align spectra based on dynamic fitting. Must specify dynamic basis sets.')
    optional.add_argument('--dynamic_align_edit', action="store_true",
                          help='Align conditions based on dynamic fitting. Must specify dynamic basis sets.')
    optional.add_argument('--dynamic_basis', type=str, nargs=2,
                          help='Paths to the two editing condition basis sets. '
                          'Only needed if --dynamic_align is specified.')
    optional.add_argument('--remove-water', action="store_true",
                          help='Apply HLSVD for residual water removal (replaces --hlsvd option).')
    optional.add_argument('--hlsvd', action="store_true",
                          help=configargparse.SUPPRESS)
    optional.add_argument('--truncate-fid', type=int, metavar='POINTS',
                          help='Remove points at the start of the fid (replaces --leftshift option).')
    optional.add_argument('--leftshift', type=int, metavar='POINTS',
                          help=configargparse.SUPPRESS)
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

    # ######################################################
    # DO THE IMPORTS AFTER PARSING TO SPEED UP HELP DISPLAY
    import shutil
    import numpy as np
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.utils.preproc import dyn_based_proc as dproc
    import fsl_mrs.core.nifti_mrs as ntools
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
        verbose_print('Conjugation explicitly set, applying conjugation.')

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

        if 'DIM_EDIT' in avg_ref_data.dim_tags:
            avg_ref_data = nifti_mrs_proc.average(avg_ref_data, 'DIM_EDIT')
        else:
            avg_ref_data = avg_ref_data

        no_prewhiten = False
        if args.noise is not None:
            noise = mrs_io.read_FID(args.noise)
            noise = np.swapaxes(
                noise[:],
                noise.dim_position('DIM_COIL'),
                -1)
            noise = noise.reshape(-1, noise.shape[-1]).T
            covariance = None
        else:
            # No noise input, but estimate a single coil covariance from the suppressed data
            # (more likely to have more data).
            noise = None
            from fsl_mrs.utils.preproc.combine import estimate_noise_cov, CovarianceEstimationError
            stacked_data = []
            for dd, _ in supp_data.iterate_over_dims(
                    dim='DIM_COIL',
                    iterate_over_space=True,
                    reduce_dim_index=True):
                stacked_data.append(dd)
            try:
                covariance = estimate_noise_cov(np.asarray(stacked_data))
            except CovarianceEstimationError as exc:
                # If the attempt to form a covariance fails, disable pre-whitening
                verbose_print(str(exc))
                verbose_print("Disabling pre-whitening in coil combination.")
                no_prewhiten = True

        supp_data = nifti_mrs_proc.coilcombine(
            supp_data,
            reference=avg_ref_data,
            report=report_dir,
            noise=noise,
            covariance=covariance,
            no_prewhiten=no_prewhiten)
        ref_data = nifti_mrs_proc.coilcombine(
            ref_data,
            reference=avg_ref_data,
            noise=noise,
            covariance=covariance,
            no_prewhiten=no_prewhiten)

        if args.quant is not None:
            quant_data = nifti_mrs_proc.coilcombine(
                quant_data,
                reference=avg_ref_data,
                noise=noise,
                covariance=covariance,
                no_prewhiten=no_prewhiten)
        if args.ecc is not None:
            ecc_data = nifti_mrs_proc.coilcombine(
                ecc_data,
                reference=avg_ref_data,
                noise=noise,
                covariance=covariance,
                no_prewhiten=no_prewhiten)

    # Frequency and phase align the FIDs in the ON/OFF condition
    if args.align:
        verbose_print('... Align Dynamics ...')
        supp_data = nifti_mrs_proc.align(
            supp_data,
            'DIM_DYN',
            ppmlim=args.align_ppm_dynamic,
            niter=4,
            window=args.align_window_dynamic,
            report=report_dir,
            report_all=True)

    if args.dynamic_align and args.align:
        verbose_print('... Align Dynamics using spectral model ...')
        # Run dynamic fitting based alignment
        edit_0, edit_1 = ntools.split(supp_data, 'DIM_EDIT', 0)

        basis_0 = mrs_io.read_basis(args.dynamic_basis[0])
        basis_1 = mrs_io.read_basis(args.dynamic_basis[1])

        fitargs_0 = {'baseline_order': 1,
                     'model': 'lorentzian',
                     'ppmlim': (0.2, 4.2)}
        fitargs_1 = {'baseline_order': 1,
                     'model': 'lorentzian',
                     'ppmlim': (2.0, 4.2)}

        edit_0_aligned, eps0, phi0 = dproc.align_by_dynamic_fit(edit_0, basis_0, fitargs_0)
        edit_1_aligned, eps1, phi1 = dproc.align_by_dynamic_fit(edit_1, basis_1, fitargs_1)

        if report_dir is not None:
            dproc.align_by_dynamic_fit_report(edit_0, edit_0_aligned, eps0, phi0, html=report_dir)
            dproc.align_by_dynamic_fit_report(edit_1, edit_1_aligned, eps1, phi1, html=report_dir)

        supp_data = ntools.merge([edit_0_aligned, edit_1_aligned], 'DIM_EDIT')

    if args.align:
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
    if args.remove_water or args.hlsvd:
        verbose_print('... Residual water removal ...')
        hlsvdlimits = [-0.25, 0.25]
        supp_data = nifti_mrs_proc.remove_peaks(supp_data, hlsvdlimits, limit_units='ppm', report=report_dir)

    if args.leftshift or args.truncate_fid:
        verbose_print('... Truncation ...')
        trunc_amount = args.truncate_fid if args.truncate_fid else args.leftshift
        supp_data = nifti_mrs_proc.truncate_or_pad(supp_data, -trunc_amount, 'first', report=report_dir)
        ref_data = nifti_mrs_proc.truncate_or_pad(ref_data, -trunc_amount, 'first')
        if args.quant is not None:
            quant_data = nifti_mrs_proc.truncate_or_pad(quant_data, -trunc_amount, 'first')

    # Apply shift to reference
    verbose_print('... Shift Cr to 3.027 ...')
    supp_data = nifti_mrs_proc.shift_to_reference(supp_data, 3.027, (2.9, 3.1), report=report_dir)

    # Apply phasing based on a single peak (Cr)
    verbose_print('... Phasing on tCr ...')
    supp_data = nifti_mrs_proc.phase_correct(supp_data, (2.9, 3.1), report=report_dir)

    if args.quant is not None:
        if 'DIM_EDIT' in quant_data.dim_tags:
            quant_data = nifti_mrs_proc.align(quant_data, 'DIM_EDIT', ppmlim=(0, 8), niter=1)
            quant_data = nifti_mrs_proc.average(quant_data, 'DIM_EDIT')
        final_wref = nifti_mrs_proc.phase_correct(quant_data, (4.55, 4.7), hlsvd=False, report=report_dir)
    else:
        if 'DIM_EDIT' in ref_data.dim_tags:
            ref_data = nifti_mrs_proc.align(ref_data, 'DIM_EDIT', ppmlim=(0, 8), niter=1)
            ref_data = nifti_mrs_proc.average(ref_data, 'DIM_EDIT')
        final_wref = nifti_mrs_proc.phase_correct(ref_data, (4.55, 4.7), hlsvd=False, report=report_dir)

    # Align between edit dimension
    if args.dynamic_align_edit and args.average:
        basis_0 = mrs_io.read_basis(args.dynamic_basis[0])
        basis_1 = mrs_io.read_basis(args.dynamic_basis[1])
        fitargs_edit = {
            'baseline_order': 2,
            'ppmlim': (0, 4.2)}
        metab_edit_align, eps, phi, _ = dproc.align_by_dynamic_fit(supp_data, [basis_0, basis_1], fitargs_edit)
        if report_dir is not None:
            dproc.align_by_dynamic_fit_report(supp_data, metab_edit_align, eps, phi, html=report_dir)

    elif args.dynamic_align_edit:
        # Do not average and dynamic alignment
        # Work out alignment on averaged data
        supp_data_avg = nifti_mrs_proc.average(supp_data, 'DIM_DYN')
        metab_edit_align, eps, phi = dproc.align_by_dynamic_fit(supp_data_avg, [basis_0, basis_1], fitargs_1)

        # Split, apply correction, recombine...
        unavg_0 = ntools.split(supp_data, 'DIM_EDIT', 0)
        unavg_1 = ntools.split(supp_data, 'DIM_EDIT', 0)
        unavg_0_aligned = nifti_mrs_proc.fshift(unavg_0, eps[0] * 1 / (2 * np.pi))
        unavg_0_aligned = nifti_mrs_proc.apply_fixed_phase(unavg_0_aligned, phi[0] * 180 / np.pi)

        unavg_1_aligned = nifti_mrs_proc.fshift(unavg_1, eps[1] * 1 / (2 * np.pi))
        unavg_1_aligned = nifti_mrs_proc.apply_fixed_phase(unavg_1_aligned, phi[1] * 180 / np.pi)

        metab_edit_align = ntools.merge([unavg_0_aligned, unavg_1_aligned], 'DIM_EDIT')

    else:
        metab_edit_align = nifti_mrs_proc.align(
            supp_data,
            'DIM_EDIT',
            ppmlim=args.align_ppm_edit,
            niter=4,
            report=report_dir)

    # Differencing
    metab_diff = nifti_mrs_proc.subtract(metab_edit_align, dim='DIM_EDIT', figure=False)
    metab_diff = nifti_mrs_proc.apply_fixed_phase(metab_diff, 180.0, report=report_dir)

    cond_0, cond_1 = ntools.split(metab_edit_align, 'DIM_EDIT', 0)

    # Save the data
    verbose_print('... Saving data ...')
    metab_diff.save(op.join(args.output, 'diff'))
    cond_0.save(op.join(args.output, 'edit_0'))
    cond_1.save(op.join(args.output, 'edit_1'))
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
