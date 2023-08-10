#!/usr/bin/env python

# fsl_mrs_proc - script for individual MRS preprocessing stages
#
# Author:   Will Clarke <william.clarke@ndcn.ox.ac.uk>
#           Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

# Imports
from __future__ import annotations
from typing import TYPE_CHECKING

from os import makedirs
from shutil import rmtree
import os.path as op
from dataclasses import dataclass
from pathlib import Path

from fsl.data.image import Image

from fsl_mrs.auxiliary import configargparse
from fsl_mrs import __version__
from fsl_mrs.utils.splash import splash
if TYPE_CHECKING:
    # Performed to enable NIFTI_MRS typing in dataclass
    # and a quick startup by not importing NIFTI_MRS yet.
    from fsl_mrs.core import NIFTI_MRS


class InappropriateDataError(Exception):
    pass


class ArgumentError(Exception):
    pass


@dataclass
class datacontainer:
    '''Class for keeping track of data and reference data together.'''
    data: NIFTI_MRS
    datafilename: str
    reference: NIFTI_MRS = None
    reffilename: str = None


def main():
    # Parse command-line arguments
    p = configargparse.ArgParser(
        add_config_file_help=False,
        description="FSL Magnetic Resonance Spectroscopy - Preprocessing")

    p.add_argument('-v', '--version', action='version', version=__version__)
    p.add('--config',
          required=False,
          is_config_file=True,
          help='configuration file')

    sp = p.add_subparsers(title='subcommands',
                          description='Preprocessing subcommands',
                          required=True,
                          dest='subcommand')

    # Coil combination subcommand
    ccparser = sp.add_parser('coilcombine',
                             help='Combine coils.',
                             add_help=False)
    cc_group = ccparser.add_argument_group('coilcombine arguments')
    cc_group.add_argument('--file', type=str, required=True,
                          help='Uncombined coil data file(s)')
    cc_group.add_argument('--reference', type=str, required=False,
                          help='Water unsuppressed reference data')
    cc_group.add_argument('--no_prewhiten', action="store_true",
                          help="Don't pre-whiten data before coil combination")
    cc_cov_group = cc_group.add_mutually_exclusive_group(required=False)
    cc_cov_group.add_argument(
        '--covariance',
        type=Path,
        default=None,
        help='Path to coil covariance matrix (for pre-whitening). '
             'Stored as space/line delimited text file. ')
    cc_cov_group.add_argument(
        '--noise',
        type=Path,
        default=None,
        help='Path to noise samples to calculate coil covariance matrix (for pre-whitening). '
             'Stored as ??. ')
    ccparser.set_defaults(func=coilcombine)
    add_common_args(ccparser)

    # Average subcommand
    avgparser = sp.add_parser('average', help='Average FIDs.', add_help=False)
    avg_group = avgparser.add_argument_group('average arguments')
    avg_group.add_argument('--file', type=str, required=True,
                           help='MRS file(s)')
    avg_group.add_argument('--dim', type=str, default='DIM_DYN',
                           help='Select dimension to average across. '
                                'Should be a NIfTI-MRS dimension tag, e.g. DIM_DYN (default)')
    avgparser.set_defaults(func=average)
    add_common_args(avgparser)

    # Align subcommand - frequency/phase alignment
    alignparser = sp.add_parser('align', help='Align FIDs.', add_help=False)
    align_group = alignparser.add_argument_group('Align arguments')
    align_group.add_argument('--file', type=str, required=True,
                             help='List of files to align')
    align_group.add_argument('--dim', type=str, default='DIM_DYN',
                             help='NIFTI-MRS dimension tag to align across.'
                                  'Or "all" to align over all spectra in higer dimensions.'
                                  'Default = DIM_DYN')
    align_group.add_argument('--ppm', type=float, nargs=2,
                             metavar=('<lower-limit>', '<upper-limit>'),
                             default=(0.2, 4.2),
                             help='ppm limits of alignment window'
                                  ' (default=0.2->4.2)')
    align_group.add_argument('--reference', type=str, required=False,
                             help='Align to this reference data.')
    align_group.add_argument('--apod', type=float, default=10,
                             help='Apodise data to reduce noise (Hz).')
    alignparser.set_defaults(func=align)
    add_common_args(alignparser)

    # Align difference spectra subcommand - frequency/phase alignment
    alignDparser = sp.add_parser('align-diff', add_help=False,
                                 help='Align subspectra for differencing.')
    alignD_group = alignDparser.add_argument_group('Align subspec arguments')
    alignD_group.add_argument('--file', type=str, required=True,
                              help='Subspectra 1 - List of files to align')
    alignD_group.add_argument('--dim', type=str, default='DIM_DYN',
                              help='NIFTI-MRS dimension tag to align across')
    alignD_group.add_argument('--dim_diff', type=str, default='DIM_EDIT',
                              help='NIFTI-MRS dimension tag to difference across')
    alignD_group.add_argument('--ppm', type=float, nargs=2,
                              metavar='<lower-limit upper-limit>',
                              default=(0.2, 4.2),
                              help='ppm limits of alignment window'
                                   ' (default=0.2->4.2)')
    alignD_group.add_argument('--diff_type', type=str, required=False,
                              default='add',
                              help='add (default) or subtract.')
    alignDparser.set_defaults(func=aligndiff)
    add_common_args(alignDparser)

    # ECC subcommand - eddy current correction
    eccparser = sp.add_parser('ecc', add_help=False,
                              help='Eddy current correction')
    ecc_group = eccparser.add_argument_group('ECC arguments')
    ecc_group.add_argument('--file', type=str, required=True,
                           help='Uncombined coil data file(s)')
    ecc_group.add_argument('--reference', type=str, required=True,
                           help='Phase reference data file(s)')
    eccparser.set_defaults(func=ecc)
    add_common_args(eccparser)

    # remove subcommand - remove peak using HLSVD
    hlsvdparser = sp.add_parser('remove', add_help=False,
                                help='Remove peak (default water) with HLSVD.')
    hlsvd_group = hlsvdparser.add_argument_group('HLSVD arguments')
    hlsvd_group.add_argument('--file', type=str, required=True,
                             help='Data file(s)')
    hlsvd_group.add_argument('--ppm', type=float, nargs=2,
                             metavar='<lower-limit upper-limit>',
                             default=[4.5, 4.8],
                             help='ppm limits of removal window.'
                                  ' Defaults to 4.5 to 4.8 ppm.'
                                  ' Includes (4.65 ppm) shift to TMS reference.')
    hlsvdparser.set_defaults(func=remove)
    add_common_args(hlsvdparser)

    # model subcommand - model peaks using HLSVD
    modelparser = sp.add_parser('model', add_help=False,
                                help='Model peaks with HLSVD.')
    model_group = modelparser.add_argument_group('HLSVD modelling arguments')
    model_group.add_argument('--file', type=str, required=True,
                             help='Data file(s)')
    model_group.add_argument('--ppm', type=float, nargs=2,
                             metavar='<lower-limit upper-limit>',
                             default=[4.5, 4.8],
                             help='ppm limits of removal window')
    model_group.add_argument('--components', type=int,
                             default=5,
                             help='Number of components to model.')
    modelparser.set_defaults(func=model)
    add_common_args(modelparser)

    # tshift subcommand - shift/resample in timedomain
    tshiftparser = sp.add_parser('tshift', add_help=False,
                                 help='shift/resample in timedomain.')
    tshift_group = tshiftparser.add_argument_group('Time shift arguments')
    tshift_group.add_argument('--file', type=str, required=True,
                              help='Data file(s) to shift')
    tshift_group.add_argument('--tshiftStart', type=float, default=0.0,
                              help='Time shift at start (ms),'
                                   ' negative pads with zeros,'
                                   ' positive truncates')
    tshift_group.add_argument('--tshiftEnd', type=float, default=0.0,
                              help='Time shift at end (ms),'
                                   ' negative truncates,'
                                   ' positive pads with zeros')
    tshift_group.add_argument('--samples', type=int,
                              help='Resample to N points in FID.')
    tshiftparser.set_defaults(func=tshift)
    add_common_args(tshiftparser)

    # truncate
    truncateparser = sp.add_parser('truncate', add_help=False,
                                   help='truncate or pad by integer'
                                        ' points in timedomain.')
    truncate_group = truncateparser.add_argument_group(
        'Truncate/pad arguments')
    truncate_group.add_argument('--file', type=str, required=True,
                                help='Data file(s) to shift')
    truncate_group.add_argument('--points', type=int, default=0,
                                help='Points to add/remove (+/-)')
    truncate_group.add_argument('--pos', type=str, default='last',
                                help="'first' or 'last' (default)")
    truncateparser.set_defaults(func=truncate)
    add_common_args(truncateparser)

    # apodize
    apodparser = sp.add_parser('apodize', help='Apodize FID.', add_help=False)
    apod_group = apodparser.add_argument_group('Apodize arguments')
    apod_group.add_argument('--file', type=str, required=True,
                            help='Data file(s) to shift')
    apod_group.add_argument('--filter', type=str, default='exp',
                            help="Filter choice."
                                 "Either 'exp' (default) or 'l2g'.")
    apod_group.add_argument('--amount', type=float, nargs='+',
                            help='Amount of broadening.'
                                 ' In Hz for exp mode.'
                                 ' Use space separated list for l2g.')
    apodparser.set_defaults(func=apodize)
    add_common_args(apodparser)

    # fshift subcommand - shift in frequency domain
    fshiftparser = sp.add_parser('fshift', add_help=False,
                                 help='shift in frequency domain.')
    fshift_group = fshiftparser.add_argument_group('Frequency shift arguments')
    fshift_group.add_argument('--file', type=str, required=True,
                              help='Data file(s) to shift')
    fshift_group.add_argument('--shiftppm', type=_float_or_array_arg,
                              metavar='Value | Image',
                              help='Apply fixed shift (ppm scale). '
                                   'Can be a nifti image of matched size for per-voxel shift')
    fshift_group.add_argument('--shifthz', type=_float_or_array_arg,
                              metavar='Value | Image',
                              help='Apply fixed shift (Hz scale). '
                                   'Can be a nifti image of matched size for per-voxel shift')
    fshift_group.add_argument('--shiftRef', action="store_true",
                              help='Shift to reference (default = Cr)')
    fshift_group.add_argument('--ppm', type=float, nargs=2,
                              metavar=('<lower-limit', 'upper-limit>'),
                              default=(2.8, 3.2),
                              help='Shift maximum point in this range'
                                   ' to target (must specify --target).')
    fshift_group.add_argument('--target', type=float, default=3.027,
                              help='Target position (must be used with ppm).'
                                   ' Default = 3.027')
    fshift_group.add_argument('--use_avg', action="store_true",
                              help='Use the average of higher dimensions to calculate shift.')
    fshiftparser.set_defaults(func=fshift)
    add_common_args(fshiftparser)

    # unlike subcomand - find FIDs that are unlike
    unlikeparser = sp.add_parser('unlike', add_help=False,
                                 help='Identify unlike FIDs.')
    unlike_group = unlikeparser.add_argument_group('unlike arguments')
    unlike_group.add_argument('--file', type=str, required=True,
                              help='Data file(s) to shift')
    unlike_group.add_argument('--sd', type=float, default=1.96,
                              help='Exclusion limit'
                                   ' (# of SD from mean,default=1.96)')
    unlike_group.add_argument('--iter', type=int, default=2,
                              help='Iterations of algorithm.')
    unlike_group.add_argument('--ppm', type=float, nargs=2,
                              metavar='<lower-limit upper-limit>',
                              default=None,
                              help='ppm limits of alignment window')
    unlike_group.add_argument('--outputbad', action="store_true",
                              help='Output failed FIDs')
    unlikeparser.set_defaults(func=unlike)
    add_common_args(unlikeparser)

    # Phasing - based on maximum point in range
    phaseparser = sp.add_parser('phase', add_help=False,
                                help='Phase spectrum based on'
                                     ' maximum point in range')
    phase_group = phaseparser.add_argument_group('Phase arguments')
    phase_group.add_argument('--file', type=str, required=True,
                             help='Data file(s) to shift')
    phase_group.add_argument('--ppm', type=float, nargs=2,
                             metavar='<lower-limit upper-limit>',
                             default=(2.8, 3.2),
                             help='ppm limits of alignment window')
    phase_group.add_argument('--hlsvd', action="store_true",
                             help='Remove peaks outside the search area')
    phase_group.add_argument('--use_avg', action="store_true",
                             help='Use the average of higher dimensions to calculate phase.')
    phaseparser.set_defaults(func=phase)
    add_common_args(phaseparser)

    fixphaseparser = sp.add_parser('fixed_phase', add_help=False,
                                   help='Apply fixed phase to spectrum')
    fphase_group = fixphaseparser.add_argument_group('Phase arguments')
    fphase_group.add_argument('--file', type=str, required=True,
                              help='Data file(s) to shift')
    fphase_group.add_argument('--p0', type=float,
                              metavar='<degrees>',
                              help='Zero order phase (degrees)')
    fphase_group.add_argument('--p1', type=float,
                              default=0.0,
                              metavar='<seconds>',
                              help='First order phase (seconds)')
    fphase_group.add_argument('--p1_type', type=str,
                              choices=['shift', 'linphase'],
                              default='shift',
                              help='Apply first order phase as timeshift or linear phase')
    fixphaseparser.set_defaults(func=fixed_phase)
    add_common_args(fixphaseparser)

    # subtraction - subtraction of FIDs
    subtractparser = sp.add_parser('subtract', add_help=False,
                                   help='Subtract two FID files or across a dimension')
    subtract_group = subtractparser.add_argument_group('Subtraction arguments')
    subtract_group.add_argument('--file', type=str, required=True,
                                help='File to subtract from')
    subtract_group.add_argument('--reference', type=str,
                                help='File to subtract from --file'
                                     '(output is file - reference)')
    subtract_group.add_argument('--dim', type=str,
                                help='NIFTI-MRS dimension tag to subtract across')
    subtractparser.set_defaults(func=subtract)
    add_common_args(subtractparser)

    # add - addition of FIDs
    addparser = sp.add_parser('add', add_help=False, help='Add two FIDs or across a dimension')
    add_group = addparser.add_argument_group('Addition arguments')
    add_group.add_argument('--file', type=str, required=True,
                           help='File to add to.')
    add_group.add_argument('--reference', type=str,
                           help='File to add to --file')
    add_group.add_argument('--dim', type=str,
                           help='NIFTI-MRS dimension tag to add across')
    addparser.set_defaults(func=add)
    add_common_args(addparser)

    # conj - conjugation
    conjparser = sp.add_parser('conj', add_help=False, help='Conjugate fids')
    conj_group = conjparser.add_argument_group('Conjugation arguments')
    conj_group.add_argument('--file', type=str, required=True,
                            help='Data file(s) to conjugate')
    conj_group.set_defaults(func=conj)
    add_common_args(conj_group)

    # mrsi-align - mrsi alignment across voxels
    malignparser = sp.add_parser(
        'mrsi-align',
        add_help=False,
        help='Phase and/or frequency align across voxels.')
    ma_group = malignparser.add_argument_group('MRSI alignment arguments')
    ma_group.add_argument('--file', type=str, required=True,
                          help='File to align.')
    ma_group.add_argument('--mask', type=str, required=False,
                          help='Mask file, NIfTI formated, only align on voxels selected.')
    ma_group.add_argument('--freq-align', action="store_true",
                          help='Run crosscorrelation frequency alignment.')
    ma_group.add_argument('--zpad', type=int, default=1,
                          help='Frequency alignment zero pading factor. 1 = double, 0 disables')
    ma_group.add_argument('--phase-correct', action="store_true",
                          help='Run phase correction.')
    ma_group.add_argument('--ppm', type=float, nargs=2,
                          metavar=('<lower-limit', 'upper-limit>'),
                          default=None,
                          help='ppm limits of phase correction window, default = no limits')
    ma_group.add_argument('--save-params', action="store_true",
                          help='Save shfits and/or phases to nifti format files.')
    ma_group.set_defaults(func=mrsi_align)
    add_common_args(malignparser)

    # mrsi-lipid - mrsi lipid removal
    mlipidparser = sp.add_parser(
        'mrsi-lipid',
        add_help=False,
        help='Remove lipids from MRSI by L2 regularisation.')
    ml_group = mlipidparser.add_argument_group('MRSI alignment arguments')
    ml_group.add_argument('--file', type=str, required=True,
                          help='File to align.')
    ml_group.add_argument('--mask', type=Image, required=False,
                          help='Mask file, NIfTI formated, only align on voxels selected.')
    ml_group.add_argument('--beta', type=float, default=1E-5,
                          help='Regularisation scaling, default = 1E-5. Adjust to scale lipid removal strength')
    ml_group.set_defaults(func=mrsi_lipid)
    add_common_args(mlipidparser)

    # Parse command-line arguments
    args = p.parse_args()

    # Output kickass splash screen
    if args.verbose:
        splash(logo='mrs')

    # Parse file arguments
    datafiles, reffiles = parsefilearguments(args)

    # Handle data loading
    dataList = loadData(datafiles,
                        refdatafile=reffiles)

    # Create output folder if required
    if not op.isdir(args.output):
        makedirs(args.output)
    elif op.isdir(args.output) and args.overwrite:
        rmtree(args.output)
        makedirs(args.output)

    # Handle report generation output location.
    # Bit of a hack, but I messed up the type expected by the
    # nifti mrs proc functions.
    if args.generateReports:
        args.generateReports = args.output
    else:
        args.generateReports = None

    # Call function - pass dict like view of args
    #  for compatibility with other modules
    dataout = args.func(dataList, vars(args))
    if isinstance(dataout, tuple):
        additionalOutputs = dataout[1:]
        dataout = dataout[0]
    else:
        additionalOutputs = None

    # Write data
    writeData(dataout, args)

    # Output any additional arguments
    if additionalOutputs is not None:
        print(additionalOutputs)


def add_common_args(p):
    """Add any arguments which are common between the sub commands."""
    # This is so the arguments can appear after the subcommand.

    # Arguments not associated with subcommands
    required = p.add_argument_group('required arguments')
    optional = p.add_argument_group('additional options')

    # REQUIRED ARGUMENTS
    required.add_argument('--output',
                          required=True, type=str, metavar='<str>',
                          help='output folder')

    # ADDITIONAL OPTIONAL ARGUMENTS
    optional.add_argument('--overwrite', action="store_true",
                          help='overwrite existing output folder')
    optional.add_argument('-r', '--generateReports', action="store_true",
                          help='Generate HTML reports.')
    # optional.add_argument('-i', '--reportIndicies',
    #                       type=int,
    #                       nargs='+',
    #                       default=[0],
    #                       help='Generate reports for selected inputs where'
    #                            ' multiple input files exist.'
    #                            ' Defaults to first (0).'
    #                            ' Specify as indices counting from 0.')
    optional.add_argument('--allreports', action="store_true",
                          help='Generate reports for all inputs.')
    # optional.add_argument('--conjugate', action="store_true",
    #                       help='apply conjugate to FID')
    optional.add_argument('--filename', type=str, metavar='<str>',
                          help='Override output file name.')
    optional.add_argument('--verbose', action="store_true",
                          help='spit out verbose info')
    optional.add_argument('-h', '--help', action='help',
                          help='show this help message and exit')


def parsefilearguments(args):
    # print(args.file)
    datafiles = args.file
    if 'reference' in args:
        # print(args.reference)
        reffiles = args.reference
    else:
        reffiles = None

    return datafiles, reffiles


# Data I/O functions
def loadData(datafile, refdatafile=None):
    """ Load data from path.

    The data must be of NIFTI MRS format.
    Optionaly loads a reference file.
    """
    from fsl_mrs.core import NIFTI_MRS, is_nifti_mrs

    # Do a check on the data file passed. The data must be of nifti type.
    if not is_nifti_mrs(datafile):
        raise ValueError('Preprocessing routines only handle NIFTI MRS'
                         ' format data. Please convert your data using'
                         ' spec2nii.')

    if refdatafile and not is_nifti_mrs(refdatafile):
        raise ValueError('Preprocessing routines only handle NIFTI MRS'
                         ' format data. Please convert your data using'
                         ' spec2nii.')

    if refdatafile:
        loaded_data = datacontainer(NIFTI_MRS(datafile),
                                    op.basename(datafile),
                                    NIFTI_MRS(refdatafile),
                                    op.basename(datafile))
    else:
        loaded_data = datacontainer(NIFTI_MRS(datafile),
                                    op.basename(datafile))

    return loaded_data


def writeData(dataobj, args):

    if args.filename is None:
        fileout = op.join(args.output, dataobj.datafilename)
    else:
        fileout = op.join(args.output, args.filename + '.nii.gz')

    dataobj.data.save(fileout)


# Option functions
# Functions below here should be associated with a
# subcommand method specified above.
# They should call a method in nifti_mrs_proc.py.

# Preprocessing functions
def coilcombine(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if 'DIM_COIL' not in dataobj.data.dim_tags:
        raise InappropriateDataError(f'Data ({dataobj.datafilename}) has no coil dimension.'
                                     f' Dimensions are {dataobj.data.dim_tags}.')

    if dataobj.reference is not None\
            and ('DIM_DYN' in dataobj.reference.dim_tags
                 or 'DIM_EDIT' in dataobj.reference.dim_tags):
        raise InappropriateDataError(f'Reference data ({dataobj.reference}) has addition dimensions. '
                                     f'Reduce dimensionality e.g. by averaging before using. '
                                     f'Dimensions are {dataobj.reference.dim_tags}.')

    # Covariance/noise inputs
    if args['covariance'] is not None:
        import numpy as np
        cov = np.loadtxt(args['covariance'])
        noise = None
    elif args['noise'] is not None:
        import numpy as np
        from fsl_mrs.core import NIFTI_MRS
        cov = None
        noise = NIFTI_MRS(args['noise'])
        noise = np.swapaxes(
            noise[:],
            noise.dim_position('DIM_COIL'),
            -1)
        noise = noise.reshape(-1, noise.shape[-1]).T
    else:
        cov = None
        noise = None

    combined = preproc.coilcombine(dataobj.data,
                                   reference=dataobj.reference,
                                   covariance=cov,
                                   noise=noise,
                                   no_prewhiten=args['no_prewhiten'],
                                   report=args['generateReports'],
                                   report_all=args['allreports'])

    return datacontainer(combined, dataobj.datafilename)


def average(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if args['dim'] not in dataobj.data.dim_tags:
        raise InappropriateDataError(f'Data ({dataobj.datafilename}) has no {args["dim"]} dimension.'
                                     f' Dimensions are is {dataobj.data.dim_tags}.')

    averaged = preproc.average(dataobj.data,
                               args["dim"],
                               report=args['generateReports'],
                               report_all=args['allreports'])

    return datacontainer(averaged, dataobj.datafilename)


def align(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if args['dim'].lower() == 'all':
        pass
    elif args['dim'] not in dataobj.data.dim_tags:
        raise InappropriateDataError(f'Data ({dataobj.datafilename}) has no {args["dim"]} dimension.'
                                     f' Dimensions are is {dataobj.data.dim_tags}.')

    aligned = preproc.align(dataobj.data,
                            args['dim'],
                            ppmlim=args['ppm'],
                            apodize=args['apod'],
                            report=args['generateReports'],
                            report_all=args['allreports'])

    return datacontainer(aligned, dataobj.datafilename)


def aligndiff(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if args['dim'] not in dataobj.data.dim_tags:
        raise InappropriateDataError(f'Data ({dataobj.datafilename}) has no {args["dim"]} dimension.'
                                     f' Dimensions are is {dataobj.data.dim_tags}.')

    aligned = preproc.aligndiff(dataobj.data,
                                args['dim'],
                                args['dim_diff'],
                                args['diff_type'],
                                ppmlim=args['ppm'],
                                report=args['generateReports'],
                                report_all=args['allreports'])

    return datacontainer(aligned, dataobj.datafilename)


def ecc(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    corrected = preproc.ecc(dataobj.data,
                            dataobj.reference,
                            report=args['generateReports'],
                            report_all=args['allreports'])

    return datacontainer(corrected, dataobj.datafilename)


def remove(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    corrected = preproc.remove_peaks(dataobj.data,
                                     limits=args['ppm'],
                                     report=args['generateReports'],
                                     report_all=args['allreports'])

    return datacontainer(corrected, dataobj.datafilename)


def model(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    modelled = preproc.hlsvd_model_peaks(dataobj.data,
                                         limits=args['ppm'],
                                         components=args['components'],
                                         report=args['generateReports'],
                                         report_all=args['allreports'])

    return datacontainer(modelled, dataobj.datafilename)


def tshift(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    shifted = preproc.tshift(dataobj.data,
                             tshiftStart=args['tshiftStart'],
                             tshiftEnd=args['tshiftEnd'],
                             samples=args['samples'],
                             report=args['generateReports'],
                             report_all=args['allreports'])

    return datacontainer(shifted, dataobj.datafilename)


def truncate(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    truncated = preproc.truncate_or_pad(dataobj.data,
                                        args['points'],
                                        args['pos'],
                                        report=args['generateReports'],
                                        report_all=args['allreports'])

    return datacontainer(truncated, dataobj.datafilename)


def apodize(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    apodized = preproc.apodize(dataobj.data,
                               args['amount'],
                               filter=args['filter'],
                               report=args['generateReports'],
                               report_all=args['allreports'])

    return datacontainer(apodized, dataobj.datafilename)


def fshift(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if args['shiftppm'] is not None:
        shift = args['shiftppm'] * dataobj.data.spectrometer_frequency[0]
        callMode = 'fixed'
    elif args['shifthz'] is not None:
        shift = args['shifthz']
        callMode = 'fixed'
    elif args['shiftRef']:
        callMode = 'ref'
    else:
        raise ArgumentError('Specify --shiftppm or --shifthz.')

    if callMode == 'fixed':
        shifted = preproc.fshift(dataobj.data,
                                 shift,
                                 report=args['generateReports'],
                                 report_all=args['allreports'])

    elif callMode == 'ref':
        shifted = preproc.shift_to_reference(dataobj.data,
                                             args['target'],
                                             args['ppm'],
                                             use_avg=args['use_avg'],
                                             report=args['generateReports'],
                                             report_all=args['allreports'])

    return datacontainer(shifted, dataobj.datafilename)


def unlike(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if dataobj.data.shape[:3] != (1, 1, 1):
        raise InappropriateDataError('unlike subcommand only works on single voxel data.'
                                     ' It is unclear what should happen with MRSI data.')

    good, bad = preproc.remove_unlike(
        dataobj.data,
        ppmlim=args['ppm'],
        sdlimit=args['sd'],
        niter=args['iter'],
        report=args['generateReports'])

    if args['outputbad'] and bad is not None:
        # Save bad results here - bit of a hack!
        if args['filename'] is None:
            badname = op.splitext(op.splitext(dataobj.datafilename)[0])[0] + '_FAIL'
        else:
            badname = args['filename'] + '_FAIL'

        bad.save(op.join(args['output'], badname))

    return datacontainer(good, dataobj.datafilename)


def phase(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    phased = preproc.phase_correct(dataobj.data,
                                   args['ppm'],
                                   hlsvd=args['hlsvd'],
                                   use_avg=args['use_avg'],
                                   report=args['generateReports'],
                                   report_all=args['allreports'])

    return datacontainer(phased, dataobj.datafilename)


def fixed_phase(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    phased = preproc.apply_fixed_phase(dataobj.data,
                                       args['p0'],
                                       p1=args['p1'],
                                       p1_type=args['p1_type'],
                                       report=args['generateReports'],
                                       report_all=args['allreports'])

    return datacontainer(phased, dataobj.datafilename)


def subtract(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if dataobj.reference is not None:
        subtracted = preproc.subtract(dataobj.data,
                                      data1=dataobj.reference,
                                      report=args['generateReports'],
                                      report_all=args['allreports'])
    elif args['dim'] is not None:
        subtracted = preproc.subtract(dataobj.data,
                                      dim=args['dim'],
                                      report=args['generateReports'],
                                      report_all=args['allreports'])
    else:
        raise ArgumentError('Specify --reference or --dim.')

    return datacontainer(subtracted, dataobj.datafilename)


def add(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    if dataobj.reference is not None:
        added = preproc.add(dataobj.data,
                            data1=dataobj.reference,
                            report=args['generateReports'],
                            report_all=args['allreports'])
    elif args['dim'] is not None:
        added = preproc.add(dataobj.data,
                            dim=args['dim'],
                            report=args['generateReports'],
                            report_all=args['allreports'])
    else:
        raise ArgumentError('Specify --reference or --dim.')

    return datacontainer(added, dataobj.datafilename)


def conj(dataobj, args):
    from fsl_mrs.utils.preproc import nifti_mrs_proc as preproc

    conjugated = preproc.conjugate(dataobj.data,
                                   report=args['generateReports'],
                                   report_all=args['allreports'])

    return datacontainer(conjugated, dataobj.datafilename)


def mrsi_align(dataobj, args):
    '''Function that applys frequency and/or phase correction to mrsi.'''
    from fsl_mrs.utils.preproc import mrsi

    if dataobj.data.shape[:3] == (1, 1, 1):
        raise ValueError('mrsi-align is not suitable for single voxel data.')

    if args['mask'] is not None:
        from fsl.data.image import Image
        mask = Image(args['mask'])
    else:
        mask = None

    if args['filename'] is None:
        fname = dataobj.datafilename
    else:
        fname = args['filename']

    data = dataobj.data
    if args['freq_align']:
        data, shifts = mrsi.mrsi_freq_align(
            data,
            mask=mask,
            zpad_factor=args['zpad'])
        if args['save_params']:
            shifts.save(op.join(args['output'], fname + '_shifts_hz.nii.gz'))

    if args['phase_correct']:
        data, phs = mrsi.mrsi_phase_corr(
            data,
            mask=mask,
            ppmlim=args['ppm'])
        if args['save_params']:
            phs.save(op.join(args['output'], fname + '_phase_deg.nii.gz'))

    return datacontainer(data, dataobj.datafilename)


def mrsi_lipid(dataobj, args):
    '''Apply lipid removel (L2 regularised) to MRSI'''
    from fsl_mrs.utils.preproc import mrsi

    if dataobj.data.shape[:3] == (1, 1, 1):
        raise ValueError('mrsi-align is not suitable for single voxel data.')

    return datacontainer(
        mrsi.lipid_removal_l2(
            dataobj.data,
            args['beta'],
            lipid_mask=args['mask']),
        dataobj.datafilename)


def _float_or_array_arg(x):
    '''Return either a float or array loaded from a nifti image'''
    try:
        return float(x)
    except ValueError:
        try:
            x = Path(x)
            assert x.exists()
            return Image(x)[:]
        except TypeError:
            raise ArgumentError('Argument must be a valid file path or float.')
        except AssertionError:
            raise ArgumentError(f'{x} does not exist as a path.')


if __name__ == '__main__':
    main()
