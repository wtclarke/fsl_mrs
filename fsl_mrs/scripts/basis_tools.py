#!/usr/bin/env python

""" basis_tools - top level script for calling general mrs basis set handling tools

Author: William Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 2021 University of Oxford
SHBASECOPYRIGHT
"""

import argparse
from pathlib import Path

from fsl_mrs import __version__


def main():
    # Parse command-line arguments
    p = argparse.ArgumentParser(description="FSL Magnetic Resonance Spectroscopy - Basis Set Tools")

    p.add_argument('-v', '--version', action='version', version=__version__)

    sp = p.add_subparsers(title='subcommands',
                          description='Availible tools',
                          required=True,
                          dest='subcommand')

    # Info tool
    infoparser = sp.add_parser(
        'info',
        help='Information about the basis set.')
    infoparser.add_argument(
        'file',
        type=Path,
        help='Basis file or folder')
    infoparser.set_defaults(func=info)

    # Vis tool
    visparser = sp.add_parser(
        'vis',
        help='Quick visualisation of a basis set.')
    visparser.add_argument('file', type=Path, metavar='FILE or DIR',
                           help='NIfTI file or directory of basis sets')
    visparser.add_argument('--ppmlim', default=(.2, 4.2), type=float,
                           nargs=2, metavar=('LOW', 'HIGH'),
                           help='limit the fit to a freq range (default=(.2,4.2))')
    visparser.set_defaults(func=vis)

    # Convert tool - Convert lcm or jmrui format to fsl format
    convertparser = sp.add_parser(
        'convert',
        help='Convert LCModel or jMRUI formated basis to FSL format.')
    convertparser.add_argument('input', type=Path,
                               help='Input basis file or folder')
    convertparser.add_argument('output', type=Path,
                               help='Output fsl formatted folder, will be created if needed.')
    convertparser.add_argument('--bandwidth', type=float, default=None,
                               help='Required for LCModel RAW format only: spectral bandwidth in Hz.')
    convertparser.add_argument('--fieldstrength', type=float, default=None,
                               help='Required for LCModel RAW format only: field strength in tesla.')
    convertparser.add_argument('--remove_reference', action="store_true",
                               help='Remove LCModel reference peak.')
    convertparser.add_argument('--hlsvd', action="store_true",
                               help='Use HLSVD peak removal, rather than zeroing.')
    convertparser.set_defaults(func=convert)

    # Add tool - add a json formatted fid to a basis set
    addparser = sp.add_parser(
        'add',
        help='Add a json formatted basis spectrum to a basis set.')
    addparser.add_argument('new', type=Path,
                           help='Basis to add')
    addparser.add_argument('target', type=str,
                           help='Basis to add to.')
    addparser.add_argument('--info', type=str, default=None,
                           help='Optional info string added to basis file.')
    addparser.add_argument('--name', type=str, default=None,
                           help='Optionally provide a different name for the basis.')
    addparser.add_argument('--scale', action="store_true",
                           help='Rescale amplitude to mean of target basis.')
    addparser.add_argument('--conj', action="store_true",
                           help='Conjugate added basis.')
    addparser.add_argument('--pad', action="store_true",
                           help='Pad added FID if required.')
    addparser.set_defaults(func=add)

    # Shift tool
    shiftparser = sp.add_parser(
        'shift',
        help='Frequency shift a single basis spectrum.')
    shiftparser.add_argument('file', type=Path,
                             help='Basis file to modify')
    shiftparser.add_argument('metabolite', type=str,
                             help='Metabolite to shift,'
                             ' use "all" to shift all metabolites')
    shiftparser.add_argument('ppm_shift', type=float,
                             help='Shift to apply (in ppm).')
    shiftparser.add_argument('output', type=Path,
                             help='Output location, can overwrite.')
    shiftparser.set_defaults(func=shift)

    # Shift all peaks based on results
    shiftparser = sp.add_parser(
        'shift_all',
        help='ADVANCED: Frequency shift all basis spectra using fit results. '
             'This should only be carried out using high SNR population/cohort average data.')
    shiftparser.add_argument('file', type=Path,
                             help='Basis file to modify')
    shiftparser.add_argument('results_dir', type=Path,
                             help='Path to fit results dir. '
                                  'Fit model must be free_shift. '
                                  'Basis set metabolites must be identical.')
    shiftparser.add_argument('output', type=Path,
                             help='Output location, can overwrite.')
    shiftparser.set_defaults(func=all_shift)

    # Rescale tool
    scaleparser = sp.add_parser(
        'scale',
        help='Rescale a single basis spectrum to the mean of the other spectra.')
    scaleparser.add_argument('file', type=Path,
                             help='Basis file to modify')
    scaleparser.add_argument('metabolite', type=str,
                             help='Metabolite to shift')
    scaleparser.add_argument('--target_scale', type=float,
                             default=None,
                             help='Use to specifiy an explicit scaling.')
    scaleparser.add_argument('output', type=Path,
                             help='Output location, can overwrite.')
    scaleparser.set_defaults(func=rescale)

    # Differenceing (add/subtract)
    diffparser = sp.add_parser(
        'diff',
        help='Form difference basis spectra from two basis spectra.')
    diffparser.add_argument('file1', type=Path,
                            help='Basis file one')
    diffparser.add_argument('file2', type=Path,
                            help='Basis file two, if subtract file 2 is subtracted from file 1.')
    diffparser.add_argument('output', type=Path,
                            help='Output fsl formatted folder, will be created if needed.')
    diffparser.add_argument('--add_or_sub', type=str,
                            default='sub',
                            help="'add' or 'sub'")
    diffparser.add_argument('--ignore_missing', action="store_true",
                            help='Ignore missing basis sets in one basis.')
    diffparser.set_defaults(func=diff)

    # Conjugate
    conjparser = sp.add_parser(
        'conj',
        help='Conjugate (reverse frequency axis) all or single basis.')
    conjparser.add_argument('file', type=Path,
                            help='Basis file')
    conjparser.add_argument('output', type=Path,
                            help='Output location, can overwrite.')
    conjparser.add_argument('--metabolite', type=str, default=None,
                            help='Specify metabolite to conjugate')
    conjparser.set_defaults(func=conj)

    # Remove peak
    rempeakparser = sp.add_parser(
        'remove_peak',
        help='Edit a basis set by removing a peak (e.g. TMS reference).')
    rempeakparser.add_argument('file', type=Path,
                               help='Basis file')
    rempeakparser.add_argument('output', type=Path,
                               help='Output location, can overwrite.')
    mutual = rempeakparser.add_mutually_exclusive_group(required=True)
    mutual.add_argument('--metabolite', type=str, default=None,
                        help='Specify metabolite to edit')
    mutual.add_argument('--all', action="store_true",
                        help='Edit all basis spectra.')
    rempeakparser.add_argument('--hlsvd', action="store_true",
                               help='Use HLSVD peak removal, rather than zeroing.')
    rempeakparser.add_argument('--ppmlim', default=(-.2, .2), type=float,
                               nargs=2, metavar=('LOW', 'HIGH'),
                               help='Peak removal ppm range (default=(-.2, .2))')
    rempeakparser.set_defaults(func=rem_peak)

    # Add manual peak (sets) defined by ppm etc
    add_p_parser = sp.add_parser(
        'add_peak',
        help='Edit a basis set by adding a peak specified by position, amplitude, width.')
    add_p_parser.add_argument('file', type=Path,
                              help='Basis file')
    add_p_parser.add_argument('output', type=Path,
                              help='Output location, can overwrite.')
    add_p_parser.add_argument('ppm', type=float, nargs='+',
                              help='Peak positions in ppm.')
    add_p_parser.add_argument('amp', type=float, nargs='+',
                              help='Peak amplitudes in ppm.')
    add_p_parser.add_argument('--gamma', type=float, default=0,
                              help='Peak widths (lorentzian) in Hz.')
    add_p_parser.add_argument('--sigma', type=float, default=0,
                              help='Peak widths (gaussian) in Hz.')
    add_p_parser.set_defaults(func=add_peak)

    # Add defined sets of peaks to the basis set
    add_p_sets_parser = sp.add_parser(
        'add_set',
        help='Edit a basis set by adding a pre-specified peak sets e.g. default MM.')
    add_p_sets_parser.add_argument('file', type=Path,
                                   help='Basis file')
    add_p_sets_parser.add_argument('output', type=Path,
                                   help='Output location, can overwrite.')
    mutual_ps = add_p_sets_parser.add_mutually_exclusive_group(required=True)
    mutual_ps.add_argument('--add_MM', action="store_true",
                           help="include default macromolecule peaks")
    mutual_ps.add_argument('--add_MM_MEGA', action="store_true",
                           help="include default MEGA-PRESS macromolecule peaks. This option is experimental!")
    mutual_ps.add_argument('--add_water', action="store_true",
                           help="include water peak.")
    add_p_sets_parser.add_argument('--gamma', type=float, default=None,
                                   help='Peak widths (lorentzian) in Hz.')
    add_p_sets_parser.add_argument('--sigma', type=float, default=None,
                                   help='Peak widths (gaussian) in Hz.')
    add_p_sets_parser.set_defaults(func=add_peak_set)

    # Parse command-line arguments
    args = p.parse_args()

    # Call function
    args.func(args)


def info(args):
    """Prints basic information about Basis sets
    :param args: Argparse interpreted arguments
    :type args: Namespace
    """
    from fsl_mrs.utils.mrs_io import read_basis
    print(read_basis(args.file))


def vis(args):
    """Visualiser for Basis set
    :param args: Argparse interpreted arguments
    :type args: Namespace
    """
    from fsl_mrs.utils.mrs_io import read_basis
    import matplotlib.pyplot as plt

    # Some heuristics
    if args.file.is_dir():
        conj = True
    else:
        conj = False

    basis = read_basis(args.file)
    _ = basis.plot(ppmlim=args.ppmlim, conjugate=conj)
    plt.show()


def convert(args):
    """Converter for lcm/jmrui basis sets
    :param args: Argparse interpreted arguments
    :type args: Namespace
    """
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis
    from fsl_mrs.utils.constants import GYRO_MAG_RATIO

    if args.input.is_file():
        basis_tools.convert_lcm_basis(args.input, args.output)
    elif args.input.is_dir()\
            and (len(list(args.input.glob('*.raw'))) > 0 or len(list(args.input.glob('*.RAW'))) > 0):
        basis_tools.convert_lcm_raw_basis(
            args.input,
            args.bandwidth,
            args.fieldstrength * GYRO_MAG_RATIO['1H'],
            args.output)
    elif args.input.is_dir()\
            and len(list(args.input.glob('*.txt'))) > 0:
        basis_tools.convert_jmrui_basis(
            args.input,
            args.output)

    if args.remove_reference:
        # TODO sort this conjugation mess out.
        basis = read_basis(args.output)
        basis = basis_tools.conjugate_basis(basis)
        basis = basis_tools.remove_peak(
            basis,
            (-.2, .2),
            all=True,
            use_hlsvd=args.hlsvd)
        basis_tools.conjugate_basis(basis).save(args.output, overwrite=True)


def add(args):
    from fsl_mrs.utils.mrs_io import read_basis
    from fsl_mrs.utils import basis_tools
    import json
    import numpy as np

    with open(args.new, 'r') as fp:
        json_dict = json.load(fp)

    if 'basis' in json_dict:
        json_dict = json_dict['basis']

    fid = np.asarray(json_dict['basis_re']) + 1j * np.asarray(json_dict['basis_im'])
    if args.name:
        name = args.name
    else:
        name = json_dict['basis_name']
    cf = json_dict['basis_centre']
    bw = 1 / json_dict['basis_dwell']
    width = json_dict['basis_width']

    # Check that target exists
    target = Path(args.target)
    if not target.is_dir():
        raise NotADirectoryError('Target must be a directory of FSL-MRS basis (json) files')

    # Load target
    target_basis = read_basis(target)

    target_basis = basis_tools.add_basis(
        fid, name, cf, bw, target_basis,
        scale=args.scale,
        width=width,
        conj=args.conj,
        pad=args.pad)

    # Write to json without overwriting existing files
    target_basis.save(target, info_str=args.info)


def shift(args):
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis

    basis = read_basis(args.file)
    if args.metabolite == 'all':
        for name in basis.names:
            basis = basis_tools.shift_basis(basis, name, args.ppm_shift)
    else:
        basis = basis_tools.shift_basis(basis, args.metabolite, args.ppm_shift)

    basis.save(args.output, overwrite=True)


def all_shift(args):
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis
    import pandas as pd
    import numpy as np

    basis = read_basis(args.file)
    all_results = pd.read_csv(args.results_dir / 'all_parameters.csv', index_col=0)
    shift_res = all_results.filter(regex='eps', axis=0)['mean'] / (2 * np.pi * basis.cf)
    for name, shift in zip(basis.names, shift_res.to_list()):
        basis = basis_tools.shift_basis(basis, name, shift)

    basis.save(args.output, overwrite=True)


def rescale(args):
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis

    basis = read_basis(args.file)
    basis = basis_tools.rescale_basis(basis, args.metabolite, args.target_scale)

    basis.save(args.output, overwrite=True)


def diff(args):
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis

    basis1 = read_basis(args.file1)
    basis2 = read_basis(args.file2)

    if args.ignore_missing:
        missing_action = 'ignore'
    else:
        missing_action = 'raise'

    new = basis_tools.difference_basis_sets(
        basis1,
        basis2,
        add_or_subtract=args.add_or_sub,
        missing_metabs=missing_action)

    new.save(args.output)


def conj(args):
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis
    basis_tools.conjugate_basis(
        read_basis(args.file),
        name=args.metabolite
    ).save(args.output, overwrite=True)


def rem_peak(args):
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis
    basis_tools.remove_peak(
        read_basis(args.file),
        args.ppmlim,
        name=args.metabolite,
        all=args.all,
        use_hlsvd=args.hlsvd
    ).save(args.output, overwrite=True)


def add_peak(args):
    from fsl_mrs.utils.mrs_io import read_basis
    from fsl_mrs.utils import basis_tools

    basis = read_basis(args.file)
    names = basis.add_MM_peaks(args.ppm, args.amp, args.gamma, args.sigma, conj=True)
    for name in names:
        basis = basis_tools.rescale_basis(basis, name)
    basis.save(args.output, overwrite=True)


def add_peak_set(args):
    from fsl_mrs.utils import basis_tools
    from fsl_mrs.utils.mrs_io import read_basis
    import numpy as np

    basis = read_basis(args.file)
    all_original = basis.original_basis_array
    original_target = np.linalg.norm(np.mean(all_original, axis=1), axis=0)

    if args.add_MM:
        if args.gamma is None:
            gamma = 40
        else:
            gamma = args.gamma
        if args.sigma is None:
            sigma = 30
        else:
            sigma = args.sigma
        names = basis.add_default_MM_peaks(gamma, sigma, conj=False)
    elif args.add_MM_MEGA:
        if args.gamma is None:
            gamma = 10
        else:
            gamma = args.gamma
        if args.sigma is None:
            sigma = 0
        else:
            sigma = args.sigma
        names = basis.add_default_MEGA_MM_peaks(gamma, sigma, conj=False)
    elif args.add_water:
        if args.gamma is None:
            gamma = 0
        else:
            gamma = args.gamma
        if args.sigma is None:
            sigma = 0
        else:
            sigma = args.sigma
        names = basis.add_water_peak(gamma, sigma, conj=False)

    for name in names:
        basis = basis_tools.rescale_basis(basis, name, target_scale=original_target)

    basis.save(args.output, overwrite=True)


if __name__ == '__main__':
    main()
