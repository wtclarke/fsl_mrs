"""basis2spec - Create a NIfTI-MRS spectrum from a basis set

Author: William Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 2026 University of Oxford"""

# Quick imports
from pathlib import Path

import argparse


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="FSL-MRS: create a NIfTI-MRS formatted spectrum from a basis set",
        add_help=True)

    parser.add_argument(
        '--basis',
        required=True,
        type=Path,
        metavar='<BASIS DIR>',
        help='FSL-MRS formatted basis folder containing basis spectra')
    parser.add_argument(
        '--reference',
        required=True,
        type=Path,
        help='Reference spectrum. Number of spectral points and dwell time set from reference. NIfTI-MRS format.')
    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        metavar='<PATH>',
        help='Output file path, parent folders created if required.')
    parser.add_argument(
        '--linewidth',
        required=False,
        type=float,
        default=0,
        help='Lorentzian broadening applied to basis spectrum. Default is no broadening.')
    parser.add_argument(
        '--ignore',
        required=False,
        nargs='+',
        type=str,
        help='Lorentzian broadening applied to basis spectrum.')

    args = parser.parse_args()

    from fsl_mrs.utils.synthetic import syntheticFromBasisFile
    from math import pi
    from fsl_mrs.utils import mrs_io

    ref = mrs_io.read_FID(args.reference)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    syntheticFromBasisFile(
        basisFile=args.basis,
        ignore=args.ignore,
        broadening=(args.linewidth * pi, 0),
        noisecovariance=[[0.0]],
        points=ref.shape[3],
        bandwidth=ref.bandwidth,
        nifti_output=True
    )[0].save(args.output)


if __name__ == '__main__':
    main()
