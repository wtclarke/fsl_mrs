#!/usr/bin/env python

# mrsi_segment - use fsl to segment a T1 and register it to an mrsi scan
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

# Quick imports
import argparse
from pathlib import Path
import sys
# More imports after argument parsing!

os_name = sys.platform
win_platforms = {'win32', 'cygwin'}


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="FSL Magnetic Resonance Spectroscopy"
                    " - register fast segmentation to mrsi.")

    parser.add_argument('mrsi', type=str, metavar='MRSI',
                        help='MRSI nifti file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--t1', type=Path, metavar='T1',
                       help='T1 nifti file')
    group.add_argument('-a', '--anat', type=Path,
                       help='fsl_anat output directory.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Output directory', default='.')
    parser.add_argument('-f', '--filename', type=str,
                        help='Output file name', default='mrsi_seg')
    parser.add_argument(
        '--no_normalisation',
        action="store_false",
        dest='normalise',
        help='Do not normalise output to 1 in all voxels.')
    args = parser.parse_args()

    # Imports post argparse:
    from fsl.wrappers import fsl_anat, fslmaths
    from fsl.wrappers.fnirt import applywarp
    import numpy as np
    from fsl.data.image import Image
    from fsl.utils.run import FSLNotPresent

    # For windows implementations we must supply absolute
    # paths. This enables conversion to wsl paths.
    # The fslpy wrapper code requires a string rather than pathlib Path.
    def str_resolve_path(pathlib_path):
        return str(pathlib_path.resolve())

    # If not prevented run fsl_anat for fast segmentation
    if args.anat is None:
        anat = args.output  / 'fsl_anat'
        try:
            fsl_anat(
                str_resolve_path(args.t1),
                out=str_resolve_path(anat),
                nosubcortseg=True)
        except FileNotFoundError:
            if os_name in win_platforms:
                msg = 'FSL tool fsl_anat not found. It is not installable on Windows, \
                    unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool fsl_anat not found. Please install FSL.'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m")
        except FSLNotPresent:
            raise FSLNotPresent("$FSLDIR is not set - please use: 'export FSLDIR=${CONDA_PREFIX}'")
        anat = anat.with_suffix('.anat')
    else:
        anat = args.anat

    # Make dummy nifti as nothing works with complex data
    mrsi_in = Image(args.mrsi)
    tmp_img = np.zeros(mrsi_in.shape[0:3])
    tmp_img = Image(tmp_img, xform=mrsi_in.voxToWorldMat)

    # HACK because applywarp is shite - Paul McCarthy 2022 ;-)
    tmp_img.header.set_sform(tmp_img.header.get_qform())
    # Can remove in FSL 6.1.0 when applywarp is fixed

    # Register the pvseg to the MRSI data using flirt
    def applywarp_func(i, o):
        try:
            applywarp(str_resolve_path(i),
                      tmp_img,
                      str_resolve_path(o),
                      usesqform=True,
                      super=True,
                      superlevel='a')
        except FileNotFoundError:
            if os_name in win_platforms:
                msg = 'FSL tool applywarp not found. It is not installable on Windows, \
                    unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool applywarp not found. Please install FSL or fsl-fugue using conda.'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m")
        except FSLNotPresent:
            raise FSLNotPresent("$FSLDIR is not set - please use: 'export FSLDIR=${CONDA_PREFIX}'")

    # T1_fast_pve_0, T1_fast_pve_1, T1_fast_pve_2
    # partial volume segmentations (CSF, GM, WM respectively)
    csf_output = args.output / (args.filename + '_csf.nii.gz')
    gm_output = args.output / (args.filename + '_gm.nii.gz')
    wm_output = args.output / (args.filename + '_wm.nii.gz')

    applywarp_func(anat / 'T1_fast_pve_0.nii.gz',
                   csf_output)
    applywarp_func(anat / 'T1_fast_pve_1.nii.gz',
                   gm_output)
    applywarp_func(anat / 'T1_fast_pve_2.nii.gz',
                   wm_output)

    if args.normalise:
        try:
            fslmaths(csf_output)\
                .add(gm_output)\
                .add(wm_output)\
                .run(args.output / 'tmp_sum')

            fslmaths(csf_output)\
                .div(args.output / 'tmp_sum')\
                .run(csf_output)

            fslmaths(gm_output)\
                .div(args.output / 'tmp_sum')\
                .run(gm_output)

            fslmaths(wm_output)\
                .div(args.output / 'tmp_sum')\
                .run(wm_output)

        except FileNotFoundError:
            if os_name in win_platforms:
                msg = 'FSL tool fslmaths not found. It is not installable on Windows, \
                    unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool fslmaths not found. Please install FSL or fsl-avwutils using conda.'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m")
        except FSLNotPresent:
            raise FSLNotPresent("$FSLDIR is not set - please use: 'export FSLDIR=${CONDA_PREFIX}'")

        (args.output / 'tmp_sum.nii.gz').unlink()


if __name__ == '__main__':
    main()
