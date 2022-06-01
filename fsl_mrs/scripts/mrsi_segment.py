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
from fsl.wrappers import fsl_anat
from fsl.wrappers.fnirt import applywarp
import numpy as np
from fsl.data.image import Image


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
    args = parser.parse_args()

    # For windows implementations we must supply absolute
    # paths. This enables conversion to wsl paths.
    # The fslpy wrapper code requires a string rather than pathlib Path.
    def str_resolve_path(pathlib_path):
        return str(pathlib_path.resolve())

    # If not prevented run fsl_anat for fast segmentation
    if (args.anat is None) and (not args.mask_only):
        anat = args.output  / 'fsl_anat'
        fsl_anat(
            str_resolve_path(args.t1),
            out=str_resolve_path(anat),
            nosubcortseg=True)
        anat = anat.with_suffix('.anat')
    else:
        anat = args.anat

    # Make dummy nifti as nothing works with complex data
    mrsi_in = Image(args.mrsi)
    tmp_img = np.zeros(mrsi_in.shape[0:3])
    tmp_img = Image(tmp_img, xform=mrsi_in.voxToWorldMat)

    # Register the pvseg to the MRSI data using flirt
    def applywarp_func(i, o):
        applywarp(str_resolve_path(i),
                  tmp_img,
                  str_resolve_path(o),
                  usesqform=True,
                  super=True,
                  superlevel='a')

    # T1_fast_pve_0, T1_fast_pve_1, T1_fast_pve_2
    # partial volume segmentations (CSF, GM, WM respectively)
    applywarp_func(anat / 'T1_fast_pve_0.nii.gz',
                   args.output / (args.filename + '_csf.nii.gz'))
    applywarp_func(anat / 'T1_fast_pve_1.nii.gz',
                   args.output / (args.filename + '_gm.nii.gz'))
    applywarp_func(anat / 'T1_fast_pve_2.nii.gz',
                   args.output / (args.filename + '_wm.nii.gz'))


if __name__ == '__main__':
    main()
