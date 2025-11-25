#!/usr/bin/env python

# svs_segment - use fsl to make a mask from a svs voxel and T1 nifti,
# then produce tissue segmentation file.
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

# Quick imports
import argparse
from pathlib import Path
import json
import warnings
import sys
# More imports below after argument parsing!

os_name = sys.platform
win_platforms = {'win32', 'cygwin'}


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="FSL Magnetic Resonance Spectroscopy"
                    " - Construct mask in T1 space of an SVS voxel"
                    " and generate a tissue segmentation file.")

    parser.add_argument('svs', type=Path, metavar='SVS',
                        help='SVS nifti file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--t1', type=Path, metavar='T1',
                       help='T1 nifti file')
    group.add_argument('-a', '--anat', type=Path,
                       help='fsl_anat output directory.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Output directory', default=Path.cwd())
    parser.add_argument('-f', '--filename', type=str,
                        help='file name stem. _mask.nii.gz'
                             ' or _segmentation.json will be added.',
                        default=None)
    parser.add_argument('-m', '--mask_only', action="store_true",
                        help='Only perform masking stage,'
                             ' do not run fsl_anat if only T1 passed.')
    parser.add_argument(
        '--no_normalisation',
        action="store_false",
        dest='normalise',
        help='Do not normalise output to 1 in all voxels.')
    args = parser.parse_args()

    # Imports post argparse
    import nibabel as nib
    import numpy as np
    from fsl.wrappers import flirt, fslstats, fsl_anat, fslmaths

    # For windows implementations we must supply absolute
    # paths. This enables conversion to wsl paths.
    # The fslpy wrapper code requires a string rather than pathlib Path.
    def str_resolve_path(pathlib_path):
        return str(pathlib_path.resolve())

    # If not prevented run fsl_anat for fast segmentation
    if (args.anat is None) and (not args.mask_only):
        anat = args.output  / 'fsl_anat'
        try:
            fsl_anat(
                str_resolve_path(args.t1),
                out=str_resolve_path(anat),
                nosubcortseg=True)
        except FileNotFoundError:
            if os_name in win_platforms:
                msg = 'FSL tool fsl_anat not found. It is not installable on Windows, unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool fsl_anat not found. Please install FSL.'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m")
        anat = anat.with_suffix('.anat')
    else:
        anat = args.anat

    data_hdr = nib.load(args.svs)

    # Create 3D mock data
    mockData = np.zeros((2, 2, 2))
    mockData[0, 0, 0] = 1.0
    tmp = nib.Nifti1Image(mockData, affine=data_hdr.affine)

    # Run flirt
    if anat is not None:
        flirt_ref = anat / 'T1_biascorr.nii.gz'
    else:
        flirt_ref = args.t1
    flirt_ref = str_resolve_path(flirt_ref)

    if args.filename is None:
        mask_name = 'mask.nii.gz'
    else:
        mask_name = args.filename + '_mask.nii.gz'
    flirt_out = str_resolve_path(args.output / mask_name)

    try:
        flirt(tmp,
            flirt_ref,
            out=flirt_out,
            usesqform=True,
            applyxfm=True,
            noresampblur=True,
            interp='nearestneighbour',
            setbackground=0,
            paddingsize=1)
    except FileNotFoundError:
        if os_name in win_platforms:
            msg = 'FSL tool flirt not found. It is not installable on Windows, unless you follow the WSL instructions in the FSL-MRS documentation.'
        else:
            msg = 'FSL tool flirt not found. Please install FSL or fsl-flirt using conda.'
        raise FileNotFoundError("\033[91m"+msg+"\033[0m")
        

    # Provide tissue segmentation if anat is available
    if anat is not None:
        # Check that the svs mask intersects with brain, issue warning if not.
        mask_path = str_resolve_path(anat / 'T1_biascorr_brain_mask.nii.gz')
        try:
            tmp_out = fslmaths(flirt_out)\
                .add(mask_path)\
                .mas(flirt_out)\
                .run()
        except FileNotFoundError:
            if os_name in win_platforms:
                msg = 'FSL tool fslmaths not found. It is not installable on Windows, unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool fslmaths not found. Please install FSL or fsl-avwutils using conda.'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m")

        try:
            meanInVox = fslstats(tmp_out).M.run()
        except FileNotFoundError:
            if os_name in win_platforms:
                msg = 'FSL tool fslstats not found. It is not installable on Windows, unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool fslstats not found. Please install FSL or fsl-avwutils using conda.'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m")
            
        if meanInVox < 2.0:
            warnings.warn('The mask does not fully intersect'
                          ' with the brain mask. Check manually.')

        # Count up segmentation values in mask.
        seg_csf = str_resolve_path(anat / 'T1_fast_pve_0.nii.gz')
        seg_gm = str_resolve_path(anat / 'T1_fast_pve_1.nii.gz')
        seg_wm = str_resolve_path(anat / 'T1_fast_pve_2.nii.gz')

        # fslstats -t /fast_output/fast_output_pve_0 -k SVS_mask.nii –m
        try:
            CSF = fslstats(seg_csf).k(flirt_out).m.run()
            GM = fslstats(seg_gm).k(flirt_out).m.run()
            WM = fslstats(seg_wm).k(flirt_out).m.run()
        except FileNotFoundError:
            if os_name in win_platforms:
                msg = 'FSL tool fslstats not found. It is not installable on Windows, unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool fslstats not found. Please install FSL or fsl-avwutils using conda.'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m")

        if args.normalise:
            sum_val = CSF + GM + WM
            CSF /= sum_val
            GM /= sum_val
            WM /= sum_val

        print(f'CSF: {CSF:0.2f}, GM: {GM:0.2f}, WM: {WM:0.2f}.')
        segresults = {'CSF': CSF, 'GM': GM, 'WM': WM}

        if args.filename is None:
            seg_name = 'segmentation.json'
        else:
            seg_name = args.filename + '_segmentation.json'

        with open(args.output / seg_name, 'w', encoding='utf-8') as f:
            json.dump(segresults, f, ensure_ascii=False, indent='\t')


if __name__ == '__main__':
    main()
