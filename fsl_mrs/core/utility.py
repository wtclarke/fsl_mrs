'''utility.py -Module containing utility functions
for creating MRS, MRSI and NIFTI_MRS objects
Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
        Will Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 2021 University of Oxford
# SHBASECOPYRIGHT
'''
import nibabel as nib
import numpy as np
import fsl.utils.path as fslpath

from fsl_mrs.utils import mrs_io
from fsl_mrs.core.nifti_mrs import NIFTI_MRS, NotNIFTI_MRS


def mrs_from_files(FID_file, Basis_file, H2O_file=None):
    '''Construct an MRS object from FID, basis, and
     (optionally) a reference file

    :param FID_file: path to data file
    :param Basis_file: path to basis file
    :param H2O_file: Optional path to reference file

    :return mrs: MRS object
    '''

    FID = mrs_io.read_FID(FID_file)
    basis = mrs_io.read_basis(Basis_file)
    if H2O_file is not None:
        H2O = mrs_io.read_FID(H2O_file).data
    else:
        H2O = None

    return FID.mrs(basis=basis, ref_data=H2O)


def mrsi_from_files(data_file,
                    mask_file=None,
                    basis_file=None,
                    H2O_file=None,
                    csf_file=None,
                    gm_file=None,
                    wm_file=None):
    '''Construct an MRS object from data, and
     (optionally) basis, mask, reference and segmentation files

    :param FID_file: path to data file
    :param mask_file: Optional path to basis file
    :param basis_file: Optional path to reference file
    :param H2O_file: Optional path to reference file
    :param csf_file: Optional path to reference file
    :param gm_file: Optional path to reference file
    :param wm_file: Optional path to reference file

    :return mrs: MRSI object
    '''
    data = mrs_io.read_FID(data_file)
    if mask_file is not None:
        nib_img = nib.load(mask_file)
        mask = np.asanyarray(nib_img.dataobj)
    else:
        mask = None

    if basis_file is not None:
        basis = mrs_io.read_basis(basis_file)
    else:
        basis = None
    if H2O_file is not None:
        data_w = mrs_io.read_FID(H2O_file)
    else:
        data_w = None

    out = data.mrs(basis=basis, ref_data=data_w)

    out.set_mask(mask)

    def loadNii(f):
        nii = np.asanyarray(nib.load(f).dataobj)
        if nii.ndim == 2:
            nii = np.expand_dims(nii, 2)
        return nii

    if (csf_file is not None) and (gm_file is not None) and (wm_file is not None):
        csf = loadNii(csf_file)
        gm = loadNii(gm_file)
        wm = loadNii(wm_file)
        out.set_tissue_seg(csf, wm, gm)

    return out


def is_nifti_mrs(file_path):
    '''Check that a file is of the NIFTI-MRS format type.'''
    try:
        NIFTI_MRS(file_path)
        return True
    except fslpath.PathError:
        raise NotNIFTI_MRS("File isn't NIFTI-MRS, wrong extension type.")
