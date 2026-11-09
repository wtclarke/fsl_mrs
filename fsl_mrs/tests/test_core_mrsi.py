'''FSL-MRS test script

Test core MRSI class.

Copyright Will Clarke, University of Oxford, 2021'''

from fsl_mrs.core import MRSI, mrsi_from_files
from pathlib import Path
from fsl_mrs import read_FID, read_basis
import numpy as np
import nibabel as nib

testsPath = Path(__file__).parent
data = {'metab': testsPath / 'testdata/fsl_mrsi/FID_Metab.nii.gz',
        'water': testsPath / 'testdata/fsl_mrsi/FID_ref.nii.gz',
        'basis': testsPath / 'testdata/fsl_mrsi/3T_slaser_32vespa_1250.BASIS',
        'mask': testsPath / 'testdata/fsl_mrsi/small_mask.nii.gz',
        'seg_wm': testsPath / 'testdata/fsl_mrsi/mrsi_seg_wm.nii.gz',
        'seg_gm': testsPath / 'testdata/fsl_mrsi/mrsi_seg_gm.nii.gz',
        'seg_csf': testsPath / 'testdata/fsl_mrsi/mrsi_seg_csf.nii.gz'}


def test_manual_load():

    fid = read_FID(str(data['metab']))
    fid_w = read_FID(str(data['water']))
    basis = read_basis(str(data['basis']))

    mrsi = MRSI(fid,
                cf=fid.spectrometer_frequency[0],
                bw=fid.bandwidth,
                nucleus=fid.nucleus[0],
                basis=basis, H2O=fid_w)

    assert mrsi.spatial_shape == (48, 48, 1)
    assert mrsi.num_voxels == 2304
    assert mrsi.FID_points == 1024
    assert mrsi.data.shape == (48, 48, 1, 1024)
    assert mrsi.H2O.shape == (48, 48, 1, 1024)
    assert mrsi.numBasis == 22

    def loadNii(f):
        nii = np.asanyarray(nib.load(f).dataobj)
        if nii.ndim == 2:
            nii = np.expand_dims(nii, 2)
        return nii

    mask = loadNii(str(data['mask']))
    mrsi.set_mask(mask)

    assert mrsi.num_masked_voxels == 6
    assert mrsi.mask.shape == (48, 48, 1)

    wm = loadNii(str(data['seg_wm']))
    gm = loadNii(str(data['seg_gm']))
    csf = loadNii(str(data['seg_csf']))
    mrsi.set_tissue_seg(csf, wm, gm)

    assert mrsi.tissue_seg_loaded
    assert mrsi.csf.shape == (48, 48, 1)
    assert mrsi.wm.shape == (48, 48, 1)
    assert mrsi.gm.shape == (48, 48, 1)


def test_load_from_file():
    mrsi = mrsi_from_files(str(data['metab']),
                           mask_file=str(data['mask']),
                           basis_file=str(data['basis']),
                           H2O_file=str(data['water']),
                           csf_file=str(data['seg_csf']),
                           gm_file=str(data['seg_gm']),
                           wm_file=str(data['seg_wm']))

    assert mrsi.spatial_shape == (48, 48, 1)
    assert mrsi.data.shape == (48, 48, 1, 1024)
    assert mrsi.H2O.shape == (48, 48, 1, 1024)
    assert mrsi.numBasis == 22

    assert mrsi.tissue_seg_loaded
    assert mrsi.csf.shape == (48, 48, 1)
    assert mrsi.wm.shape == (48, 48, 1)
    assert mrsi.gm.shape == (48, 48, 1)


def test_fetch_mrs():

    mrsi = mrsi_from_files(str(data['metab']),
                           mask_file=str(data['mask']),
                           basis_file=str(data['basis']),
                           H2O_file=str(data['water']),
                           csf_file=str(data['seg_csf']),
                           gm_file=str(data['seg_gm']),
                           wm_file=str(data['seg_wm']))

    iter_indices = mrsi.get_indices_in_order(mask=True)

    fid = read_FID(str(data['metab']))

    for idx, (mrs, index, seg) in enumerate(mrsi):
        assert np.allclose(mrs.FID, fid[iter_indices[idx]])
        assert index == iter_indices[idx]
        assert np.isclose(seg['CSF'] + seg['WM'] + seg['GM'], 1.0)

    # test some of the flags
    mrsi.rescale        = True
    mrsi.keep           = ['NAA']

    for idx, (mrs, index, seg) in enumerate(mrsi):
        assert mrs.names == ['NAA']
        assert np.allclose(mrs.FID / mrs.scaling['FID'], fid[iter_indices[idx]])


def test_fetch_mrs_reuses_average_basis_cache():

    mrsi = mrsi_from_files(str(data['metab']),
                           mask_file=str(data['mask']),
                           basis_file=str(data['basis']),
                           H2O_file=str(data['water']))

    avg_mrs = mrsi.mrs_from_average()
    _ = avg_mrs.basis
    formatted_cache_size = len(avg_mrs._basis._formatted_basis_cache)

    voxel_mrs = mrsi.mrs_by_index(mrsi.get_indices_in_order(mask=True)[0])

    assert voxel_mrs._basis is avg_mrs._basis
    _ = voxel_mrs.basis
    assert len(avg_mrs._basis._formatted_basis_cache) == formatted_cache_size
