'''FSL-MRS test script

Test the svs preprocessing script

Copyright Will Clarke, University of Oxford, 2021'''

import subprocess
from pathlib import Path

import pytest

from fsl_mrs.utils import mrs_io

testsPath = Path(__file__).parent
data = testsPath / 'testdata/fsl_mrs_preproc'
t1 = str(testsPath / 'testdata/svs_segment/T1.anat/T1_biascorr.nii.gz')
data_already_proc = testsPath / 'testdata' / 'fsl_mrs'


def test_preproc(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_raw.nii.gz')
    wrefq = str(data / 'quant_raw.nii.gz')
    ecc = str(data / 'ecc.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--align_limits', '0.5', '4.0',
         '--remove-water',
         '--truncate-fid', '1',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()
    assert (tmp_path / 'metab.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()

    proc_nii = mrs_io.read_FID(tmp_path / 'metab.nii.gz')
    assert proc_nii.shape == (1, 1, 1, 4095)


def test_depreciated_options(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_raw.nii.gz')
    wrefq = str(data / 'quant_raw.nii.gz')
    ecc = str(data / 'ecc.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--align_limits', '0.5', '4.0',
         '--hlsvd',
         '--leftshift', '1',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()
    assert (tmp_path / 'metab.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()

    proc_nii = mrs_io.read_FID(tmp_path / 'metab.nii.gz')
    assert proc_nii.shape == (1, 1, 1, 4095)


def test_noalign(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_raw.nii.gz')
    wrefq = str(data / 'quant_raw.nii.gz')
    ecc = str(data / 'ecc.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--noalign',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'metab.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()

    proc_nii = mrs_io.read_FID(tmp_path / 'metab.nii.gz')
    assert proc_nii.shape == (1, 1, 1, 4096)


def test_window_align_preproc(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_raw.nii.gz')
    wrefq = str(data / 'quant_raw.nii.gz')
    ecc = str(data / 'ecc.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--align_limits', '0.5', '4.0',
         '--align_window', '4',
         '--remove-water',
         '--truncate-fid', '1',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()
    assert (tmp_path / 'metab.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()

    proc_nii = mrs_io.read_FID(tmp_path / 'metab.nii.gz')
    assert proc_nii.shape == (1, 1, 1, 4095)


def test_preproc_wnoise(tmp_path):

    from fsl_mrs.core.nifti_mrs import create_nmrs
    import numpy as np
    cov = np.eye(32) / 2
    mean = np.zeros((32,))
    rng = np.random.default_rng(seed=1)
    noise = rng.multivariate_normal(mean, cov, (20000,))\
        + 1j * rng.multivariate_normal(mean, cov, (20000,))

    create_nmrs.gen_nifti_mrs(
        noise.reshape((1, 1, 1, ) + noise.shape),
        1 / 1000,
        123.2,
        dim_tags=['DIM_COIL', None, None]
    ).save(tmp_path / 'noise.nii.gz')

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_raw.nii.gz')
    out = tmp_path / 'out'
    retcode = subprocess.check_call(
        ['fsl_mrs_preproc',
         '--output', str(out),
         '--data', metab,
         '--reference', wrefc,
         '--noise', str(tmp_path / 'noise.nii.gz'),
         '--overwrite',
         '--verbose'])

    assert retcode == 0
    assert (out / 'metab.nii.gz').exists()
    assert (out / 'wref.nii.gz').exists()

    proc_nii = mrs_io.read_FID(out / 'metab.nii.gz')
    assert proc_nii.shape == (1, 1, 1, 4096)


def test_preproc_fmrs(tmp_path):

    metab = str(data / 'metab_raw.nii.gz')
    wrefc = str(data / 'wref_raw.nii.gz')
    wrefq = str(data / 'quant_raw.nii.gz')
    ecc = str(data / 'ecc.nii.gz')

    retcode = subprocess.check_call(
        ['fsl_mrs_preproc',
         '--output', str(tmp_path),
         '--data', metab,
         '--reference', wrefc,
         '--quant', wrefq,
         '--ecc', ecc,
         '--t1', t1,
         '--fmrs',
         '--truncate-fid', '1',
         '--overwrite',
         '--report',
         '--verbose'])

    assert retcode == 0
    assert (tmp_path / 'mergedReports.html').exists()
    assert (tmp_path / 'voxel_location.png').exists()
    assert (tmp_path / 'metab.nii.gz').exists()
    assert (tmp_path / 'wref.nii.gz').exists()

    proc_nii = mrs_io.read_FID(tmp_path / 'metab.nii.gz')
    assert proc_nii.shape == (1, 1, 1, 4095, 64)


def test_already_processed(tmp_path):

    metab = str(data_already_proc / 'metab.nii.gz')
    wref = str(data_already_proc / 'wref.nii.gz')

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _ = subprocess.run(
            ['fsl_mrs_preproc',
             '--output', str(tmp_path),
             '--data', metab,
             '--reference', wref,
             '--t1', t1,
             '--remove-water',
             '--truncate-fid', '1',
             '--overwrite',
             '--report'],
            check=True,
            capture_output=True)

    assert exc_info.type is subprocess.CalledProcessError
    assert exc_info.value.output == \
        b'This data contains no unaveraged transients or uncombined coils. '\
        b'Please carefully ascertain what pre-processing has already taken place, '\
        b'before running appropriate individual steps using fsl_mrs_proc. '\
        b'Note, no pre-processing may be necessary.\n'


def test_wref_singleton_preproc(tmp_path):
    """
    Test handling of reference data with singleton trailing dimensions
    """

    # Make data with right attributes
    from fsl_mrs.utils import mrs_io
    from fsl_mrs.utils.preproc import nifti_mrs_proc as nproc

    metab = data / 'metab_raw.nii.gz'
    wrefc = mrs_io.read_FID(data / 'wref_raw.nii.gz')
    wrefq = mrs_io.read_FID(data / 'quant_raw.nii.gz')
    ecc = mrs_io.read_FID(data / 'ecc.nii.gz')

    wrefc = nproc.average(wrefc, 'DIM_DYN')
    wrefq = nproc.average(wrefq, 'DIM_DYN')

    wrefc.set_dim_tag(5, 'DIM_DYN')
    assert wrefc.dim_tags[1] == 'DIM_DYN'
    assert wrefc.shape[5] == 1
    wrefc.save(tmp_path / 'wrefc.nii.gz')

    wrefq.set_dim_tag(5, 'DIM_DYN')
    assert wrefq.dim_tags[1] == 'DIM_DYN'
    assert wrefq.shape[5] == 1
    wrefq.save(tmp_path / 'wrefq.nii.gz')

    ecc.set_dim_tag(5, 'DIM_DYN')
    assert ecc.dim_tags[1] == 'DIM_DYN'
    assert ecc.shape[5] == 1
    ecc.save(tmp_path / 'ecc.nii.gz')

    out = subprocess.run(
        ['fsl_mrs_preproc',
         '--output', tmp_path / 'preproc',
         '--data', metab,
         '--reference', tmp_path / 'wrefc.nii.gz',
         '--quant', tmp_path / 'wrefq.nii.gz',
         '--ecc', tmp_path / 'ecc.nii.gz',
         '--t1', t1,
         '--align_limits', '0.5', '4.0',
         '--remove-water',
         '--truncate-fid', '1',
         '--overwrite',
         '--report',
         '--verbose'],
        check=True)

    assert out.returncode == 0
    assert (tmp_path / 'preproc' / 'mergedReports.html').exists()
    assert (tmp_path / 'preproc' / 'voxel_location.png').exists()
    assert (tmp_path / 'preproc' / 'metab.nii.gz').exists()
    assert (tmp_path / 'preproc' / 'wref.nii.gz').exists()

    proc_nii = mrs_io.read_FID(tmp_path / 'preproc' / 'metab.nii.gz')
    assert proc_nii.shape == (1, 1, 1, 4095)
