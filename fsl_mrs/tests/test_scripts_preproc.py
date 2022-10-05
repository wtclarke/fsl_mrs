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
             '--hlsvd',
             '--leftshift', '1',
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
