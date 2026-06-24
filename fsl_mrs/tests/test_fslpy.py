'''FSL-MRS test script

Test fslpy wrappers against the equivalent command line tools.

Author:  Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2026 University of Oxford
'''

import subprocess
from pathlib import Path
import sys
from unittest.mock import patch
import shutil
import pytest
pytest.importorskip('fsl.data.image')
pytest.importorskip('fsl.wrappers')

from fsl.data.image import Image
from fsl.wrappers import fsl_mrs, fsl_mrsi, fsl_mrs_preproc, fsl_mrs_preproc_edit, \
                         fsl_mrs_proc, svs_segment, mrsi_segment
from fsl_mrs import read_basis
from fsl_mrs.utils.validate_results import compare_folders

fsl_bin = str(Path(sys.prefix) / 'bin')

testsPath = Path(__file__).parent


def requires_cli(command):
    return pytest.mark.skipif(
        shutil.which(command) is None,
        reason=f'{command} is not available')


fsl_mrs_data = {
    'metab':    testsPath / 'testdata/fsl_mrs/metab.nii.gz',
    'water':    testsPath / 'testdata/fsl_mrs/wref.nii.gz',
    'basis':    testsPath / 'testdata/fsl_mrs/steam_basis',
    'seg':      testsPath / 'testdata/fsl_mrs/segmentation.json'}

fsl_mrsi_data = {
    'metab':    testsPath / 'testdata/fsl_mrsi/FID_Metab.nii.gz',
    'water':    testsPath / 'testdata/fsl_mrsi/FID_ref.nii.gz',
    'basis':    testsPath / 'testdata/fsl_mrsi/3T_slaser_32vespa_1250_wmm',
    'mask':     testsPath / 'testdata/fsl_mrsi/small_mask.nii.gz',
    'seg_wm':   testsPath / 'testdata/fsl_mrsi/mrsi_seg_wm.nii.gz',
    'seg_gm':   testsPath / 'testdata/fsl_mrsi/mrsi_seg_gm.nii.gz',
    'seg_csf':  testsPath / 'testdata/fsl_mrsi/mrsi_seg_csf.nii.gz'}

preproc_data = {
    'metab':    testsPath / 'testdata/fsl_mrs_preproc/metab_raw.nii.gz',
    'water':    testsPath / 'testdata/fsl_mrs_preproc/wref_raw.nii.gz',
    'quant':    testsPath / 'testdata/fsl_mrs_preproc/quant_raw.nii.gz',
    'ecc':      testsPath / 'testdata/fsl_mrs_preproc/ecc.nii.gz'}

preproc_edit_data = {
    'metab':    testsPath / 'testdata/fsl_mrs_preproc_edit/metab_raw.nii.gz',
    'wrefc':    testsPath / 'testdata/fsl_mrs_preproc_edit/wref_internal.nii.gz',
    'wrefq':    testsPath / 'testdata/fsl_mrs_preproc_edit/wref_quant.nii.gz',
    'ecc':      testsPath / 'testdata/fsl_mrs_preproc_edit/wref_internal.nii.gz',
    't1':       testsPath / 'testdata/svs_segment/T1.anat/T1_biascorr.nii.gz'}

svs_segment_data = {
    'metab':    testsPath / 'testdata/fsl_mrs/metab.nii.gz',
    'anat':     testsPath / 'testdata/svs_segment/T1.anat'}

mrsi_segment_data = {
    'metab':    testsPath / 'testdata/fsl_mrsi/FID_Metab.nii.gz',
    'anat':     testsPath / 'testdata/mrsi_segment/T1.anat'}


def _run_fsl_mrs_proc_cli(input_dir, output_dir):
    # 1. Combine water reference data for combination across dynamics
    file = input_dir['water']
    filename = output_dir / 'wref_comb'
    subprocess.check_call([
        'fsl_mrs_proc',
        'average',
        '--file', file,
        '--dim', 'DIM_DYN',
        '--output', output_dir,
        '--filename', filename])

    # 2. Run coil combination on the three sets of data
    for file in (input_dir['metab'], input_dir['quant'], input_dir['ecc']):
        subprocess.check_call([
            'fsl_mrs_proc',
            'coilcombine',
            '--file', file,
            '--reference', filename,
            '--output', output_dir])

    # 3. Align averages of water ref and metab data
    for file, filename, ppm in zip(
     [output_dir / input_dir['metab'].name, output_dir / input_dir['quant'].name],
     [output_dir / 'metab_align', output_dir / 'water_align'],
     [(1.8, 3.5), (4, 6)]):
        subprocess.check_call([
            'fsl_mrs_proc',
            'align',
            '--file', file,
            '--ppm', str(ppm[0]), str(ppm[1]),
            '--output', output_dir,
            '--filename', filename])

    # 4. Combine data across averages
    for file, filename in zip(
     [output_dir / 'metab_align', output_dir / 'water_align'],
     [output_dir / 'metab_comb', output_dir / 'wquant_comb']):
        subprocess.check_call([
            'fsl_mrs_proc',
            'average',
            '--file', file,
            '--dim', 'DIM_DYN',
            '--output', output_dir,
            '--filename', filename])

    # 5. Run the eddy current correction on the data
    for file, reference, filename in zip(
     [output_dir / 'metab_comb', output_dir / 'wquant_comb'],
     [output_dir / input_dir['ecc'].name, output_dir / 'wquant_comb'],
     [output_dir / 'metab_comb_ecc', output_dir / 'wquant_comb_ecc']):
        subprocess.check_call([
            'fsl_mrs_proc',
            'ecc',
            '--file', file,
            '--reference', reference,
            '--output', output_dir,
            '--filename', filename])

    # 6. Remove the first FID point
    for file in (output_dir / 'metab_comb_ecc',
                 output_dir / 'wquant_comb_ecc'):
        subprocess.check_call([
            'fsl_mrs_proc',
            'truncate',
            '--file', file,
            '--points', '-1',
            '--pos', 'first',
            '--output', output_dir])

    # 7. Run HLSVD on the data
    file = output_dir / 'metab_comb_ecc'
    filename = output_dir / 'metab_comb_ecc_hlsvd'
    subprocess.check_call([
        'fsl_mrs_proc',
        'remove',
        '--file', file,
        '--output', output_dir,
        '--filename', filename])

    # 8. Phase the data
    file = output_dir / 'metab_comb_ecc_hlsvd'
    filename = output_dir / 'metab'
    subprocess.check_call([
        'fsl_mrs_proc',
        'phase',
        '--file', file,
        '--filename', filename,
        '--output', output_dir])

    file = output_dir / 'wquant_comb_ecc'
    filename = output_dir / 'water'
    subprocess.check_call([
        'fsl_mrs_proc',
        'phase',
        '--file', file,
        '--ppm', '4.6', '4.7',
        '--filename', filename,
        '--output', output_dir])


def _run_fsl_mrs_proc_wrapper(input_dir, output_dir, use_objects=False):
    # 1. Combine water reference data for combination across dynamics
    file = input_dir['water']
    file = Image(file) if use_objects else file
    filename = output_dir / 'wref_comb'
    fsl_mrs_proc.average(file, output_dir, dim='DIM_DYN', filename=filename)

    # 2. Run coil combination on the three sets of data
    for file in (input_dir['metab'], input_dir['quant'], input_dir['ecc']):
        file = Image(file) if use_objects else file
        fsl_mrs_proc.coilcombine(file, output_dir, reference=filename)

    # 3. Align averages of water ref and metab data
    for file, filename, ppm in zip(
     [output_dir / input_dir['metab'].name, output_dir / input_dir['quant'].name],
     [output_dir / 'metab_align', output_dir / 'water_align'],
     [(1.8, 3.5), (4, 6)]):
        file = Image(file) if use_objects else file
        fsl_mrs_proc.align(file, output_dir, filename=filename, ppm=ppm)

    # 4. Combine data across averages
    for file, filename in zip(
     [output_dir / 'metab_align', output_dir / 'water_align'],
     [output_dir / 'metab_comb', output_dir / 'wquant_comb']):
        file = Image(file) if use_objects else file
        fsl_mrs_proc.average(file, output_dir, dim='DIM_DYN', filename=filename)

    # 5. Run the eddy current correction on the data
    for file, reference, filename in zip(
     [output_dir / 'metab_comb', output_dir / 'wquant_comb'],
     [output_dir / input_dir['ecc'].name, output_dir / 'wquant_comb'],
     [output_dir / 'metab_comb_ecc', output_dir / 'wquant_comb_ecc']):
        file = Image(file) if use_objects else file
        reference = Image(reference) if use_objects else reference
        fsl_mrs_proc.ecc(file, output=output_dir, reference=reference, filename=filename)

    # 6. Remove the first FID point
    for file in (output_dir / 'metab_comb_ecc',
                 output_dir / 'wquant_comb_ecc'):
        file = Image(file) if use_objects else file
        fsl_mrs_proc.truncate(file, output_dir, points=-1, pos='first')

    # 7. Run HLSVD on the data
    file = output_dir / 'metab_comb_ecc'
    file = Image(file) if use_objects else file
    filename = output_dir / 'metab_comb_ecc_hlsvd'
    fsl_mrs_proc.remove(file, output_dir, filename=filename)

    # 8. Phase the data
    file = output_dir / 'metab_comb_ecc_hlsvd'
    file = Image(file) if use_objects else file
    filename = output_dir / 'metab'
    fsl_mrs_proc.phase(file, output_dir, filename=filename)

    file = output_dir / 'wquant_comb_ecc'
    filename = output_dir / 'water'
    file = Image(file) if use_objects else file
    fsl_mrs_proc.phase(file, output_dir, filename=filename, ppm=(4.6, 4.7))


@requires_cli('fsl_mrs')
def test_fsl_mrs(tmp_path):
    cli_out     = tmp_path / 'fit_cli'
    wrapper_out = tmp_path / 'fit_wrapper'
    object_out  = tmp_path / 'fit_object'

    subprocess.check_call([
        'fsl_mrs',
        '--data', fsl_mrs_data['metab'],
        '--h2o', fsl_mrs_data['water'],
        '--output', cli_out,
        '--tissue_frac', fsl_mrs_data['seg'],
        '--overwrite',
        '--TE', '11',
        '--metab_groups', 'Mac',
        '--basis', fsl_mrs_data['basis'],
    ])
    assert (cli_out / 'summary.csv').exists()

    # fslpy call
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs(
            data=fsl_mrs_data['metab'],
            h2o=fsl_mrs_data['water'],
            output=wrapper_out,
            tissue_frac=fsl_mrs_data['seg'],
            overwrite=True,
            TE='11',
            metab_groups='Mac',
            basis=fsl_mrs_data['basis'],
        )
    assert (wrapper_out / 'summary.csv').exists()

    # fslpy call with objects
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs(
            data=Image(fsl_mrs_data['metab']),
            h2o=Image(fsl_mrs_data['water']),
            output=object_out,
            tissue_frac=fsl_mrs_data['seg'],
            overwrite=True,
            TE='11',
            metab_groups='Mac',
            basis=read_basis(fsl_mrs_data['basis']),
        )
    assert (object_out / 'summary.csv').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=False)
    assert compare_folders(cli_out, object_out,  subdir=False)


@requires_cli('fsl_mrsi')
def test_fsl_mrsi(tmp_path):
    cli_out     = tmp_path / 'fit_cli'
    wrapper_out = tmp_path / 'fit_wrapper'
    object_out  = tmp_path / 'fit_object'

    subprocess.check_call([
        'fsl_mrsi',
        '--data', fsl_mrsi_data['metab'],
        '--basis', fsl_mrsi_data['basis'],
        '--output', cli_out,
        '--metab_groups', 'MM09', 'MM12', 'MM14', 'MM17', 'MM21',
        '--h2o', fsl_mrsi_data['water'],
        '--TE', '30',
        '--TR', '2.0',
        '--mask', fsl_mrsi_data['mask'],
        '--tissue_frac',
        fsl_mrsi_data['seg_wm'],
        fsl_mrsi_data['seg_gm'],
        fsl_mrsi_data['seg_csf'],
        '--output_correlations',
        '--overwrite',
        '--combine', 'Cr', 'PCr',
    ])
    assert (cli_out / 'fit').exists()

    # fslpy call
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrsi(
            data=fsl_mrsi_data['metab'],
            basis=fsl_mrsi_data['basis'],
            output=wrapper_out,
            metab_groups=['MM09', 'MM12', 'MM14', 'MM17', 'MM21'],
            h2o=fsl_mrsi_data['water'],
            TE='30',
            TR='2.0',
            mask=fsl_mrsi_data['mask'],
            tissue_frac=[fsl_mrsi_data['seg_wm'],
                         fsl_mrsi_data['seg_gm'],
                         fsl_mrsi_data['seg_csf']],
            output_correlations=True,
            overwrite=True,
            combine=['Cr', 'PCr'],
        )
    assert (wrapper_out / 'fit').exists()

    # fslpy call with objects
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrsi(
            data=Image(fsl_mrsi_data['metab']),
            basis=read_basis(fsl_mrsi_data['basis']),
            output=object_out,
            metab_groups=['MM09', 'MM12', 'MM14', 'MM17', 'MM21'],
            h2o=Image(fsl_mrsi_data['water']),
            TE='30',
            TR='2.0',
            mask=Image(fsl_mrsi_data['mask']),
            tissue_frac=[fsl_mrsi_data['seg_wm'],
                         fsl_mrsi_data['seg_gm'],
                         fsl_mrsi_data['seg_csf']],
            output_correlations=True,
            overwrite=True,
            combine=['Cr', 'PCr'],
        )
    assert (object_out / 'fit').exists()

    # Fit time maps are wall-clock diagnostics and are expected to differ
    # between otherwise equivalent runs.
    (cli_out / 'misc/fit_times.nii.gz').unlink()
    (wrapper_out / 'misc/fit_times.nii.gz').unlink()
    (object_out / 'misc/fit_times.nii.gz').unlink()

    assert compare_folders(cli_out, wrapper_out, subdir=True)
    assert compare_folders(cli_out, object_out,  subdir=True)


@requires_cli('fsl_mrs_preproc')
def test_fsl_mrs_preproc(tmp_path):
    cli_out     = tmp_path / 'processed_cli'
    wrapper_out = tmp_path / 'processed_wrapper'
    object_out  = tmp_path / 'processed_object'

    subprocess.check_call([
        'fsl_mrs_preproc',
        '--data', preproc_data['metab'],
        '--reference', preproc_data['water'],
        '--quant', preproc_data['quant'],
        '--output', cli_out,
        '--truncate-fid', '1',
        '--remove-water',
        '--report',
        '--overwrite',
    ])
    assert (cli_out / 'metab.nii.gz').exists()

    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs_preproc(
            data=preproc_data['metab'],
            reference=preproc_data['water'],
            quant=preproc_data['quant'],
            output=wrapper_out,
            truncate_fid='1',
            remove_water=True,
            report=True,
            overwrite=True,
        )
    assert (wrapper_out / 'metab.nii.gz').exists()

    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs_preproc(
            data=Image(preproc_data['metab']),
            reference=Image(preproc_data['water']),
            quant=Image(preproc_data['quant']),
            output=object_out,
            truncate_fid='1',
            remove_water=True,
            report=True,
            overwrite=True,
        )
    assert (object_out / 'metab.nii.gz').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=False)
    assert compare_folders(cli_out, object_out,  subdir=False)


@requires_cli('fsl_mrs_preproc_edit')
def test_fsl_mrs_preproc_edit(tmp_path):
    cli_out     = tmp_path / 'processed_cli'
    wrapper_out = tmp_path / 'processed_wrapper'
    object_out  = tmp_path / 'processed_object'

    subprocess.check_call([
        'fsl_mrs_preproc_edit',
        '--data', preproc_edit_data['metab'],
        '--reference', preproc_edit_data['wrefc'],
        '--quant', preproc_edit_data['wrefq'],
        '--ecc', preproc_edit_data['ecc'],
        '--t1', preproc_edit_data['t1'],
        '--output', cli_out,
        '--truncate-fid', '2',
        '--remove-water',
        '--report',
        '--overwrite',
    ])
    assert (cli_out / 'diff.nii.gz').exists()

    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs_preproc_edit(
            data=preproc_edit_data['metab'],
            reference=preproc_edit_data['wrefc'],
            quant=preproc_edit_data['wrefq'],
            ecc=preproc_edit_data['ecc'],
            t1=preproc_edit_data['t1'],
            output=wrapper_out,
            truncate_fid='2',
            remove_water=True,
            report=True,
            overwrite=True,
        )
    assert (wrapper_out / 'diff.nii.gz').exists()

    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs_preproc_edit(
            data=Image(preproc_edit_data['metab']),
            reference=Image(preproc_edit_data['wrefc']),
            quant=Image(preproc_edit_data['wrefq']),
            ecc=Image(preproc_edit_data['ecc']),
            t1=Image(preproc_edit_data['t1']),
            output=object_out,
            truncate_fid='2',
            remove_water=True,
            report=True,
            overwrite=True,
        )
    assert (object_out / 'diff.nii.gz').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=False)
    assert compare_folders(cli_out, object_out,  subdir=False)


@requires_cli('svs_segment')
def test_svs_segment(tmp_path):
    cli_out     = tmp_path / 'out_cli'
    wrapper_out = tmp_path / 'out_wrapper'
    object_out  = tmp_path / 'out_object'

    cli_out.mkdir()
    subprocess.check_call([
        'svs_segment',
        svs_segment_data['metab'],
        '-a', svs_segment_data['anat'],
        '-o', cli_out,
    ])
    assert (cli_out / 'segmentation.json').exists()

    wrapper_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        svs_segment(
            svs=svs_segment_data['metab'],
            anat=svs_segment_data['anat'],
            output=wrapper_out,
        )
    assert (wrapper_out / 'segmentation.json').exists()

    object_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        svs_segment(
            svs=Image(svs_segment_data['metab']),
            anat=svs_segment_data['anat'],
            output=object_out,
        )
    assert (object_out / 'segmentation.json').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=False)
    assert compare_folders(cli_out, object_out,  subdir=False)


@requires_cli('mrsi_segment')
def test_mrsi_segment(tmp_path):
    cli_out     = tmp_path / 'out_cli'
    wrapper_out = tmp_path / 'out_wrapper'
    object_out  = tmp_path / 'out_object'

    cli_out.mkdir()
    subprocess.check_call([
        'mrsi_segment',
        mrsi_segment_data['metab'],
        '-a', mrsi_segment_data['anat'],
        '-o', cli_out,
    ])
    assert (cli_out / 'mrsi_seg_wm.nii.gz').exists()

    wrapper_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        mrsi_segment(
            mrsi=mrsi_segment_data['metab'],
            anat=mrsi_segment_data['anat'],
            output=wrapper_out,
        )
    assert (wrapper_out / 'mrsi_seg_wm.nii.gz').exists()

    object_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        mrsi_segment(
            mrsi=Image(mrsi_segment_data['metab']),
            anat=mrsi_segment_data['anat'],
            output=object_out,
        )
    assert (object_out / 'mrsi_seg_wm.nii.gz').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=False)
    assert compare_folders(cli_out, object_out,  subdir=False)


@requires_cli('fsl_mrs_proc')
def test_fsl_mrs_proc(tmp_path):
    cli_out     = tmp_path / 'proc_cli'
    wrapper_out = tmp_path / 'proc_wrapper'
    object_out  = tmp_path / 'proc_object'

    cli_out.mkdir()
    _run_fsl_mrs_proc_cli(preproc_data, cli_out)
    assert (cli_out / 'metab.nii.gz').exists()

    wrapper_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        _run_fsl_mrs_proc_wrapper(preproc_data, wrapper_out)
    assert (wrapper_out / 'metab.nii.gz').exists()

    object_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        _run_fsl_mrs_proc_wrapper(preproc_data, object_out, use_objects=True)
    assert (object_out / 'metab.nii.gz').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=False)
    assert compare_folders(cli_out, object_out,  subdir=False)
