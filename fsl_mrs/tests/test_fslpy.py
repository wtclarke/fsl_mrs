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
                         fsl_mrs_proc, svs_segment, mrsi_segment, fsl_dynmrs, \
                         basis2spec, fmrs_stats
from fsl_mrs import read_basis, NIFTI_MRS
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

basis2spec_data = {
    'basis':    testsPath / 'testdata/fsl_mrs/steam_basis_no_mm',
    'reference': testsPath / 'testdata/fsl_mrs/metab.nii.gz'}

fmrs_stats_data = {
    'sim_results': [
        testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/stim',
        testsPath / 'testdata/fmrs_tools/sim_fmrs/sub1/stim',
        testsPath / 'testdata/fmrs_tools/sim_fmrs/sub0/ctrl',
        testsPath / 'testdata/fmrs_tools/sim_fmrs/sub1/ctrl'],
    'fl_contrasts': testsPath / 'testdata/fmrs_tools/fl_contrasts.json',
    'design_gm': testsPath / 'testdata/fmrs_tools/design_groupmean.mat',
    'con_gm': testsPath / 'testdata/fmrs_tools/design_groupmean.con'}


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
    file = NIFTI_MRS(file) if use_objects else file
    filename = output_dir / 'wref_comb'
    fsl_mrs_proc.average(file, output_dir, dim='DIM_DYN', filename=filename)

    # 2. Run coil combination on the three sets of data
    for file in (input_dir['metab'], input_dir['quant'], input_dir['ecc']):
        file = NIFTI_MRS(file) if use_objects else file
        fsl_mrs_proc.coilcombine(file, output_dir, reference=filename)

    # 3. Align averages of water ref and metab data
    for file, filename, ppm in zip(
     [output_dir / input_dir['metab'].name, output_dir / input_dir['quant'].name],
     [output_dir / 'metab_align', output_dir / 'water_align'],
     [(1.8, 3.5), (4, 6)]):
        file = NIFTI_MRS(file) if use_objects else file
        fsl_mrs_proc.align(file, output_dir, filename=filename, ppm=ppm)

    # 4. Combine data across averages
    for file, filename in zip(
     [output_dir / 'metab_align', output_dir / 'water_align'],
     [output_dir / 'metab_comb', output_dir / 'wquant_comb']):
        file = NIFTI_MRS(file) if use_objects else file
        fsl_mrs_proc.average(file, output_dir, dim='DIM_DYN', filename=filename)

    # 5. Run the eddy current correction on the data
    for file, reference, filename in zip(
     [output_dir / 'metab_comb', output_dir / 'wquant_comb'],
     [output_dir / input_dir['ecc'].name, output_dir / 'wquant_comb'],
     [output_dir / 'metab_comb_ecc', output_dir / 'wquant_comb_ecc']):
        file = NIFTI_MRS(file) if use_objects else file
        reference = NIFTI_MRS(reference) if use_objects else reference
        fsl_mrs_proc.ecc(file, output=output_dir, reference=reference, filename=filename)

    # 6. Remove the first FID point
    for file in (output_dir / 'metab_comb_ecc',
                 output_dir / 'wquant_comb_ecc'):
        file = NIFTI_MRS(file) if use_objects else file
        fsl_mrs_proc.truncate(file, output_dir, points=-1, pos='first')

    # 7. Run HLSVD on the data
    file = output_dir / 'metab_comb_ecc'
    file = NIFTI_MRS(file) if use_objects else file
    filename = output_dir / 'metab_comb_ecc_hlsvd'
    fsl_mrs_proc.remove(file, output_dir, filename=filename)

    # 8. Phase the data
    file = output_dir / 'metab_comb_ecc_hlsvd'
    file = NIFTI_MRS(file) if use_objects else file
    filename = output_dir / 'metab'
    fsl_mrs_proc.phase(file, output_dir, filename=filename)

    file = output_dir / 'wquant_comb_ecc'
    filename = output_dir / 'water'
    file = NIFTI_MRS(file) if use_objects else file
    fsl_mrs_proc.phase(file, output_dir, filename=filename, ppm=(4.6, 4.7))


def _create_fsl_dynmrs_data(tmp_path):
    import numpy as np
    from fsl_mrs.core import basis
    import fsl_mrs.utils.synthetic as syn
    from fsl_mrs.core.nifti_mrs import gen_nifti_mrs

    FID_basis1 = syn.syntheticFID(chemicalshift=[1,], amplitude=[1], noisecovariance=[[0]], damping=[3])
    FID_basis2 = syn.syntheticFID(chemicalshift=[3,], amplitude=[1], noisecovariance=[[0]], damping=[3])
    bset = basis.Basis(
        np.stack((FID_basis1[0][0], FID_basis2[0][0]), axis=1),
        ['Met1', 'Met2'],
        axes=FID_basis1[2])
    bset.basis_fwhm = [3 * np.pi, 3 * np.pi]

    FID1 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[1, 1], noisecovariance=[[0.01]])
    FID2 = syn.syntheticFID(chemicalshift=[1, 3], amplitude=[2, 2], noisecovariance=[[0.01]])
    fid1 = FID1[0][0].reshape((1, 1, 1, 2048))
    fid2 = FID2[0][0].reshape((1, 1, 1, 2048))
    data = np.stack((fid1, fid2), axis=-1)
    data = np.conj(data)

    nmrs = gen_nifti_mrs(
        data,
        FID1[2].dwelltime,
        FID1[2].SpectrometerFrequency,
        dim_tags=['DIM_DYN', None, None])

    time_var = np.arange(2)

    # Save
    data_path = tmp_path / 'data.nii.gz'
    basis_path = tmp_path / 'basis'
    tv_path = tmp_path / 'time_var.csv'

    nmrs.save(data_path)
    bset.save(basis_path)
    np.savetxt(tv_path, time_var, delimiter=',')

    return data_path, basis_path, tv_path


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
            data=NIFTI_MRS(fsl_mrs_data['metab']),
            h2o=NIFTI_MRS(fsl_mrs_data['water']),
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

    # fslpy call
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

    # fslpy call with objects
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs_preproc(
            data=NIFTI_MRS(preproc_data['metab']),
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

    # fslpy call
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

    # fslpy call with objects
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_mrs_preproc_edit(
            data=Image(preproc_edit_data['metab']),
            reference=NIFTI_MRS(preproc_edit_data['wrefc']),
            quant=NIFTI_MRS(preproc_edit_data['wrefq']),
            ecc=NIFTI_MRS(preproc_edit_data['ecc']),
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

    # fslpy call
    wrapper_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        svs_segment(
            svs=svs_segment_data['metab'],
            anat=svs_segment_data['anat'],
            output=wrapper_out,
        )
    assert (wrapper_out / 'segmentation.json').exists()

    # fslpy call with objects
    object_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        svs_segment(
            svs=NIFTI_MRS(svs_segment_data['metab']),
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

    # fslpy call
    wrapper_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        mrsi_segment(
            mrsi=mrsi_segment_data['metab'],
            anat=mrsi_segment_data['anat'],
            output=wrapper_out,
        )
    assert (wrapper_out / 'mrsi_seg_wm.nii.gz').exists()

    # fslpy call with objects
    object_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        mrsi_segment(
            mrsi=NIFTI_MRS(mrsi_segment_data['metab']),
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

    # fslpy call
    wrapper_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        _run_fsl_mrs_proc_wrapper(preproc_data, wrapper_out)
    assert (wrapper_out / 'metab.nii.gz').exists()

    # fslpy call with objects
    object_out.mkdir()
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        _run_fsl_mrs_proc_wrapper(preproc_data, object_out, use_objects=True)
    assert (object_out / 'metab.nii.gz').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=False)
    assert compare_folders(cli_out, object_out,  subdir=False)


@requires_cli('fsl_dynmrs')
def test_fsl_dynmrs(tmp_path):
    cli_out     = tmp_path / 'dyn_cli'
    wrapper_out = tmp_path / 'dyn_wrapper'
    object_out  = tmp_path / 'dyn_object'

    data_str, basis_str, tv_str = _create_fsl_dynmrs_data(tmp_path)
    model_str = testsPath / 'testdata/dynamic/simple_linear_model.py'

    subprocess.check_call([
        'fsl_dynmrs',
        '--data', data_str,
        '--basis', basis_str,
        '--dyn_config', model_str,
        '--time_variables', tv_str,
        '--baseline_order', '0',
        '--output', cli_out,
        '--report',
    ])
    assert cli_out.exists()
    assert (cli_out / 'dyn_cov.csv').exists()
    assert (cli_out / 'init_results.csv').exists()
    assert (cli_out / 'dyn_results.csv').exists()
    assert (cli_out / 'mapped_parameters.csv').exists()
    assert (cli_out / 'free_parameters.csv').exists()
    assert (cli_out / 'options.txt').exists()
    assert (cli_out / 'report.html').exists()

    # fslpy call
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_dynmrs(
            data=data_str,
            basis=basis_str,
            dyn_config=model_str,
            time_variables=tv_str,
            baseline_order='0',
            output=wrapper_out,
            report=True,
        )
    assert wrapper_out.exists()
    assert (wrapper_out / 'dyn_cov.csv').exists()
    assert (wrapper_out / 'init_results.csv').exists()
    assert (wrapper_out / 'dyn_results.csv').exists()
    assert (wrapper_out / 'mapped_parameters.csv').exists()
    assert (wrapper_out / 'free_parameters.csv').exists()
    assert (wrapper_out / 'options.txt').exists()
    assert (wrapper_out / 'report.html').exists()

    # fslpy call with objects
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fsl_dynmrs(
            data=NIFTI_MRS(data_str),
            basis=read_basis(basis_str),
            dyn_config=model_str,
            time_variables=tv_str,
            baseline_order='0',
            output=object_out,
            report=True,
        )
    assert object_out.exists()
    assert (object_out / 'dyn_cov.csv').exists()
    assert (object_out / 'init_results.csv').exists()
    assert (object_out / 'dyn_results.csv').exists()
    assert (object_out / 'mapped_parameters.csv').exists()
    assert (object_out / 'free_parameters.csv').exists()
    assert (object_out / 'options.txt').exists()
    assert (object_out / 'report.html').exists()

    assert compare_folders(cli_out, wrapper_out, subdir=True)
    assert compare_folders(cli_out, object_out, subdir=True)


@requires_cli('basis2spec')
def test_basis2spec(tmp_path):
    cli_out     = tmp_path / 'basis2spec_cli' / 'spectrum.nii.gz'
    wrapper_out = tmp_path / 'basis2spec_wrapper' / 'spectrum.nii.gz'
    object_out  = tmp_path / 'basis2spec_object' / 'spectrum.nii.gz'

    subprocess.check_call([
        'basis2spec',
        '--basis', basis2spec_data['basis'],
        '--reference', basis2spec_data['reference'],
        '--output', cli_out,
    ])

    # fslpy call
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        basis2spec(
            basis=basis2spec_data['basis'],
            reference=basis2spec_data['reference'],
            output=wrapper_out,
        )

    # fslpy call with objects
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        basis2spec(
            basis=read_basis(basis2spec_data['basis']),
            reference=NIFTI_MRS(basis2spec_data['reference']),
            output=object_out,
        )

    assert cli_out.exists()
    assert wrapper_out.exists()
    assert object_out.exists()
    assert compare_folders(cli_out.parent, wrapper_out.parent, subdir=False)
    assert compare_folders(cli_out.parent, object_out.parent,  subdir=False)


@requires_cli('fmrs_stats')
def test_fmrs_stats(tmp_path):
    cli_out     = tmp_path / 'stats_cli'
    wrapper_out = tmp_path / 'stats_wrapper'

    results_list = tmp_path / 'results_list'
    with open(results_list, 'w') as fp:
        fp.writelines([str(x) + '\n' for x in fmrs_stats_data['sim_results']])

    subprocess.check_call([
        'fmrs_stats',
        '--data', results_list,
        '--output', cli_out,
        '--fl-contrasts', fmrs_stats_data['fl_contrasts'],
        '--combine', 'NAA', 'NAAG',
        '--combine', 'Cr', 'PCr',
        '--combine', 'PCh', 'GPC',
        '--hl-design', fmrs_stats_data['design_gm'],
        '--hl-contrasts', fmrs_stats_data['con_gm'],
        '--hl-contrast-names', 'positive', 'negative',
        '--overwrite',
    ])
    assert (cli_out / 'group_stats.csv').is_file()
    assert (cli_out / '0_stim').is_dir()
    assert (cli_out / '1_stim').is_dir()
    assert (cli_out / '2_ctrl').is_dir()
    assert (cli_out / '3_ctrl').is_dir()
    assert (cli_out / '0_stim' / 'free_parameters.csv').is_file()

    # fslpy call
    with patch('fsl.utils.run.FSL_PREFIX', fsl_bin):
        fmrs_stats(
            data=results_list,
            output=wrapper_out,
            fl_contrasts=fmrs_stats_data['fl_contrasts'],
            combine=[['NAA', 'NAAG'], ['Cr', 'PCr'], ['PCh', 'GPC']],
            hl_design=fmrs_stats_data['design_gm'],
            hl_contrasts=fmrs_stats_data['con_gm'],
            hl_contrast_names=['positive', 'negative'],
            overwrite=True,
        )
    assert (wrapper_out / 'group_stats.csv').is_file()
    assert (wrapper_out / '0_stim').is_dir()
    assert (wrapper_out / '1_stim').is_dir()
    assert (wrapper_out / '2_ctrl').is_dir()
    assert (wrapper_out / '3_ctrl').is_dir()
    assert (wrapper_out / '0_stim' / 'free_parameters.csv').is_file()

    assert compare_folders(cli_out, wrapper_out, subdir=True)
