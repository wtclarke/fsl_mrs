# Test the generation of spectra from fsl_mrs output

# Imports
import subprocess
import os.path as op

# Files
testsPath = op.dirname(__file__)
data = {'metab': op.join(testsPath, 'testdata/fsl_mrs/metab.nii.gz'),
        'water': op.join(testsPath, 'testdata/fsl_mrs/wref.nii.gz'),
        'basis': op.join(testsPath, 'testdata/fsl_mrs/steam_basis'),
        'seg': op.join(testsPath, 'testdata/fsl_mrs/segmentation.json')}


def test_results_to_spectrum(tmp_path):

    subprocess.check_call(['fsl_mrs',
                           '--data', data['metab'],
                           '--basis', data['basis'],
                           '--output', tmp_path,
                           '--h2o', data['water'],
                           '--TE', '11',
                           '--metab_groups', 'Mac',
                           '--tissue_frac', '0.45', '0.45', '0.1',
                           '--overwrite',
                           '--combine', 'Cr', 'PCr'])

    subprocess.check_call(['results_to_spectrum',
                           str(tmp_path),
                           '--output', str(tmp_path),
                           '--filename', 'test',
                           '--export_baseline',
                           '--export_no_baseline',
                           '--export_separate'])

    assert op.exists(op.join(tmp_path, 'test.nii.gz'))
    assert op.exists(op.join(tmp_path, 'test_baseline.nii.gz'))
    assert op.exists(op.join(tmp_path, 'test_no_baseline.nii.gz'))
    assert op.exists(op.join(tmp_path, 'test_NAA.nii.gz'))
    assert op.exists(op.join(tmp_path, 'test_Cr.nii.gz'))
