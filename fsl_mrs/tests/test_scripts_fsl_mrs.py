# Test the main svs fitting script

# Imports
import subprocess
import pytest
import os.path as op

# Files
testsPath = op.dirname(__file__)
data = {'metab':op.join(testsPath,'testdata/fsl_mrs/metab.nii'),
        'water':op.join(testsPath,'testdata/fsl_mrs/water.nii'),
        'basis':op.join(testsPath,'testdata/fsl_mrs/steam_basis')}

def test_fsl_mrs(tmp_path):

    retcode = subprocess.check_call(['fsl_mrs',
                                    '--data', data['metab'], 
                                    '--basis', data['basis'], 
                                    '--output', tmp_path, 
                                    '--h2o', data['water'], 
                                    '--TE', '11', 
                                    '--tissue_frac', '0.45', '0.45', '0.1',
                                    '--overwrite',
                                    '--combine', 'Cr', 'PCr',
                                    '--report'])


    assert op.exists(op.join(tmp_path,'report.html'))
    assert op.exists(op.join(tmp_path,'all_parameters.csv'))
    assert op.exists(op.join(tmp_path,'qc.csv'))
    assert op.exists(op.join(tmp_path,'results_table.csv'))
    assert op.exists(op.join(tmp_path,'options.txt'))