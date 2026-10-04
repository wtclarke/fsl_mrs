'''FSL-MRS test script

Test the signal scaling behaviour on concentration and CRLB.

Copyright Vasilis Karlaftis, University of Oxford, 2026'''

import os.path as op
import sys

import numpy as np
import pandas as pd
from fsl.data.image import Image

from fsl_mrs.scripts import fsl_mrs
from fsl_mrs.utils import mrs_io


testsPath = op.dirname(__file__)
data = {'metab': op.join(testsPath, 'testdata/fsl_mrs/metab.nii.gz'),
        'water': op.join(testsPath, 'testdata/fsl_mrs/wref.nii.gz'),
        'basis': op.join(testsPath, 'testdata/fsl_mrs/steam_basis'),
        'basis_default': op.join(testsPath, 'testdata/fsl_mrs/steam_basis_default_mm'),
        'seg': op.join(testsPath, 'testdata/fsl_mrs/segmentation.json')}


def _read_scaled_fid(filename, original_read_FID, scale):
    FID = original_read_FID(filename)
    if str(filename) == data['metab']:
        FID.image = Image(FID.image.data * scale, header=FID.image.header)
    return FID


def _assert_scaled_results(results, scales):
    reference = results[1.0]
    for scale in scales:
        current = results[scale]
        assert np.allclose(current['mMol/kg'],
                           reference['mMol/kg'] * scale,
                           rtol=1e-2, atol=1e-2), \
               f"calculated ratio≈{(current['mMol/kg']/reference['mMol/kg']).median():.2g}\n"
        assert np.allclose(current['mMol/kg CRLB'],
                           reference['mMol/kg CRLB'] * scale,
                           rtol=1e-2, atol=1e-2), \
               f"calculated ratio≈{(current['mMol/kg CRLB']/reference['mMol/kg CRLB']).median():.2g}\n"
        assert np.allclose(current['%CRLB'], reference['%CRLB'],
                           rtol=1e-2, atol=1e-2), \
               f"calculated ratio≈{(current['%CRLB']/reference['%CRLB']).median():.2g}\n"


def test_fsl_mrs_fid_scaling(tmp_path, monkeypatch):
    scales = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    original_read_FID = mrs_io.read_FID

    # run the test on each scale and collect the results
    results = {}
    for scale in scales:
        output = tmp_path / f'scale_{scale:g}'

        monkeypatch.setattr(
            mrs_io,
            'read_FID',
            lambda filename, scale=scale:
                _read_scaled_fid(filename, original_read_FID, scale),
        )

        monkeypatch.setattr(sys, 'argv', [
            'fsl_mrs',
            '--data', data['metab'],
            '--basis', data['basis'],
            '--output', str(output),
            '--h2o', data['water'],
            '--TE', '11',
            '--metab_groups', 'Mac',
            '--tissue_frac', data['seg'],
            '--overwrite',
            '--combine', 'Cr', 'PCr',
            '--report'])
        fsl_mrs.main()

        results[scale] = (pd.read_csv(output / 'summary.csv')
                            .set_index('Metab')[['mMol/kg', 'mMol/kg CRLB', '%CRLB']])

    # compare the results to assert that the scaling is correct
    _assert_scaled_results(results, scales)
