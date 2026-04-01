'''FSL-MRS test script

Test the basis2spec script

Copyright Will Clarke, University of Oxford, 2026'''

from math import pi
from pathlib import Path
import sys

import numpy as np
import pytest

from fsl_mrs.scripts import basis2spec
from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.synthetic import syntheticFromBasisFile

testsPath = Path(__file__).parent
basis_path = testsPath / 'testdata/fsl_mrs/steam_basis_no_mm'
reference_path = testsPath / 'testdata/fsl_mrs/metab.nii.gz'


def _run_basis2spec(monkeypatch, output_path, extra_args):
    cli_args = ['basis2spec',
                '--basis', str(basis_path),
                '--reference', str(reference_path),
                '--output', str(output_path)]
    cli_args.extend(extra_args)

    monkeypatch.setattr(sys, 'argv', cli_args)
    basis2spec.main()

    return mrs_io.read_FID(output_path)


def _expected_output(linewidth, ignore):
    ref = mrs_io.read_FID(reference_path)
    expected, _ = syntheticFromBasisFile(
        basisFile=basis_path,
        ignore=ignore,
        broadening=(linewidth * pi, 0),
        noisecovariance=[[0.0]],
        points=ref.shape[3],
        bandwidth=ref.bandwidth,
        nifti_output=True)
    return expected


@pytest.mark.parametrize(
    'extra_args, linewidth, ignore',
    [
        ([], 0.0, None),
        (['--linewidth', '10.0'], 10.0, None),
        (['--ignore', 'Gln'], 0.0, ['Gln']),
        (['--linewidth', '5.0', '--ignore', 'Gln', 'NAA'], 5.0, ['Gln', 'NAA']),
    ],
    ids=['default', 'linewidth', 'ignore', 'linewidth_and_ignore'])
def test_basis2spec(monkeypatch, tmp_path, extra_args, linewidth, ignore):
    output = tmp_path / 'basis2spec' / 'spectrum.nii.gz'

    result = _run_basis2spec(monkeypatch, output, extra_args)
    expected = _expected_output(linewidth, ignore)
    reference = mrs_io.read_FID(reference_path)

    assert output.is_file()
    assert result.shape == expected.shape
    assert result.shape[3] == reference.shape[3]
    assert result.bandwidth == reference.bandwidth
    assert np.allclose(result[:], expected[:])
