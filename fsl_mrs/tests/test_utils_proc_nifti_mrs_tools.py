"""Test the split, merge and reorder tools for NIFTI-MRS

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
Copyright (C) 2021 University of Oxford
"""

from pathlib import Path
import pytest

import numpy as np

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.preproc import nifti_mrs_tools as nmrs_tools

testsPath = Path(__file__).parent
test_data_split = testsPath / 'testdata' / 'fsl_mrs_preproc' / 'metab_raw.nii.gz'
test_data_merge = testsPath / 'testdata' / 'fsl_mrs_preproc' / 'wref_raw.nii.gz'


def test_split():
    """Test the split functionality
    """
    nmrs = mrs_io.read_FID(test_data_split)

    # Error testing
    # Wrong dim tag
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.split(nmrs, 'DIM_EDIT', 1)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "DIM_EDIT not found as dimension tag."\
                                     " This data contains ['DIM_COIL', 'DIM_DYN', None]."

    # Wrong dim index (no dim in this data)
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.split(nmrs, 6, 1)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Dimension must be one of 4, 5, or 6 (or DIM_TAG string)."\
                                     " This data has 6 dimensions,"\
                                     " i.e. a maximum dimension value of 5."

    # Wrong dim index (too low)
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.split(nmrs, 3, 1)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Dimension must be one of 4, 5, or 6 (or DIM_TAG string)."\
                                     " This data has 6 dimensions,"\
                                     " i.e. a maximum dimension value of 5."

    # Wrong dim index type
    with pytest.raises(TypeError) as exc_info:
        nmrs_tools.split(nmrs, [3, ], 1)

    assert exc_info.type is TypeError
    assert exc_info.value.args[0] == "Dimension must be an int (4, 5, or 6) or string (DIM_TAG string)."

    # Single index - out of range low
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.split(nmrs, 'DIM_DYN', -1)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "index_or_indicies must be between 0 and N-1,"\
                                     " where N is the size of the specified dimension (64)."

    # Single index - out of range high
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.split(nmrs, 'DIM_DYN', 64)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "index_or_indicies must be between 0 and N-1,"\
                                     " where N is the size of the specified dimension (64)."

    # List of indicies - out of range low
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.split(nmrs, 'DIM_DYN', [-1, 0, 1])

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "index_or_indicies must have elements between 0 and N,"\
                                     " where N is the size of the specified dimension (64)."

    # List of indicies - out of range high
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.split(nmrs, 'DIM_DYN', [0, 65])

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "index_or_indicies must have elements between 0 and N,"\
                                     " where N is the size of the specified dimension (64)."

    # List of indicies - wrong type
    with pytest.raises(TypeError) as exc_info:
        nmrs_tools.split(nmrs, 'DIM_DYN', '1')

    assert exc_info.type is TypeError
    assert exc_info.value.args[0] == "index_or_indicies must be single index or list of indicies"

    # Functionality testing

    out_1, out_2 = nmrs_tools.split(nmrs, 'DIM_DYN', 32)
    assert out_1.data.shape == (1, 1, 1, 4096, 32, 32)
    assert out_2.data.shape == (1, 1, 1, 4096, 32, 32)
    assert np.allclose(out_1.data, nmrs.data[:, :, :, :, :, 0:32])
    assert np.allclose(out_2.data, nmrs.data[:, :, :, :, :, 32:])
    assert out_1.hdr_ext == nmrs.hdr_ext
    assert out_1.hdr_ext == nmrs.hdr_ext
    assert np.allclose(out_1.getAffine('voxel', 'world'), nmrs.getAffine('voxel', 'world'))
    assert np.allclose(out_2.getAffine('voxel', 'world'), nmrs.getAffine('voxel', 'world'))

    out_1, out_2 = nmrs_tools.split(nmrs, 'DIM_DYN', [0, 32, 63])
    assert out_1.data.shape == (1, 1, 1, 4096, 32, 61)
    assert out_2.data.shape == (1, 1, 1, 4096, 32, 3)
    test_list = np.arange(0, 64)
    test_list = np.delete(test_list, [0, 32, 63])
    assert np.allclose(out_1.data, nmrs.data[:, :, :, :, :, test_list])
    assert np.allclose(out_2.data, nmrs.data[:, :, :, :, :, [0, 32, 63]])
