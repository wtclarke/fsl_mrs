"""Test the split, merge and reorder tools for NIFTI-MRS

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
Copyright (C) 2021 University of Oxford
"""

from pathlib import Path
import pytest

import numpy as np

from fsl_mrs.utils import mrs_io
from fsl_mrs.utils import nifti_mrs_tools as nmrs_tools
from fsl_mrs.utils.nifti_mrs_tools.split_merge import NIfTI_MRSIncompatible
from fsl_mrs.core.nifti_mrs import gen_new_nifti_mrs

testsPath = Path(__file__).parent
test_data_split = testsPath / 'testdata' / 'fsl_mrs_preproc' / 'metab_raw.nii.gz'
test_data_merge_1 = testsPath / 'testdata' / 'fsl_mrs_preproc' / 'wref_raw.nii.gz'
test_data_merge_2 = testsPath / 'testdata' / 'fsl_mrs_preproc' / 'quant_raw.nii.gz'
test_data_other = testsPath / 'testdata' / 'fsl_mrs_preproc' / 'ecc.nii.gz'


def test_split_dim_header():
    """Test the ability to split the dim_N_header fields"""
    hdr_in = {'dim_5': 'DIM_DYN',
              'dim_5_info': 'averages',
              'dim_5_header': {'p1': [1, 2, 3, 4],
                               'p2': [0.1, 0.2, 0.3, 0.4]},
              'dim_6': 'DIM_EDIT',
              'dim_6_info': 'edit',
              'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                               'p2': [0.1, 0.2, 0.3, 0.4]},
              'dim_7': 'DIM_USER_0',
              'dim_7_info': 'other',
              'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                               'p2': [0.1, 0.2, 0.3, 0.4]}}

    # Headers occuring as a list.
    hdr1, hdr2 = nmrs_tools.split_merge._split_dim_header(hdr_in, 5, 4, 1)
    assert hdr1 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 2],
                                     'p2': [0.1, 0.2]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}
    assert hdr2 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [3, 4],
                                     'p2': [0.3, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}

    hdr1, hdr2 = nmrs_tools.split_merge._split_dim_header(hdr_in, 5, 4, [1, 3])
    assert hdr1 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 3],
                                     'p2': [0.1, 0.3]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}
    assert hdr2 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [2, 4],
                                     'p2': [0.2, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}

    # Headers as a dict
    hdr1, hdr2 = nmrs_tools.split_merge._split_dim_header(hdr_in, 6, 4, 1)
    assert hdr1 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 2, 3, 4],
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                     'p2': [0.1, 0.2]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}
    assert hdr2 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 2, 3, 4],
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 3, 'increment': 1},
                                     'p2': [0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}

    hdr1, hdr2 = nmrs_tools.split_merge._split_dim_header(hdr_in, 6, 4, [1, ])
    assert hdr1 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 2, 3, 4],
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': [1, 3, 4],
                                     'p2': [0.1, 0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}
    assert hdr2 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 2, 3, 4],
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': [2, ],
                                     'p2': [0.2, ]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2, 0.3, 0.4]}}

    # User defined structures
    hdr1, hdr2 = nmrs_tools.split_merge._split_dim_header(hdr_in, 7, 4, 1)
    assert hdr1 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 2, 3, 4],
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.1, 0.2]}}
    assert hdr2 == {'dim_5': 'DIM_DYN',
                    'dim_5_info': 'averages',
                    'dim_5_header': {'p1': [1, 2, 3, 4],
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_6': 'DIM_EDIT',
                    'dim_6_info': 'edit',
                    'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                     'p2': [0.1, 0.2, 0.3, 0.4]},
                    'dim_7': 'DIM_USER_0',
                    'dim_7_info': 'other',
                    'dim_7_header': {'p1': {'Value': {'start': 3, 'increment': 1}, 'description': 'user'},
                                     'p2': [0.3, 0.4]}}


def test_merge_dim_header():
    """Test the ability to merge the dim_N_header fields"""
    hdr_in_1 = {'dim_5': 'DIM_DYN',
                'dim_5_info': 'averages',
                'dim_5_header': {'p1': [1, 2, 3, 4],
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_6': 'DIM_EDIT',
                'dim_6_info': 'edit',
                'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_7': 'DIM_USER_0',
                'dim_7_info': 'other',
                'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                 'p2': [0.1, 0.2, 0.3, 0.4]}}
    hdr_in_2 = {'dim_5': 'DIM_DYN',
                'dim_5_info': 'averages',
                'dim_5_header': {'p1': [1, 2, 3, 4],
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_6': 'DIM_EDIT',
                'dim_6_info': 'edit',
                'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_7': 'DIM_USER_0',
                'dim_7_info': 'other',
                'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                 'p2': [0.1, 0.2, 0.3, 0.4]}}

    hdr_out = nmrs_tools.split_merge._merge_dim_header(hdr_in_1, hdr_in_2, 5, 4, 4)
    assert hdr_out == {'dim_5': 'DIM_DYN',
                       'dim_5_info': 'averages',
                       'dim_5_header': {'p1': [1, 2, 3, 4, 1, 2, 3, 4],
                                        'p2': [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]},
                       'dim_6': 'DIM_EDIT',
                       'dim_6_info': 'edit',
                       'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                        'p2': [0.1, 0.2, 0.3, 0.4]},
                       'dim_7': 'DIM_USER_0',
                       'dim_7_info': 'other',
                       'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                        'p2': [0.1, 0.2, 0.3, 0.4]}}

    hdr_in_2 = {'dim_5': 'DIM_DYN',
                'dim_5_info': 'averages',
                'dim_5_header': {'p1': [1, 2, 3, 4],
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_6': 'DIM_EDIT',
                'dim_6_info': 'edit',
                'dim_6_header': {'p1': {'start': 5, 'increment': 1},
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_7': 'DIM_USER_0',
                'dim_7_info': 'other',
                'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                 'p2': [0.1, 0.2, 0.3, 0.4]}}
    hdr_out = nmrs_tools.split_merge._merge_dim_header(hdr_in_1, hdr_in_2, 6, 4, 4)
    assert hdr_out == {'dim_5': 'DIM_DYN',
                       'dim_5_info': 'averages',
                       'dim_5_header': {'p1': [1, 2, 3, 4],
                                        'p2': [0.1, 0.2, 0.3, 0.4]},
                       'dim_6': 'DIM_EDIT',
                       'dim_6_info': 'edit',
                       'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                        'p2': [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]},
                       'dim_7': 'DIM_USER_0',
                       'dim_7_info': 'other',
                       'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                        'p2': [0.1, 0.2, 0.3, 0.4]}}

    hdr_out = nmrs_tools.split_merge._merge_dim_header(hdr_in_2, hdr_in_1, 6, 4, 4)
    assert hdr_out == {'dim_5': 'DIM_DYN',
                       'dim_5_info': 'averages',
                       'dim_5_header': {'p1': [1, 2, 3, 4],
                                        'p2': [0.1, 0.2, 0.3, 0.4]},
                       'dim_6': 'DIM_EDIT',
                       'dim_6_info': 'edit',
                       'dim_6_header': {'p1': [5, 6, 7, 8, 1, 2, 3, 4],
                                        'p2': [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]},
                       'dim_7': 'DIM_USER_0',
                       'dim_7_info': 'other',
                       'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                        'p2': [0.1, 0.2, 0.3, 0.4]}}

    hdr_in_2 = {'dim_5': 'DIM_DYN',
                'dim_5_info': 'averages',
                'dim_5_header': {'p1': [1, 2, 3, 4],
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_6': 'DIM_EDIT',
                'dim_6_info': 'edit',
                'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                 'p2': [0.1, 0.2, 0.3, 0.4]},
                'dim_7': 'DIM_USER_0',
                'dim_7_info': 'other',
                'dim_7_header': {'p1': {'Value': {'start': 5, 'increment': 1}, 'description': 'user'},
                                 'p2': [0.1, 0.2, 0.3, 0.4]}}
    hdr_out = nmrs_tools.split_merge._merge_dim_header(hdr_in_1, hdr_in_2, 7, 4, 4)
    assert hdr_out == {'dim_5': 'DIM_DYN',
                       'dim_5_info': 'averages',
                       'dim_5_header': {'p1': [1, 2, 3, 4],
                                        'p2': [0.1, 0.2, 0.3, 0.4]},
                       'dim_6': 'DIM_EDIT',
                       'dim_6_info': 'edit',
                       'dim_6_header': {'p1': {'start': 1, 'increment': 1},
                                        'p2': [0.1, 0.2, 0.3, 0.4]},
                       'dim_7': 'DIM_USER_0',
                       'dim_7_info': 'other',
                       'dim_7_header': {'p1': {'Value': {'start': 1, 'increment': 1}, 'description': 'user'},
                                        'p2': [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]}}

    with pytest.raises(NIfTI_MRSIncompatible) as exc_info:
        hdr_out = nmrs_tools.split_merge._merge_dim_header(hdr_in_1, hdr_in_2, 5, 4, 4)
    assert exc_info.type is NIfTI_MRSIncompatible
    assert exc_info.value.args[0] == "Both files must have matching dimension headers apart from the one being merged."\
                                     " dim_7_header does not match."


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

    out_1, out_2 = nmrs_tools.split(nmrs, 'DIM_DYN', 31)
    assert out_1[:].shape == (1, 1, 1, 4096, 32, 32)
    assert out_2[:].shape == (1, 1, 1, 4096, 32, 32)
    assert np.allclose(out_1[:], nmrs[:, :, :, :, :, 0:32])
    assert np.allclose(out_2[:], nmrs[:, :, :, :, :, 32:])
    assert out_1.hdr_ext == nmrs.hdr_ext
    assert out_1.hdr_ext == nmrs.hdr_ext
    assert np.allclose(out_1.getAffine('voxel', 'world'), nmrs.getAffine('voxel', 'world'))
    assert np.allclose(out_2.getAffine('voxel', 'world'), nmrs.getAffine('voxel', 'world'))

    out_1, out_2 = nmrs_tools.split(nmrs, 'DIM_DYN', [0, 32, 63])
    assert out_1[:].shape == (1, 1, 1, 4096, 32, 61)
    assert out_2[:].shape == (1, 1, 1, 4096, 32, 3)
    test_list = np.arange(0, 64)
    test_list = np.delete(test_list, [0, 32, 63])
    assert np.allclose(out_1[:], nmrs[:][:, :, :, :, :, test_list])
    assert np.allclose(out_2[:], nmrs[:][:, :, :, :, :, [0, 32, 63]])

    # Split some synthetic data with header information
    nhdr_1 = gen_new_nifti_mrs(np.ones((1, 1, 1, 10, 4), dtype=complex),
                               1 / 1000,
                               100,
                               '1H',
                               dim_tags=['DIM_DYN', None, None])

    nhdr_1.set_dim_header('DIM_DYN', {'RepetitionTime': [1, 2, 3, 4]})

    out_1, out_2 = nmrs_tools.split(nhdr_1, 'DIM_DYN', 1)
    assert out_1.shape == (1, 1, 1, 10, 2)
    assert out_1.hdr_ext['dim_5'] == 'DIM_DYN'
    assert out_1.hdr_ext['dim_5_header'] == {'RepetitionTime': [1, 2]}
    assert out_2.hdr_ext['dim_5_header'] == {'RepetitionTime': [3, 4]}


def test_merge():
    """Test the merge functionality
    """
    nmrs_1 = mrs_io.read_FID(test_data_merge_1)
    nmrs_2 = mrs_io.read_FID(test_data_merge_2)

    nmrs_bad_shape, _ = nmrs_tools.split(nmrs_2, 'DIM_COIL', 3)
    nmrs_no_tag = mrs_io.read_FID(test_data_other)

    # Error testing
    # Wrong dim tag
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.merge((nmrs_1, nmrs_2), 'DIM_EDIT')

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "DIM_EDIT not found as dimension tag."\
                                     " This data contains ['DIM_COIL', 'DIM_DYN', None]."

    # Wrong dim index (no dim in this data)
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.merge((nmrs_1, nmrs_2), 6)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Dimension must be one of 4, 5, or 6 (or DIM_TAG string)."\
                                     " This data has 6 dimensions,"\
                                     " i.e. a maximum dimension value of 5."

    # Wrong dim index (too low)
    with pytest.raises(ValueError) as exc_info:
        nmrs_tools.merge((nmrs_1, nmrs_2), 3)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Dimension must be one of 4, 5, or 6 (or DIM_TAG string)."\
                                     " This data has 6 dimensions,"\
                                     " i.e. a maximum dimension value of 5."

    # Wrong dim index type
    with pytest.raises(TypeError) as exc_info:
        nmrs_tools.merge((nmrs_1, nmrs_2), [3, ])

    assert exc_info.type is TypeError
    assert exc_info.value.args[0] == "Dimension must be an int (4, 5, or 6) or string (DIM_TAG string)."

    # Incompatible shapes
    with pytest.raises(NIfTI_MRSIncompatible) as exc_info:
        nmrs_tools.merge((nmrs_1, nmrs_bad_shape), 'DIM_DYN')

    assert exc_info.type is NIfTI_MRSIncompatible
    assert exc_info.value.args[0] == "The shape of all concatentated objects must match. "\
                                     "The shape ((1, 1, 1, 4096, 4, 2)) of the 1 object does "\
                                     "not match that of the first ((1, 1, 1, 4096, 32, 2))."

    # Incompatible tags
    with pytest.raises(NIfTI_MRSIncompatible) as exc_info:
        nmrs_tools.merge((nmrs_1, nmrs_no_tag), 'DIM_DYN')

    assert exc_info.type is NIfTI_MRSIncompatible
    assert exc_info.value.args[0] == "The tags of all concatentated objects must match. "\
                                     "The tags (['DIM_COIL', None, None]) of the 1 object does "\
                                     "not match that of the first (['DIM_COIL', 'DIM_DYN', None])."

    # Functionality testing
    out = nmrs_tools.merge((nmrs_1, nmrs_2), 'DIM_DYN')
    assert out[:].shape == (1, 1, 1, 4096, 32, 4)
    assert np.allclose(out[:][:, :, :, :, :, 0:2], nmrs_1[:])
    assert np.allclose(out[:][:, :, :, :, :, 2:], nmrs_2[:])
    assert out.hdr_ext == nmrs_1.hdr_ext
    assert np.allclose(out.getAffine('voxel', 'world'), nmrs_1.getAffine('voxel', 'world'))

    # Merge along squeezed singleton
    nmrs_1_e = nmrs_tools.reorder(nmrs_1, ['DIM_COIL', 'DIM_DYN', 'DIM_EDIT'])
    nmrs_2_e = nmrs_tools.reorder(nmrs_2, ['DIM_COIL', 'DIM_DYN', 'DIM_EDIT'])
    out = nmrs_tools.merge((nmrs_1_e, nmrs_2_e), 'DIM_EDIT')
    assert out[:].shape == (1, 1, 1, 4096, 32, 2, 2)
    assert out.hdr_ext['dim_7'] == 'DIM_EDIT'

    # Merge some synthetic data with header information
    nhdr_1 = gen_new_nifti_mrs(np.ones((1, 1, 1, 10, 4), dtype=complex),
                               1 / 1000,
                               100,
                               '1H',
                               dim_tags=['DIM_DYN', None, None])
    nhdr_2 = nhdr_1.copy()

    nhdr_1.set_dim_header('DIM_DYN', {'RepetitionTime': [1, 2, 3, 4]})
    nhdr_2.set_dim_header('DIM_DYN', {'RepetitionTime': [1, 2, 3, 4]})

    out = nmrs_tools.merge((nhdr_1, nhdr_2, nhdr_2), 'DIM_DYN')
    assert out[:].shape == (1, 1, 1, 10, 12)
    assert out.hdr_ext['dim_5'] == 'DIM_DYN'
    assert out.hdr_ext['dim_5_header'] == {'RepetitionTime': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]}

    nhdr_1.set_dim_header('DIM_DYN', {'RepetitionTime': {'start': 1, 'increment': 1}})
    nhdr_2.set_dim_header('DIM_DYN', {'RepetitionTime': [5, 6, 7, 8]})

    out = nmrs_tools.merge((nhdr_1, nhdr_2), 'DIM_DYN')
    assert out[:].shape == (1, 1, 1, 10, 8)
    assert out.hdr_ext['dim_5'] == 'DIM_DYN'
    assert out.hdr_ext['dim_5_header'] == {'RepetitionTime': {'start': 1, 'increment': 1}}

    # Merge along squeezed singleton with header
    nhdr_1 = gen_new_nifti_mrs(np.ones((1, 1, 1, 10, 4), dtype=complex),
                               1 / 1000,
                               100,
                               '1H',
                               dim_tags=['DIM_DYN', None, None])
    nhdr_2 = nhdr_1.copy()
    nhdr_1_e = nmrs_tools.reorder(nhdr_1, ['DIM_DYN', 'DIM_EDIT', None])
    nhdr_2_e = nmrs_tools.reorder(nhdr_2, ['DIM_DYN', 'DIM_EDIT', None])

    nhdr_1_e.set_dim_header('DIM_DYN', {'RepetitionTime': {'start': 1, 'increment': 1}})
    nhdr_2_e.set_dim_header('DIM_DYN', {'RepetitionTime': {'start': 1, 'increment': 1}})
    nhdr_1_e.set_dim_header('DIM_EDIT', {'OtherTime': [0.1, ]})
    nhdr_2_e.set_dim_header('DIM_EDIT', {'OtherTime': [0.2, ]})

    out = nmrs_tools.merge((nhdr_1_e, nhdr_2_e), 'DIM_EDIT')
    assert out[:].shape == (1, 1, 1, 10, 4, 2)
    assert out.hdr_ext['dim_6'] == 'DIM_EDIT'
    assert out.hdr_ext['dim_6_header'] == {'OtherTime': [0.1, 0.2, ]}


def test_reorder():
    """Test the reorder functionality
    """
    nmrs = mrs_io.read_FID(test_data_split)

    # Error testing
    # Miss existing tag
    with pytest.raises(NIfTI_MRSIncompatible) as exc_info:
        nmrs_tools.reorder(nmrs, ['DIM_COIL', 'DIM_EDIT'])

    assert exc_info.type is NIfTI_MRSIncompatible
    assert exc_info.value.args[0] == "The existing tag (DIM_DYN) does not appear"\
                                     " in the requested tag order (['DIM_COIL', 'DIM_EDIT'])."

    # Functionality testing
    # Swap order of dimensions
    out = nmrs_tools.reorder(nmrs, ['DIM_DYN', 'DIM_COIL'])
    assert out[:].shape == (1, 1, 1, 4096, 64, 32)
    assert np.allclose(np.swapaxes(nmrs[:], 4, 5), out[:])
    assert out.hdr_ext['dim_5'] == 'DIM_DYN'
    assert out.hdr_ext['dim_6'] == 'DIM_COIL'

    # # Add an additional singleton at end (not reported in shape)
    out = nmrs_tools.reorder(nmrs, ['DIM_COIL', 'DIM_DYN', 'DIM_EDIT'])
    assert out[:].shape == (1, 1, 1, 4096, 32, 64)
    assert out.hdr_ext['dim_5'] == 'DIM_COIL'
    assert out.hdr_ext['dim_6'] == 'DIM_DYN'
    assert out.hdr_ext['dim_7'] == 'DIM_EDIT'

    # Add an additional singleton at 5 (not reported in shape)
    out = nmrs_tools.reorder(nmrs, ['DIM_EDIT', 'DIM_COIL', 'DIM_DYN'])
    assert out[:].shape == (1, 1, 1, 4096, 1, 32, 64)
    assert out.hdr_ext['dim_5'] == 'DIM_EDIT'
    assert out.hdr_ext['dim_6'] == 'DIM_COIL'
    assert out.hdr_ext['dim_7'] == 'DIM_DYN'
