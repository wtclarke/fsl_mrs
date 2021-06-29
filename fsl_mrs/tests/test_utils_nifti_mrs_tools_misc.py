"""Test the miscellaneous functions for NIFTI-MRS tools

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
Copyright (C) 2021 University of Oxford
"""

import fsl_mrs.utils.nifti_mrs_tools.misc as misc


def test_short_to_long():
    dict_repr = misc.dim_n_header_short_to_long({'start': 0.0, 'increment': 0.1}, 3)
    assert dict_repr == [0.0, 0.1, 0.2]

    dict_repr = misc.dim_n_header_short_to_long([0.0, 0.1, 0.2], 3)
    assert dict_repr == [0.0, 0.1, 0.2]

    dict_repr = misc.dim_n_header_short_to_long({'Value': [0.0, 0.1, 0.2], 'description': 'test'}, 3)
    assert dict_repr == {'Value': [0.0, 0.1, 0.2], 'description': 'test'}

    dict_repr = misc.dim_n_header_short_to_long({'Value': {'start': 0.0, 'increment': 0.1}, 'description': 'test'}, 3)
    assert dict_repr == {'Value': [0.0, 0.1, 0.2], 'description': 'test'}


def test_long_to_short():
    dict_repr = misc.dim_n_header_long_to_short([0.0, 0.1, 0.2])
    assert dict_repr == {'start': 0.0, 'increment': 0.1}

    dict_repr = misc.dim_n_header_long_to_short({'start': 0.0, 'increment': 0.1})
    assert dict_repr == {'start': 0.0, 'increment': 0.1}

    dict_repr = misc.dim_n_header_long_to_short({'Value': [0.0, 0.1, 0.2], 'description': 'test'})
    assert dict_repr == {'Value': {'start': 0.0, 'increment': 0.1}, 'description': 'test'}

    dict_repr = misc.dim_n_header_long_to_short({'Value': {'start': 0.0, 'increment': 0.1}, 'description': 'test'})
    assert dict_repr == {'Value': {'start': 0.0, 'increment': 0.1}, 'description': 'test'}


def test_dict_to_list():
    list_repr = misc._dict_to_list({'start': 0.0, 'increment': 0.1}, 3)
    assert list_repr == [0.0, 0.1, 0.2]

    list_repr = misc._dict_to_list({'start': 0.0, 'increment': 0.1}, 1)
    assert list_repr == [0.0, ]

    list_repr = misc._dict_to_list({'start': 1, 'increment': 1}, 3)
    assert list_repr == [1, 2, 3]

    list_repr = misc._dict_to_list({'start': 1, 'increment': -1}, 3)
    assert list_repr == [1, 0, -1]

    list_repr = misc._dict_to_list([1, 0, -1], 3)
    assert list_repr == [1, 0, -1]


def test_list_to_dict():
    dict_repr = misc._list_to_dict([0.0, 0.1, 0.2])
    assert dict_repr == {'start': 0.0, 'increment': 0.1}

    dict_repr = misc._list_to_dict([1, 2, 3])
    assert dict_repr == {'start': 1, 'increment': 1}

    dict_repr = misc._list_to_dict([1, 0, -1])
    assert dict_repr == {'start': 1, 'increment': -1}

    dict_repr = misc._list_to_dict([1, 2, 4])
    assert dict_repr == [1, 2, 4]

    dict_repr = misc._list_to_dict(['ON', 'OFF'])
    assert dict_repr == ['ON', 'OFF']
