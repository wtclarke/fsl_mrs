""" Utility functions for NIfTI-MRS utilities

    Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
    Copyright (C) 2021 University of Oxford
"""
import json

import numpy as np
from nibabel.nifti1 import Nifti1Extension


def modify_hdr_ext(new_hdr_ext, hdr):
    """Generate a new NIfTI header with a modified header extension.
    New header is a copy of the one passed

    :param new_hdr_ext: Modified header extension
    :type new_hdr_ext: dict
    :param hdr: NIfTI header
    :type hdr: nibabel.nifti2.Nifti2Header
    :return: Copied header with modified hdr extension
    :rtype: nibabel.nifti2.Nifti2Header
    """
    modded_hdr = hdr.copy()
    json_s = json.dumps(new_hdr_ext)
    extension = Nifti1Extension(44, json_s.encode('UTF-8'))
    modded_hdr.extensions.clear()
    modded_hdr.extensions.append(extension)

    return modded_hdr


def check_type(in_format):
    """Return type of header: long (list) or short (dict)

    :param in_format: Value of header key
    :type in_format: list or dict
    :return: 'long' or 'short'
    :rtype: str
    """
    if isinstance(in_format, list):
        return 'long'
    elif isinstance(in_format, dict)\
            and 'start' in in_format:
        return 'short'
    elif isinstance(in_format, dict)\
            and 'Value' in in_format:
        return check_type(in_format['Value'])


def dim_n_header_short_to_long(in_format, elements):
    if isinstance(in_format, list):
        return in_format
    elif isinstance(in_format, dict)\
            and 'start' in in_format:
        return _dict_to_list(in_format, elements)
    elif isinstance(in_format, dict)\
            and 'Value' in in_format:
        out = in_format.copy()
        out['Value'] = _dict_to_list(out['Value'], elements)
        return out


def dim_n_header_long_to_short(in_format):
    if isinstance(in_format, list):
        return _list_to_dict(in_format)
    elif isinstance(in_format, dict)\
            and 'start' in in_format:
        return in_format
    elif isinstance(in_format, dict)\
            and 'Value' in in_format:
        out = in_format.copy()
        out['Value'] = _list_to_dict(in_format['Value'])
        return out


def _dict_to_list(dict_desc, elements):
    """ Convert a short dict format to list format

    :param dict_desc: dict with start and increment fields
    :type dict_desc: dict
    :param elements: Number of elements in dimension
    :type elements: int
    :return: Converted list representation
    :rtype: list
    """
    if isinstance(dict_desc, dict):
        stop = dict_desc['start'] + dict_desc['increment'] * (elements - 1)
        return np.linspace(dict_desc['start'], stop, elements).tolist()
    else:
        return dict_desc


def _list_to_dict(list_desc):
    """If possible (i.e. vector is monotonic and equally spaced),
    convert list to dict (start + increment) representation.

    If conversion is not possible return the passed list.

    :param list_desc: List of header indicies
    :type list_desc: list
    :return: Dict representation or unmodified list
    :rtype: Dict or list
    """
    list_desc = np.asarray(list_desc)
    if np.issubdtype(list_desc.dtype, np.number)\
            and np.unique(np.diff(list_desc)).size == 1:
        return {'start': list_desc[0], 'increment': np.diff(list_desc)[0]}
    else:
        return list_desc.tolist()
