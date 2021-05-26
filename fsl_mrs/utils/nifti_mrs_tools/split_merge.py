"""Tools for merging, splitting and reordering the dimensions of NIfTI-MRS

    Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
    Copyright (C) 2021 University of Oxford
"""
import re
import json

import numpy as np
from nibabel.nifti1 import Nifti1Extension
from fsl_mrs.core.nifti_mrs import NIFTI_MRS, NIFTIMRS_DimDoesntExist
from fsl_mrs.utils.nifti_mrs_tools import misc


class NIfTI_MRSIncompatible(Exception):
    pass


def split(nmrs, dimension, index_or_indicies):
    """Splits, or extracts indices from, a specified dimension of a
    NIFTI_MRS object. Output is two NIFTI_MRS objects. Header information preserved.

    :param nmrs: Input nifti_mrs object to split
    :type nmrs: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param dimension: Dimension along which to split.
        Dimension tag or one of 4, 5, 6 (for 0-indexed 5th, 6th, and 7th)
    :type dimension: str or int
    :param index_or_indicies: Single integer index to split after,
        or list of interger indices to insert into second array.
        E.g. '0' will place the first index into the first output
        and 1 -> N in the second.
        '[1, 5, 10]' will place 1, 5 and 10 into the second output
        and all other will remain in the first.
    :type index_or_indicies: int or [int]
    :return: Two NIFTI_MRS object containing the split files
    :rtype: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    """

    if isinstance(dimension, str):
        try:
            dim_index = nmrs.dim_position(dimension)
        except NIFTIMRS_DimDoesntExist:
            raise ValueError(f'{dimension} not found as dimension tag. This data contains {nmrs.dim_tags}.')
    elif isinstance(dimension, int):
        if dimension > (nmrs.ndim - 1) or dimension < 4:
            raise ValueError('Dimension must be one of 4, 5, or 6 (or DIM_TAG string).'
                             f' This data has {nmrs.ndim} dimensions,'
                             f' i.e. a maximum dimension value of {nmrs.ndim-1}.')
        dim_index = dimension
    else:
        raise TypeError('Dimension must be an int (4, 5, or 6) or string (DIM_TAG string).')

    # Construct indexing
    if isinstance(index_or_indicies, int):
        if index_or_indicies < 0\
                or index_or_indicies >= nmrs.shape[dim_index]:
            raise ValueError('index_or_indicies must be between 0 and N-1,'
                             f' where N is the size of the specified dimension ({nmrs.shape[dim_index]}).')
        index = np.arange(index_or_indicies + 1, nmrs.shape[dim_index])

    elif isinstance(index_or_indicies, list):
        if not np.logical_and(np.asarray(index_or_indicies) >= 0,
                              np.asarray(index_or_indicies) <= nmrs.shape[dim_index]).all():
            raise ValueError('index_or_indicies must have elements between 0 and N,'
                             f' where N is the size of the specified dimension ({nmrs.shape[dim_index]}).')
        index = index_or_indicies

    else:
        raise TypeError('index_or_indicies must be single index or list of indicies')

    # Split header down
    split_hdr_ext_1, split_hdr_ext_2 = _split_dim_header(nmrs.hdr_ext,
                                                         dim_index + 1,
                                                         nmrs.shape[dim_index],
                                                         index_or_indicies)
    out_hdr_1 = misc.modify_hdr_ext(split_hdr_ext_1, nmrs.header)
    out_hdr_2 = misc.modify_hdr_ext(split_hdr_ext_2, nmrs.header)

    nmrs_1 = NIFTI_MRS(np.delete(nmrs.data, index, axis=dim_index), header=out_hdr_1)
    nmrs_2 = NIFTI_MRS(np.take(nmrs.data, index, axis=dim_index), header=out_hdr_2)

    return nmrs_1, nmrs_2


def _split_dim_header(hdr, dimension, dim_length, index):
    """Split dim_N_header keys in header extensions.

    :param hdr: Header extension to split
    :type hdr: dict
    :param dimension: Dimension (5, 6, or 7) to split along
    :type dimension: int
    :param dim_length: Length of dimension
    :type index: int
    :param index: Index to split after or indicies to extract
    :type index: int or list of ints
    :return: Split header eextension dicts
    :rtype: dict
    """
    hdr1 = hdr.copy()
    hdr2 = hdr.copy()

    def split_list(in_list):
        if isinstance(index, int):
            out_1 = in_list[:(index + 1)]
            out_2 = in_list[(index + 1):]
        elif isinstance(index, list):
            out_1 = np.delete(np.asarray(in_list), index).tolist()
            out_2 = np.take(np.asarray(in_list), index).tolist()
        return out_1, out_2

    def split_user_or_std(hdr_val):
        if isinstance(hdr_val, dict)\
                and 'value' in hdr_val:
            tmp_1, tmp_2 = split_list(hdr_val['value'])
            out_1 = hdr_val.copy()
            out_2 = hdr_val.copy()
            out_1.update({'value': tmp_1})
            out_2.update({'value': tmp_2})
            return out_1, out_2
        else:
            return split_list(hdr_val)

    def split_single(hdr_val):
        hdr_type = misc.check_type(hdr_val)
        long_fmt = misc.dim_n_header_short_to_long(hdr_val, dim_length)
        long_fmt_1, long_fmt_2 = split_user_or_std(long_fmt)
        if hdr_type == 'long':
            return long_fmt_1, long_fmt_2
        else:
            return misc.dim_n_header_long_to_short(long_fmt_1), misc.dim_n_header_long_to_short(long_fmt_2)

    key_str = f'dim_{dimension}_header'
    if key_str in hdr:
        new_h1 = {}
        new_h2 = {}
        for sub_key in hdr[key_str]:
            new_h1[sub_key], new_h2[sub_key] = split_single(hdr[key_str][sub_key])

        hdr1[key_str] = new_h1
        hdr2[key_str] = new_h2
    return hdr1, hdr2


def merge(array_of_nmrs, dimension):
    """Concatenate NIfTI-MRS objects along specified higher dimension

    :param array_of_nmrs: Array of NIFTI-MRS objects to concatenate
    :type array_of_nmrs: tuple or list of fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param dimension: Dimension along which to concatenate.
        Dimension tag or one of 4, 5, 6 (for 0-indexed 5th, 6th, and 7th).
    :type dimension: int or str
    :return: Concatenated NIFTI-MRS object
    :rtype: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    """
    if isinstance(dimension, str):
        try:
            dim_index = array_of_nmrs[0].dim_position(dimension)
        except NIFTIMRS_DimDoesntExist:
            raise ValueError(f'{dimension} not found as dimension tag. This data contains {array_of_nmrs[0].dim_tags}.')
    elif isinstance(dimension, int):
        if dimension > (array_of_nmrs[0].ndim - 1) or dimension < 4:
            raise ValueError('Dimension must be one of 4, 5, or 6 (or DIM_TAG string).'
                             f' This data has {array_of_nmrs[0].ndim} dimensions,'
                             f' i.e. a maximum dimension value of {array_of_nmrs[0].ndim-1}.')
        dim_index = dimension
    else:
        raise TypeError('Dimension must be an int (4, 5, or 6) or string (DIM_TAG string).')

    # Check shapes and tags are compatible.
    # If they are and enter the data into a tuple for concatenation
    def check_shape(to_compare):
        for dim in range(to_compare.ndim):
            # Do not compare on selected dimension
            if dim == dim_index:
                continue
            if to_compare.shape[dim] != array_of_nmrs[0].shape[dim]:
                return False
        return True

    def check_tag(to_compare):
        for tdx in range(3):
            if array_of_nmrs[0].dim_tags[tdx] != to_compare.dim_tags[tdx]:
                return False
        return True

    to_concat = []
    for idx, nmrs in enumerate(array_of_nmrs):
        # Check shape
        if not check_shape(nmrs):
            raise NIfTI_MRSIncompatible('The shape of all concatentated objects must match.'
                                        f' The shape ({nmrs.shape}) of the {idx} object does'
                                        f' not match that of the first ({array_of_nmrs[0].shape}).')
        # Check dim tags for compatibility
        if not check_tag(nmrs):
            raise NIfTI_MRSIncompatible('The tags of all concatentated objects must match.'
                                        f' The tags ({nmrs.dim_tags}) of the {idx} object does'
                                        f' not match that of the first ({array_of_nmrs[0].dim_tags}).')

        if nmrs.ndim == dim_index:
            # If a squeezed singleton on the end.
            to_concat.append(np.expand_dims(nmrs.data, -1))
        else:
            to_concat.append(nmrs.data)

        # Merge header extension
        if idx == 0:
            merged_hdr_ext = nmrs.hdr_ext
            merged_length = to_concat[-1].shape[dim_index]
        else:
            merged_hdr_ext = _merge_dim_header(merged_hdr_ext,
                                               nmrs.hdr_ext,
                                               dim_index + 1,
                                               merged_length,
                                               to_concat[-1].shape[dim_index])
            merged_length += to_concat[-1].shape[dim_index]

    out_hdr = misc.modify_hdr_ext(merged_hdr_ext, array_of_nmrs[0].header)

    return NIFTI_MRS(np.concatenate(to_concat, axis=dim_index), header=out_hdr)


def _merge_dim_header(hdr1, hdr2, dimension, dim_length1, dim_length2):
    """Merge dim_N_header keys in header extensions.
    Output header copies all other fields from hdr1

    :param hdr1: header extension from 1st file
    :type hdr1: dict
    :param hdr2: header extension from 2nd file
    :type hdr2: dict
    :param dimension: Dimension (5,6, or 7) to merge along
    :type dimension: int
    :param dim_length1: Dimension length of first file
    :type dimension: int
    :param dim_length2: Dimension length of second file
    :type dimension: int
    :return: Merged header extension dict
    :rtype: dict
    """
    out_hdr = hdr1.copy()

    def merge_list(list_1, list_2):
        return list_1 + list_2

    def merge_user_or_std(hdr_val1, hdr_val2):
        if isinstance(hdr_val1, dict)\
                and 'value' in hdr_val1:
            tmp = merge_list(hdr_val1['value'], hdr_val2['value'])
            out = hdr_val1.copy()
            out.update({'value': tmp})
            return out
        else:
            return merge_list(hdr_val1, hdr_val2)

    def merge_single(hdr_val1, hdr_val2):
        hdr_type = misc.check_type(hdr_val1)
        long_fmt_1 = misc.dim_n_header_short_to_long(hdr_val1, dim_length1)
        long_fmt_2 = misc.dim_n_header_short_to_long(hdr_val2, dim_length1)
        long_fmt = merge_user_or_std(long_fmt_1, long_fmt_2)
        if hdr_type == 'long':
            return long_fmt
        else:
            return misc.dim_n_header_long_to_short(long_fmt)

    key_str = f'dim_{dimension}_header'

    def run_check():
        # Check all other dimension fields are consistent
        dim_n = re.compile(r'dim_[567].*')
        for key in hdr1:
            if dim_n.match(key) and key != key_str:
                if hdr1[key] != hdr2[key]:
                    raise NIfTI_MRSIncompatible(f'Both files must have matching dimension headers apart from the '
                                                f'one being merged. {key} does not match.')

    if key_str in hdr1 and key_str in hdr2:
        run_check()
        # Check the subfields of the header to merge are consistent
        if not hdr1[key_str].keys() == hdr2[key_str].keys():
            raise NIfTI_MRSIncompatible(f'Both NIfTI-MRS files must have matching dim {dimension} header fields.'
                                        f'The first header contains {hdr1[key_str].keys()}. '
                                        f'The second header contains {hdr2[key_str].keys()}.')
        new_h = {}
        for sub_key in hdr1[key_str]:
            new_h[sub_key] = merge_single(hdr1[key_str][sub_key], hdr2[key_str][sub_key])

        out_hdr[key_str] = new_h
    elif key_str in hdr1 and key_str not in hdr2\
            or key_str not in hdr1 and key_str in hdr2:
        # Incompatible headers
        raise NIfTI_MRSIncompatible(f'Both NIfTI-MRS files must have matching dim {dimension} header fields')
    elif key_str not in hdr1 and key_str not in hdr2:
        # Nothing to merge - still run check
        run_check()
    return out_hdr


def reorder(nmrs, dim_tag_list):
    """Reorder the higher dimensions of a NIfTI-MRS object.
    Can force a singleton dimension with new tag.

    :param nmrs: NIFTI-MRS object to reorder.
    :type nmrs: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    :param dim_tag_list: List of dimension tags in desired order
    :type dim_tag_list: List of str
    :return: Reordered NIfTI-MRS object.
    :rtype: fsl_mrs.core.nifti_mrs.NIFTI_MRS
    """

    # Check existing tags are in the list of desired tags
    for idx, tag in enumerate(nmrs.dim_tags):
        if tag not in dim_tag_list\
                and tag is not None:
            raise NIfTI_MRSIncompatible(f'The existing tag ({tag}) does not appear '
                                        f'in the requested tag order ({dim_tag_list}).')

    # Create singleton dimensions if required
    original_dims = nmrs.ndim
    new_dim = sum(x is not None for x in nmrs.dim_tags) + 4
    dims_to_add = tuple(range(original_dims, new_dim + 1))
    data_with_singleton = np.expand_dims(nmrs.data, dims_to_add)

    # Create list of source indicies
    # Create list of destination indicies
    # Keep track of singleton tags
    source_indicies = []
    dest_indicies = []
    singleton_tags = {}
    counter = 0
    for idx, tag in enumerate(dim_tag_list):
        if tag is not None:
            if tag in nmrs.dim_tags:
                source_indicies.append(nmrs.dim_tags.index(tag) + 4)
            else:
                source_indicies.append(nmrs.ndim + counter)
                counter += 1
                singleton_tags.update({(idx + 5): tag})

            dest_indicies.append(idx + 4)

    # Sort header_ext dim_tags
    dim_n = re.compile(r'dim_[567].*')
    new_hdr_dict = {}
    for key in nmrs.hdr_ext:
        if dim_n.match(key):
            new_index = dest_indicies[source_indicies.index(int(key[4]) - 1)] + 1
            new_key = 'dim_' + str(new_index) + key[5:]
            new_hdr_dict.update({new_key: nmrs.hdr_ext[key]})
        else:
            new_hdr_dict.update({key: nmrs.hdr_ext[key]})

    # For any singleton dimensions we've added
    for dim in singleton_tags:
        new_hdr_dict.update({f'dim_{dim}': singleton_tags[dim]})

    new_header = nmrs.header.copy()
    json_s = json.dumps(new_hdr_dict)
    extension = Nifti1Extension(44, json_s.encode('UTF-8'))
    new_header.extensions.clear()
    new_header.extensions.append(extension)

    new_nmrs = NIFTI_MRS(np.moveaxis(data_with_singleton, source_indicies, dest_indicies),
                         header=new_header)

    return new_nmrs
