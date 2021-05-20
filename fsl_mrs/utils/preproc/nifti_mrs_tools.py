"""Tools for merging, splitting and reordering the dimensions of NIfTI-MRS

    Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
    Copyright (C) 2021 University of Oxford
"""

import numpy as np
from fsl_mrs.core.nifti_mrs import NIFTI_MRS, NIFTIMRS_DimDoesntExist


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
        index = np.arange(index_or_indicies, nmrs.shape[dim_index])

    elif isinstance(index_or_indicies, list):
        if not np.logical_and(np.asarray(index_or_indicies) >= 0,
                              np.asarray(index_or_indicies) <= nmrs.shape[dim_index]).all():
            raise ValueError('index_or_indicies must have elements between 0 and N,'
                             f' where N is the size of the specified dimension ({nmrs.shape[dim_index]}).')
        index = index_or_indicies

    else:
        raise TypeError('index_or_indicies must be single index or list of indicies')

    nmrs_1 = NIFTI_MRS(np.delete(nmrs.data, index, axis=dim_index), header=nmrs.header)
    nmrs_2 = NIFTI_MRS(np.take(nmrs.data, index, axis=dim_index), header=nmrs.header)

    return nmrs_1, nmrs_2


class NIfTI_MRSIncompatible(Exception):
    pass


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
        if not check_tag(nmrs):
            raise NIfTI_MRSIncompatible('The tags of all concatentated objects must match.'
                                        f' The tags ({nmrs.dim_tags}) of the {idx} object does'
                                        f' not match that of the first ({array_of_nmrs[0].dim_tags}).')
        # Check dim tags for compatibility
        to_concat.append(nmrs.data)

    return NIFTI_MRS(np.concatenate(to_concat, axis=dim_index), header=array_of_nmrs[0].header)
