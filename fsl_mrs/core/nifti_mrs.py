# NIFTI_MRS.py - NFITI_MRS class definition
# This module is primarily a shim around the niftimrs package
# The intention is to extend the functionality of the
# nifti-mrs package definitions for use in FSL-MRS
# The need for this aroise when splitting these useful generic nifti-mrs
# tools off from fsl-mrs, but wanting to keep the original fsl-mrs api
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2021 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from nifti_mrs import nifti_mrs
from nifti_mrs import create_nmrs
from nifti_mrs import tools
import fsl_mrs.core as core


def gen_nifti_mrs(*args, **kwargs):
    """Generate NIfTI-MRS from data and required metadata

    This is a wrapper around the nifti_mrs.create_nmrs.gen_nifti_mrs function to
     implement additional functionality for FSL-MRS

    :param data: Complex-typed numpy array of at least 4 dimensions (max 7)
    :type data: numpy.array
    :param dwelltime: Spectral (4th dimension) dwelltime in seconds
    :type dwelltime: float
    :param spec_freq: Spectrometer Frequency in MHz
    :type spec_freq: float
    :param nucleus: Resonant Nucleus string (e.g. 1H, 31P, 2H), defaults to '1H'
    :type nucleus: str, optional
    :param affine: 4x4 orientation/position affine, defaults to None which will use default (scaled identity).
    :type affine: numpy.array, optional
    :param dim_tags: List of dimension tags (e.g. DIM_DYN), defaults to [None, None, None]
    :type dim_tags: list, optional
    :param nifti_version: Version of NIfTI header format, defaults to 2
    :type nifti_version: int, optional
    :param no_conj: If true stops conjugation of data on creation, defaults to False
    :type no_conj: bool, optional
    :return: NIfTI-MRS object
    :rtype: nifti_mrs.nifti_mrs.NIFTI_MRS
    """
    # To ensure the additional FSL-MRS functionallity present.
    return NIFTI_MRS(create_nmrs.gen_nifti_mrs(*args, **kwargs))


class NIFTI_MRS(nifti_mrs.NIFTI_MRS):
    """A class to load and represent NIfTI-MRS formatted data.
    Utilises the fslpy Image class and nibabel nifti headers."""

    def __init__(self, *args, **kwargs):
        """Create a NIFTI_MRS object with the given image data or file name.

        This is a wrapper around the nifti_mrs.nifti_mrs.NIFTI_MRS class to
         implement additional functionality for FSL-MRS

        Aguments mirror those of the leveraged fsl.data.image.Image class.

        :arg image:      A string containing the name of an image file to load,
                         or a Path object pointing to an image file, or a
                         :mod:`numpy` array, or a :mod:`nibabel` image object,
                         or an ``Image`` object.

        :arg name:       A name for the image.

        :arg header:     If not ``None``, assumed to be a
                         :class:`nibabel.nifti1.Nifti1Header` or
                         :class:`nibabel.nifti2.Nifti2Header` to be used as the
                         image header. Not applied to images loaded from file,
                         or existing :mod:`nibabel` images.

        :arg xform:      A :math:`4\\times 4` affine transformation matrix
                         which transforms voxel coordinates into real world
                         coordinates. If not provided, and a ``header`` is
                         provided, the transformation in the header is used.
                         If neither a ``xform`` nor a ``header`` are provided,
                         an identity matrix is used. If both a ``xform`` and a
                         ``header`` are provided, the ``xform`` is used in
                         preference to the header transformation.

        :arg dataSource: If ``image`` is not a file name, this argument may be
                         used to specify the file from which the image was
                         loaded.

        :arg loadMeta:   If ``True``, any metadata contained in JSON sidecar
                         files is loaded and attached to this ``Image`` via
                         the :class:`.Meta` interface. if ``False``, metadata
                         can be loaded at a later stage via the
                         :func:`loadMeta` function. Defaults to ``False``.

        :arg dataMgr:    Object implementing the :class:`DataManager`
                         interface, for managing access to the image data.

        All other arguments are passed through to the ``nibabel.load`` function
        (if it is called).
        """
        super().__init__(*args, **kwargs)

    def copy(self, remove_dim=None):
        """Return a copy of this image, optionally with a dimension removed.

        :param remove_dim: dimension index (4, 5, 6) or tag. None iterates over all indices., defaults to None
        :type remove_dim: str or int, optional
        :return: Copy of object
        :rtype: NIFTI_MRS
        """
        return NIFTI_MRS(super().copy(remove_dim=remove_dim))

    def generate_mrs(self, dim=None, basis_file=None, basis=None, ref_data=None):
        """Generator for MRS or MRSI objects from the data, optionally returning a whole dimension as a list.

        :param dim: Dimension to generate over, dimension index (4, 5, 6) or tag. None iterates over all indices,
            defaults to None
        :type dim: Int or str, optional
        :param basis_file: Path to basis file, defaults to None
        :type basis_file: Str, optional
        :param basis: Basis object, defaults to None
        :type basis: fsl_mrs.core.basis.Basis, optional
        :param ref_data: Reference data as a path string, NIfTI-MRS object or ndarray, defaults to None
        :type ref_data: Str or fsl_mrs.core.NIFTI_MRS or numpy.ndarray, optional
        :yield: MRS or MRSI object
        :rtype: fsl_mrs.core.MRS or fsl_mrs.core.MRSI
        """

        if basis_file is not None:
            import fsl_mrs.utils.mrs_io as mrs_io
            basis = mrs_io.read_basis(basis_file)

        if ref_data is not None:
            if isinstance(ref_data, str):
                ref_data = NIFTI_MRS(ref_data)[:]
            elif isinstance(ref_data, NIFTI_MRS):
                ref_data = ref_data[:]
            elif isinstance(ref_data, np.ndarray):
                pass
            else:
                raise TypeError('ref_data must be a path to a NIFTI-MRS file,'
                                'a NIFTI_MRS object, or a numpy array.')

        for data, _ in self.iterate_over_dims(dim=dim):
            if np.prod(data.shape[:3]) > 1:
                # Generate MRSI objects
                if data.ndim > 4:
                    pass
                    out = []
                    for dd in np.moveaxis(data.reshape(*data.shape[:4], -1), -1, 0):
                        out.append(core.MRSI(FID=dd,
                                             bw=self.bandwidth,
                                             cf=self.spectrometer_frequency[0],
                                             nucleus=self.nucleus[0],
                                             basis=basis,
                                             H2O=ref_data))
                    yield out
                else:
                    yield core.MRSI(FID=data,
                                    bw=self.bandwidth,
                                    cf=self.spectrometer_frequency[0],
                                    nucleus=self.nucleus[0],
                                    basis=basis,
                                    H2O=ref_data)
            else:
                if ref_data is not None:
                    ref_data = ref_data.squeeze()

                # Generate MRS objects
                if data.ndim > 4:
                    out = []
                    for dd in np.moveaxis(data.reshape(*data.shape[:4], -1), -1, 0):
                        out.append(core.MRS(FID=dd.squeeze(),
                                            bw=self.bandwidth,
                                            cf=self.spectrometer_frequency[0],
                                            nucleus=self.nucleus[0],
                                            basis=basis,
                                            H2O=ref_data))
                    yield out
                else:
                    yield core.MRS(FID=data.squeeze(),
                                   bw=self.bandwidth,
                                   cf=self.spectrometer_frequency[0],
                                   nucleus=self.nucleus[0],
                                   basis=basis,
                                   H2O=ref_data)

    def mrs(self, *args, **kwargs):
        out = list(self.generate_mrs(*args, **kwargs))
        if len(out) == 1:
            out = out[0]
        return out


# Shims around the nifti_mrs.tools functions
# Force these tools to return and FSL-MRS extended NIFTI_MRS class object
def conjugate(nmrs):
    """Conjugate a nifti-mrs object.

    :param nmrs: NIFTI_MRS object to conjugate
    :type nmrs: NIFTI_MRS
    :return: Conjugated NIFTI_MRS
    :rtype: NIFTI_MRS
    """
    return NIFTI_MRS(tools.conjugate(nmrs))


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
    return NIFTI_MRS(tools.merge(array_of_nmrs, dimension))


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
    return (NIFTI_MRS(x) for x in tools.split(nmrs, dimension, index_or_indicies))


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
    return NIFTI_MRS(tools.reorder(nmrs, dim_tag_list))


def reshape(nmrs, reshape, d5=None, d6=None, d7=None):
    """Reshape the higher dimensions (5-7) of an nifti-mrs file.
    Uses numpy reshape syntax to reshape. Use -1 for automatic sizing.

    If the dimension exists after reshaping a tag is required. If None is passed
    but one already exists no change will be made. If no value exists then an
    exception will be raised.

    :param nmrs: Input NIfTI-MRS file
    :type nmrs: NIFTI_MRS
    :param reshape: Tuple of target sizes in style of numpy.reshape, higher dimensions only.
    :type reshape: tuple
    :param d5: Dimension tag to set dim_5, defaults to None
    :type d5: str, optional
    :param d6: Dimension tag to set dim_6, defaults to None
    :type d6: str, optional
    :param d7: Dimension tag to set dim_7, defaults to None
    :type d7: str, optional
    """
    return NIFTI_MRS(tools.reshape(nmrs, reshape, d5=d5, d6=d6, d7=d7))
