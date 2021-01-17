# NIFTI_MRS.py - NFITI MRS class definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2021 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
import json
from fsl.data.image import Image
from fsl_mrs.core import MRS, MRSI
from nibabel.nifti1 import Nifti1Extension

from nibabel.nifti2 import Nifti2Header
from fsl_mrs.utils.misc import checkCFUnits


def gen_new_nifti_mrs(data, dwelltime, spec_freq, nucleus='1H', affine=None, dim_tags=[None, None, None]):
    '''Generate a NIFTI_MRS object from a np array and header info.

    :param np.ndarray data: FID (time-domain) data. Must be atleast 4D.
    :param float dwelltime: Dwelltime of FID data in seconds.
    :param float spec_freq: Spectrometer (or central) frequency in MHz.
    :param str nucleus: Nucleus string, defaults to '1H'
    :param np.ndarray affine: Optional 4x4 position affine.
    :param [str] affine: List of dimension tags.

    :return: NIFTI_MRS object
    '''

    if not np.iscomplexobj(data):
        raise ValueError('data must be complex')
    if data.ndim < 4 or data.ndim > 7:
        raise ValueError(f'data must between 4 and 7 dimensions, currently has {data.ndim}')

    header = Nifti2Header()
    header['pixdim'][4] = dwelltime
    hdr_dict = {'SpectrometerFrequency': [checkCFUnits(spec_freq, units='MHz'), ],
                'ResonantNucleus': [nucleus, ]}

    for idx, dt in enumerate(dim_tags):
        if dt is not None:
            if (idx + 4) > data.ndim:
                raise ValueError('Too many dimension tags passed.')
            hdr_dict[f'dim_{idx+5}'] = dt

    json_s = json.dumps(hdr_dict)
    extension = Nifti1Extension(44, json_s.encode('UTF-8'))
    header.extensions.append(extension)

    header.set_qform(affine)
    header.set_sform(affine)

    header['intent_name'] = 'mrs_v0_2'.encode()

    return NIFTI_MRS(data, header=header)


class NIFTIMRS_DimDoesntExist(Exception):
    pass


class NotNIFTI_MRS(Exception):
    pass


class NIFTI_MRS(Image):
    """Load NIFTI MRS format data. Derived from nibabel's Nifti2Image."""
    def __init__(self, *args, **kwargs):
        # If generated from np array include conjugation
        # to make sure storage is right-handed
        if isinstance(args[0], np.ndarray):
            args = list(args)
            args[0] = args[0].conj()
        super().__init__(*args, **kwargs)

        # Check that file meets minimum requirements
        try:
            nmrs_version = self.mrs_nifti_version
            if float(nmrs_version) < 0.2:
                raise NotNIFTI_MRS('NIFTI-MRS > V0.2 required.')
        except IndexError:
            raise NotNIFTI_MRS('NIFTI-MRS intent code not set.')

        if 44 not in self.header.extensions.get_codes():
            raise NotNIFTI_MRS('NIFTI-MRS must have a header extension.')

        try:
            self.nucleus
            self.spectrometer_frequency
        except KeyError:
            raise NotNIFTI_MRS('NIFTI-MRS header extension must have nucleus and spectrometerFrequency keys.')

        # Extract key parameters from the header extension
        self._set_dim_tags()

    def _set_dim_tags(self):
        self.dim_tags = [None, None, None]
        std_tags = ['DIM_COIL', 'DIM_DYN', 'DIM_INDIRECT_0']
        for idx in range(3):
            curr_dim = idx + 5
            if self.ndim >= curr_dim:
                curr_tag = f'dim_{curr_dim}'
                if curr_tag in self.hdr_ext:
                    self.dim_tags[idx] = self.hdr_ext[curr_tag]
                else:
                    self.dim_tags[idx] = std_tags[idx]

    def __getitem__(self, sliceobj):
        '''Apply conjugation at use. This swaps from the
        NIFTI-MRS and Levvit inspired right-handed reference frame
        to a left-handed one, which FSL-MRS development started in.'''
        # print(f'getting {sliceobj} to conjugate {super().__getitem__(sliceobj)}')
        return super().__getitem__(sliceobj).conj()

    def __setitem__(self, sliceobj, values):
        '''Apply conjugation back at write. This swaps from the
        FSL-MRS left handed convention to the NIFTI-MRS and Levvit
        inspired right-handed reference frame.'''
        # print(f'setting {sliceobj} to conjugate of {values[0]}')
        # print(super().__getitem__(sliceobj)[0])
        super().__setitem__(sliceobj, values.conj())
        # print(super().__getitem__(sliceobj)[0])

    @property
    def mrs_nifti_version(self):
        '''Get version string'''
        tmp_vstr = self.header.get_intent()[2].split('_')
        return tmp_vstr[1].lstrip('v') + '.' + tmp_vstr[2]

    @property
    def dwelltime(self):
        '''Return bandwidth in seconds'''
        return self.header['pixdim'][4]

    @dwelltime.setter
    def dwelltime(self, new_dt):
        self.header['pixdim'][4] = new_dt

    @property
    def bandwidth(self):
        '''Return spectral width in Hz'''
        return 1 / self.dwelltime

    @property
    def nucleus(self):
        return self.hdr_ext['ResonantNucleus']

    @property
    def spectrometer_frequency(self):
        '''Central or spectrometer frequency in MHz - returns list'''
        return self.hdr_ext['SpectrometerFrequency']

    @property
    def hdr_ext(self):
        '''Return MRS JSON header extension as python dict'''
        hdr_ext_codes = self.header.extensions.get_codes()
        return json.loads(self.header.extensions[hdr_ext_codes.index(44)].get_content())

    @hdr_ext.setter
    def hdr_ext(self, hdr_dict):
        '''Update MRS JSON header extension from python dict'''
        json_s = json.dumps(hdr_dict)
        extension = Nifti1Extension(44, json_s.encode('UTF-8'))
        self.header.extensions.clear()
        self.header.extensions.append(extension)

    def dim_position(self, dim_tag):
        '''Return position of dim if it exists.'''
        if dim_tag in self.dim_tags:
            return self._dim_tag_to_index(dim_tag)
        else:
            raise NIFTIMRS_DimDoesntExist(f"{dim_tag} doesn't exist in list of tags: {self.dim_tags}")

    def _dim_tag_to_index(self, dim):
        '''Convert DIM tag str or index (4, 5, 6) to numpy dimension index'''
        if isinstance(dim, str):
            if dim in self.dim_tags:
                dim = self.dim_tags.index(dim)
                dim += 4
        return dim

    def copy(self, remove_dim=None):
        '''Return a copy of this image, optionally with a dimension removed.
        Args:
            dim - None, dimension index (4, 5, 6) or tag. None iterates over all indices.'''
        if remove_dim:
            dim = self._dim_tag_to_index(remove_dim)
            reduced_data = self.data.take(0, axis=dim)
            new_obj = NIFTI_MRS(reduced_data, header=self.header)

            # Modify the dim information in
            hdr_ext = new_obj.hdr_ext
            # Remove the dim_ tag from hdr_ext
            dim += 1
            for dd in range(dim, 8):
                if dd > new_obj.ndim:
                    hdr_ext.pop(f'dim_{dd}', None)
                    hdr_ext.pop(f'dim_{dd}_header', None)
                elif dd >= dim:
                    hdr_ext[f'dim_{dd}'] = hdr_ext[f'dim_{dd + 1}']
                    if f'dim_{dd + 1}_header' in hdr_ext:
                        hdr_ext[f'dim_{dd}_header'] = hdr_ext[f'dim_{dd + 1}_header']
            new_obj.hdr_ext = hdr_ext

            new_obj._set_dim_tags()

            return new_obj
        else:
            return NIFTI_MRS(self.data, header=self.header)

    def iterate_over_dims(self, dim=None, iterate_over_space=False, reduce_dim_index=False, voxel_index=None):
        '''Return generator to iterate over all indices or one dimension (and FID).
        Args:
            dim - None, dimension index (4, 5, 6) or tag. None iterates over all indices.
            iterate_over_space - If True also iterate over spatial dimensions.
            reduce_dim_index - If True the returned slice index will have the selected dimension removed.
            voxel_index - slice or tuple of first three spatial dimensions.
        Returns:
            data - numpy array of sliced data
            index - data location slice object.
        '''

        data = self.data
        dim = self._dim_tag_to_index(dim)

        # Convert indicies to slices to preserve singleton dimensions
        if voxel_index is not None:
            tmp = []
            for vi in voxel_index:
                if isinstance(vi, slice):
                    tmp.append(vi)
                elif isinstance(vi, int):
                    tmp.append(slice(vi, vi + 1))
                else:
                    raise TypeError('voxel index elements must be slice or int type.')
            voxel_index = tuple(tmp)

        def calc_slice_idx(idx):
            if iterate_over_space:
                slice_obj = list(idx[:3]) + [slice(None), ] + list(idx[3:])
            else:
                slice_obj = [slice(None), slice(None), slice(None), slice(None)]\
                    + list(idx[0:])
            if dim is not None and not reduce_dim_index:
                slice_obj.insert(dim + 1, slice(None))
            return tuple(slice_obj)

        if isinstance(dim, (int, str)):
            # Move FID dim to last
            data = np.moveaxis(data, 3, -1)
            dim -= 1
            # Move identified dim to last
            data = np.moveaxis(data, dim, -1)

            if voxel_index is not None:
                voxel_index
                data = data[voxel_index]

            if iterate_over_space:
                iteration_skip = -2
            else:
                data = np.moveaxis(data, (0, 1, 2), (-5, -4, -3))
                iteration_skip = -5

            for idx in np.ndindex(data.shape[:iteration_skip]):
                yield data[idx], calc_slice_idx(idx)

        elif dim is None:
            # Move FID dim to last
            data = np.moveaxis(data, 3, -1)

            if voxel_index is not None:
                data = data[voxel_index]

            if iterate_over_space:
                iteration_skip = -1
            else:
                data = np.moveaxis(data, (0, 1, 2), (-4, -3, -2))
                iteration_skip = -4

            for idx in np.ndindex(data.shape[:iteration_skip]):
                yield data[idx], calc_slice_idx(idx)

        else:
            raise TypeError('dim should be int or a string matching one of the dim tags.')

    def generate_mrs(self, dim=None, basis_file=None, basis=None, names=None, basis_hdr=None, ref_data=None):
        """Generator for MRS or MRSI objects from the data, optionally returning a whole dimension as a list."""

        if basis_file is not None:
            import fsl_mrs.utils.mrs_io as mrs_io
            basis, names, basis_hdr = mrs_io.read_basis(basis_file)
            basis_hdr = basis_hdr[0]

        if ref_data is not None:
            if isinstance(ref_data, str):
                ref_data = NIFTI_MRS(ref_data).data
            elif isinstance(ref_data, NIFTI_MRS):
                ref_data = ref_data.data
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
                        out.append(MRSI(FID=dd,
                                        bw=self.bandwidth,
                                        cf=self.spectrometer_frequency[0],
                                        nucleus=self.nucleus[0],
                                        basis=basis,
                                        names=names,
                                        basis_hdr=basis_hdr,
                                        H2O=ref_data))
                    yield out
                else:
                    yield MRSI(FID=data,
                               bw=self.bandwidth,
                               cf=self.spectrometer_frequency[0],
                               nucleus=self.nucleus[0],
                               basis=basis,
                               names=names,
                               basis_hdr=basis_hdr,
                               H2O=ref_data)
            else:
                if ref_data is not None:
                    ref_data = ref_data.squeeze()

                # Generate MRS objects
                if data.ndim > 4:
                    out = []
                    for dd in np.moveaxis(data.reshape(*data.shape[:4], -1), -1, 0):
                        out.append(MRS(FID=dd.squeeze(),
                                       bw=self.bandwidth,
                                       cf=self.spectrometer_frequency[0],
                                       nucleus=self.nucleus[0],
                                       basis=basis,
                                       names=names,
                                       basis_hdr=basis_hdr,
                                       H2O=ref_data))
                    yield out
                else:
                    yield MRS(FID=data.squeeze(),
                              bw=self.bandwidth,
                              cf=self.spectrometer_frequency[0],
                              nucleus=self.nucleus[0],
                              basis=basis,
                              names=names,
                              basis_hdr=basis_hdr,
                              H2O=ref_data)

    def mrs(self, *args, **kwargs):
        out = list(self.generate_mrs(*args, **kwargs))
        if len(out) == 1:
            out = out[0]
        return out

    # If save called do some hairy temporary conjugation
    # The save method of fsl.data.image.Image makes a call
    # to __getitem__ resulting in a final undesired conjugation.
    def save(self, filename=None):
        self[:] = self[:].conj()
        super().save(filename=filename)
