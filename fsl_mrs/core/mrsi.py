#!/usr/bin/env python

# core.py - main MRS class definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford
# SHBASECOPYRIGHT

from copy import deepcopy

import numpy as np

from fsl_mrs.core import MRS
from fsl_mrs.core.basis import Basis
from fsl_mrs.utils import misc


class MRSI(object):

    def __init__(self, FID, header=None,
                 cf=None, bw=None, nucleus='1H',
                 mask=None, basis=None, names=None,
                 basis_hdr=None, H2O=None):

        # process H2O
        if H2O is None:
            H2O = np.full(FID.shape[:3], None)
        elif H2O.shape != FID.shape:
            raise ValueError('H2O must be None or numpy array '
                             'of the same shape as FID.')

        # Load into properties
        self.data   = FID
        self.H2O    = H2O

        # process mask
        self.set_mask(mask)

        if header is not None:
            self.header = header
        elif cf is not None\
                and bw is not None:
            self.header = {'centralFrequency': cf,
                           'bandwidth': bw,
                           'ResonantNucleus': nucleus}
        else:
            raise ValueError('Either header or cf and bw must not be None.')

        # Basis
        if basis is not None:
            if isinstance(basis, np.ndarray):
                self._basis = Basis(basis, names, basis_hdr)
            elif isinstance(basis, Basis):
                self._basis = deepcopy(basis)
            else:
                raise TypeError('Basis must be a numpy array (+ names & headers) or a fsl_mrs.core.Basis object.')
        else:
            self._basis = None

        # tissue segmentation
        self.csf    = None
        self.wm     = None
        self.gm     = None
        self.tissue_seg_loaded  = False

        # Helpful properties
        self.spatial_shape = self.data.shape[:3]
        self.FID_points = self.data.shape[3]
        self.num_voxels = np.prod(self.spatial_shape)
        self.num_masked_voxels = np.sum(self.mask)

        # MRS output options
        self.conj_basis     = None
        self.conj_FID       = False
        self.no_conj_FID    = False
        self.rescale        = False
        self._keep          = []
        self._ignore        = []
        self._keep_ignore   = []
        self.ind_scaling    = None

        self._store_scalings = None

    @property
    def names(self):
        """Return the names of the basis spectra currently configured."""
        if self._basis is None:
            return None
        else:
            return self._basis.get_formatted_names(self._keep_ignore)

    @property
    def numBasis(self):
        """Returns the number of currently configured basis spectra"""
        if self._basis is None:
            return None
        else:
            return len(self._basis.get_formatted_names(self._keep_ignore))

    @property
    def keep(self):
        return self._keep

    @keep.setter
    def keep(self, metabs):
        """Keep a subset of metabolites in the basis by ignoring all others.

        :param metabs: List of metabolite names to keep.
        :type metabs: List of str
        """
        if metabs is None or metabs == []:
            self._keep = []
            self._keep_ignore = []
            if self._ignore != []:
                self.ignore = self._ignore
            return

        for m in metabs:
            if m not in self.names:
                raise ValueError(f'{m} not in current list of metabolites'
                                 f' ({self.names}).')

        self._keep = metabs
        self._keep_ignore += [m for m in self.names if m not in metabs]

    @property
    def ignore(self):
        return self._ignore

    @ignore.setter
    def ignore(self, metabs):
        """Ignore a subset of metabolites in the basis

        :param metabs: List of metabolite names to remove
        :type metabs: List of str
        """
        if metabs is None or metabs == []:
            self._ignore = []
            self._keep_ignore = []
            if self._keep != []:
                self.keep = self._keep
            return

        for m in metabs:
            if m not in self.names:
                raise ValueError(f'{m} not in current list of metabolites'
                                 f' ({self.names}).')
        self._ignore += metabs
        self._keep_ignore += metabs

    def __iter__(self):
        shape = self.data.shape
        self._store_scalings = []
        for idx in np.ndindex(shape[:3]):
            if self.mask[idx]:
                mrs_out = MRS(FID=self.data[idx],
                              header=self.header,
                              basis=self._basis,
                              H2O=self.H2O[idx])

                self._process_mrs(mrs_out)
                self._store_scalings.append(mrs_out.scaling)

                if self.tissue_seg_loaded:
                    tissue_seg = {'CSF': self.csf[idx],
                                  'WM': self.wm[idx],
                                  'GM': self.gm[idx]}
                else:
                    tissue_seg = None

                yield mrs_out, idx, tissue_seg

    def get_indicies_in_order(self, mask=True):
        """Return a list of iteration indices in order"""
        out = []
        shape = self.data.shape
        for idx in np.ndindex(shape[:3]):
            if mask:
                if self.mask[idx]:
                    out.append(idx)
            else:
                out.append(idx)
        return out

    def get_scalings_in_order(self, mask=True):
        """Return a list of MRS object scalings in order"""
        if self._store_scalings is None:
            raise ValueError('Fetch mrs by iterable first.')
        else:
            return self._store_scalings

    def mrs_by_index(self, index):
        ''' Return MRS object by index (tuple - x,y,z).'''
        if not np.array_equal(self.H2O, np.full(self.data.shape[:3], None)):
            H2O = self.H2O[index[0], index[1], index[2], :]
        else:
            H2O = None
        mrs_out = MRS(FID=self.data[index[0], index[1], index[2], :],
                      header=self.header,
                      basis=self._basis,
                      H2O=H2O)
        self._process_mrs(mrs_out)
        return mrs_out

    def mrs_from_average(self):
        '''
        Return average of all masked voxels
        as a single MRS object.
        '''
        FID = misc.volume_to_list(self.data, self.mask)
        FID = sum(FID) / len(FID)
        if not np.array_equal(self.H2O, np.full(self.data.shape[:3], None)):
            H2O = misc.volume_to_list(self.H2O, self.mask)
            H2O = sum(H2O) / len(H2O)
        else:
            H2O = None

        mrs_out = MRS(FID=FID,
                      header=self.header,
                      basis=self._basis,
                      H2O=H2O)
        self._process_mrs(mrs_out)
        return mrs_out

    def seg_by_index(self, index):
        '''Return segmentation information by index.'''
        if self.tissue_seg_loaded:
            return {'CSF': self.csf[index],
                    'WM': self.wm[index],
                    'GM': self.gm[index]}
        else:
            raise ValueError('Load tissue segmentation first.')

    def _process_mrs(self, mrs):
        ''' Process (conjugate, rescale)
            basis and FID and apply basis operations
            to all voxels.
        '''
        if self._basis is not None:
            if self.conj_basis is True:
                mrs.conj_Basis = True
            elif self.conj_basis is False:
                mrs.conj_Basis = False
            else:
                mrs.check_Basis(repair=True)

            mrs.keep = self._keep
            mrs.ignore = self._ignore

        if self.conj_FID:
            mrs.conj_FID = True
        elif self.no_conj_FID:
            pass
        else:
            mrs.check_FID(repair=True)

        if self.rescale:
            mrs.rescaleForFitting(ind_scaling=self.ind_scaling)

    def plot(self, mask=True, ppmlim=None):
        '''Plot (masked) grid of spectra.'''
        import matplotlib.pyplot as plt

        if ppmlim is None:
            ppmlim = self.mrs_from_average().default_ppm_range

        if mask:
            mask_indices = np.where(self.mask)
        else:
            mask_indices = np.where(np.full(self.mask.shape, True))
        dim1 = np.asarray((np.min(mask_indices[0]), np.max(mask_indices[0])))
        dim2 = np.asarray((np.min(mask_indices[1]), np.max(mask_indices[1])))
        dim3 = np.asarray((np.min(mask_indices[2]), np.max(mask_indices[2])))

        size1 = 1 + dim1[1] - dim1[0]
        size2 = 1 + dim2[1] - dim2[0]
        size3 = 1 + dim3[1] - dim3[0]

        ar1 = size1 / (size1 + size2)
        ar2 = size2 / (size1 + size2)

        for sDx in range(size3):
            # import pdb; pdb.set_trace()
            index_in_slice = mask_indices[2] == sDx
            slice_indices = (x[index_in_slice] for x in mask_indices)

            if not any(index_in_slice):
                continue
            fig, axes = plt.subplots(size1, size2, figsize=(20 * ar2, 20 * ar1))

            for i, j, k in zip(*slice_indices):
                ii = i - dim1[0]
                jj = j - dim2[0]
                ax = axes[ii, jj]
                if (not self.mask[i, j, k]) and mask:
                    continue
                mrs = self.mrs_by_index([i, j, k])
                ax.plot(mrs.getAxes(ppmlim=ppmlim), np.real(mrs.get_spec(ppmlim=ppmlim)))
                ax.invert_xaxis()

            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            plt.subplots_adjust(left=0.03,  # the left side of the subplots of the figure
                                right=0.97,   # the right side of the subplots of the figure
                                bottom=0.01,  # the bottom of the subplots of the figure
                                top=0.95,     # the top of the subplots of the figure
                                wspace=0,  # the amount of width reserved for space between subplots,
                                hspace=0)
            fig.suptitle(f'Slice {sDx}')
            plt.show()

    def __str__(self):
        return f'MRSI with shape {self.data.shape}\n' \
               f'Number of voxels = {self.num_voxels}\n' \
               f'Number of masked voxels = {self.num_masked_voxels}'

    def __repr__(self):
        return str(self)

    def set_mask(self, mask):
        """ Load mask as numpy array."""
        if mask is None:
            mask = np.full(self.data.shape[0:3], True)
        elif mask.shape[0:3] == self.data.shape[0:3]:
            mask = mask != 0.0
        else:
            raise ValueError(f'Mask must be None or numpy array of the same shape as FID.'
                             f' Mask {mask.shape[0:3]}, FID {self.data.shape[0:3]}.')

        self.mask = mask
        self.num_masked_voxels = np.sum(self.mask)

    def set_tissue_seg(self, csf, wm, gm):
        """ Load tissue segmentation as numpy arrays."""
        if (csf.shape != self.spatial_shape) or (wm.shape != self.spatial_shape) or (gm.shape != self.spatial_shape):
            raise ValueError(f'Tissue segmentation arrays have wrong shape '
                             f'(CSF:{csf.shape}, GM:{gm.shape}, WM:{wm.shape}).'
                             f' Must match FID ({self.spatial_shape}).')

        self.csf = csf
        self.wm = wm
        self.gm = gm
        self.tissue_seg_loaded = True

    def list_to_matched_array(self, data_list, indicies=None, cleanup=True, dtype=float):
        '''Convert 3D or 4D array of data indexed from an mrsi object
        to a  numpy array matching the shape of the mrsi data.'''
        if indicies is None:
            indicies = self.get_indicies_in_order()

        # Deal with the variable types (float vs np.float64) that pandas
        # seems to generate depending on (python?) version.
        if isinstance(data_list[0], (float, int)):
            nt = 1
        else:
            nt = data_list[0].size

        if nt > 1:
            data = np.zeros(self.spatial_shape + (nt,), dtype=dtype)
        else:
            data = np.zeros(self.spatial_shape, dtype=dtype)

        for d, ind in zip(data_list, indicies):
            data[ind] = d

        if cleanup:
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0
            data[data < 1e-10]   = 0
            data[data > 1e10]    = 0

        return data

    def list_to_correlation_array(self, data_list, indicies=None, cleanup=True, dtype=float):
        '''Convert 5D array of correlation matrices indexed from an MRSI object
        to a numpy array with the shape of the first three dimensions matching
        that of the MRSI object.'''
        if indicies is None:
            indicies = self.get_indicies_in_order()

        size_m, size_n = data_list[0].shape
        if size_m != size_n:
            raise ValueError(f'Only symmetric matrices are handled, size is ({size_m},{size_n}).')
        data = np.zeros(self.spatial_shape + (size_m, size_n), dtype=dtype)

        for d, ind in zip(data_list, indicies):
            data[ind] = d

        if cleanup:
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0

        return data

    def check_basis(self, ppmlim=None):
        """Check orientation of basis using a single generated mrs object.

        :param ppmlim: Region of expected signal, defaults to nucleus standard
        :type ppmlim: tuple, optional
        """
        if self._basis is not None:
            mrs = self.mrs_by_index((0, 0, 0))
            mrs.check_Basis(ppmlim=ppmlim, repair=True)
            self.conj_basis = mrs.conj_Basis
        else:
            raise AttributeError('MRSI._basis not populated, add basis.')
