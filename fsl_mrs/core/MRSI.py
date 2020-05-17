#!/usr/bin/env python

# core.py - main MRS class definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.core import MRS
from fsl_mrs.utils import mrs_io,plotting,fitting
import matplotlib.pyplot as plt

class MRSI(object):
            
    def __init__(self,FID,header,mask=None,basis=None,names=None,basis_hdr=None,H2O=None):
        
        # process mask
        if mask is None:
            mask = np.full(FID.shape,True)
        elif mask.shape[0:3]==FID.shape[0:3]:
            mask = mask!=0.0
        else:
            raise ValueError(f'Mask must be None or numpy array of the same shape as FID. Mask {mask.shape[0:3]}, FID {FID.shape[0:3]}.')
        
        # process H2O
        if H2O is None:
            H2O = np.full(FID.shape,None)
        elif H2O.shape!=FID.shape:
            raise ValueError('H2O must be None or numpy array of the same shape as FID.')
        
        # Load into properties
        self.data = FID
        self.H2O = H2O
        self.mask = mask
        self.header = header
        
        # Basis 
        self.basis        = basis
        self.names        = names
        self.basis_hdr = basis_hdr
        
        # Helpful properties
        self.spatial_shape = self.data.shape[:3]
        self.FID_points = self.data.shape[3]
        self.num_voxels = np.prod(self.spatial_shape)
        self.num_masked_voxels = np.sum(self.mask)
        if self.names is not None:
            self.num_basis = len(names)
        
    def __iter__(self):
        shape = self.data.shape
        for idx in np.ndindex(shape[:3]):
            if self.mask[idx]:
                mrs_out = MRS(FID=self.data[idx],
                                header=self.header,
                                basis=self.basis,
                                names=self.names,
                                basis_hdr=self.basis_hdr,
                                H2O=self.H2O[idx])
                mrs_out.check_FID(repair=True)
                mrs_out.check_Basis(repair=True)
                yield mrs_out,idx
    
    def mrsByIndex(self,index):
        mrs_out = MRS(FID=self.data[index[0],index[1],index[2],:],
                                header=self.header,
                                basis=self.basis,
                                names=self.names,
                                basis_hdr=self.basis_hdr,
                                H2O=self.H2O[index[0],index[1],index[2],:])
        mrs_out.check_FID(repair=True)
        mrs_out.check_Basis(repair=True)
        return mrs_out
                
    def plot(self,mask=True,ppmlim=(0.2,4.2)):
        if mask:
            mask_indicies = np.where(self.mask)
        else:
            mask_indicies = np.where(np.full(self.mask.shape,True))
        dim1 = np.asarray((np.min(mask_indicies[0]),np.max(mask_indicies[0])))
        dim2 = np.asarray((np.min(mask_indicies[1]),np.max(mask_indicies[1])))
        dim3 = np.asarray((np.min(mask_indicies[2]),np.max(mask_indicies[2])))

        size1 = 1+ dim1[1]-dim1[0]
        size2 = 1+ dim2[1]-dim2[0]
        size3 = 1+ dim3[1]-dim3[0]

        ar1 = size1/(size1+size2)
        ar2 = size2/(size1+size2)

        for sDx in range(size3):
            fig,axes = plt.subplots(size1,size2,figsize=(20*ar2,20*ar1))
            for i,j,k in zip(*mask_indicies):
                if (not self.mask[i,j,k]) and mask:
                    continue
                ii = i - dim1[0]
                jj = j - dim2[0]
                ax = axes[ii,jj]
                mrs = self.mrsByIndex([i,j,k])
                ax.plot(mrs.getAxes(ppmlim=ppmlim),np.real(mrs.getSpectrum(ppmlim=ppmlim)))
                ax.invert_xaxis()
                ax.set_xticks([])
                ax.set_yticks([])
            plt.subplots_adjust(left = 0.03,  # the left side of the subplots of the figure
                                right = 0.97,   # the right side of the subplots of the figure
                                bottom = 0.01,  # the bottom of the subplots of the figure
                                top = 0.95,     # the top of the subplots of the figure
                                wspace = 0,  # the amount of width reserved for space between subplots,
                                hspace = 0)
            fig.suptitle(f'Slice {k}')        
    
    def __str__(self):
        return f'MRSI with shape {self.data.shape}\nNumber of voxels = {self.num_voxels}\nNumber of masked voxels = {self.num_masked_voxels}'
    def __repr__(self):
        return str(self)