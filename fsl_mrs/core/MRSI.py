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
from fsl_mrs.utils import mrs_io, misc
import matplotlib.pyplot as plt
import nibabel as nib
from fsl_mrs.utils.mrs_io.fsl_io import saveNIFTI

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
        
        # tissue segmentation
        self.csf = None
        self.wm = None
        self.gm = None
        self.tissue_seg_loaded = False

        # Helpful properties
        self.spatial_shape = self.data.shape[:3]
        self.FID_points = self.data.shape[3]
        self.num_voxels = np.prod(self.spatial_shape)
        self.num_masked_voxels = np.sum(self.mask)
        if self.names is not None:
            self.num_basis = len(names)

        # MRS output options
        self.conj_basis     = False
        self.no_conj_basis  = False
        self.conj_FID       = False
        self.no_conj_FID    = False
        self.rescale        = False
        self.keep           = None
        self.ignore         = None
        self.ind_scaling    = None

        self._store_scalings = None
        
    def __iter__(self):
        shape = self.data.shape
        self._store_scalings = []
        for idx in np.ndindex(shape[:3]):
            if self.mask[idx]:
                mrs_out = MRS(FID=self.data[idx],
                                header=self.header,
                                basis=self.basis,
                                names=self.names,
                                basis_hdr=self.basis_hdr,
                                H2O=self.H2O[idx])
                
                self._process_mrs(mrs_out)
                self._store_scalings.append(mrs_out.scaling)

                if self.tissue_seg_loaded:
                    tissue_seg = {'CSF':self.csf[idx],'WM':self.wm[idx],'GM':self.gm[idx]}
                else:
                    tissue_seg = None

                yield mrs_out,idx,tissue_seg
    
    def get_indicies_in_order(self,mask=True):
        """Return a list of iteration indicies in order""" 
        out = []
        shape = self.data.shape
        for idx in np.ndindex(shape[:3]):
            if mask:
                if self.mask[idx]:
                    out.append(idx)
            else:
                out.append(idx)
        return out

    def get_scalings_in_order(self,mask=True):
        """Return a list of MRS object scalings in order""" 
        if self._store_scalings is None:
            raise ValueError('Fetch mrs by iterable first.')
        else:
            return self._store_scalings

    def mrs_by_index(self,index):
        mrs_out = MRS(FID=self.data[index[0],index[1],index[2],:],
                                header=self.header,
                                basis=self.basis,
                                names=self.names,
                                basis_hdr=self.basis_hdr,
                                H2O=self.H2O[index[0],index[1],index[2],:])
        self._process_mrs(mrs_out)
        return mrs_out


    def mrs_from_average(self):
        FID = misc.volume_to_list(self.data,self.mask)
        H2O = misc.volume_to_list(self.H2O,self.mask)
        FID = sum(FID)/len(FID)
        H2O = sum(H2O)/len(H2O)
        
        mrs_out = MRS(FID=FID,
                      header=self.header,
                      basis=self.basis,
                      names=self.names,
                      basis_hdr=self.basis_hdr,
                      H2O=H2O)
        self._process_mrs(mrs_out)
        return mrs_out

    
    def seg_by_index(self,index):
        if self.tissue_seg_loaded:
            return {'CSF':self.csf[index],'WM':self.wm[index],'GM':self.gm[index]}
        else:
            raise ValueError('Load tissue segmentation first.')

    def _process_mrs(self,mrs):

        if self.basis is not None:
            if self.conj_basis:
                mrs.conj_Basis()
            elif self.no_conj_basis:
                pass
            else:
                mrs.check_Basis(repair=True)
            
            mrs.keep(self.keep)
            mrs.ignore(self.ignore)

        if self.conj_FID:
            mrs.conj_FID()
        elif self.no_conj_FID:
            pass
        else:
            mrs.check_FID(repair=True)

        if self.rescale:
            mrs.rescaleForFitting(ind_scaling=self.ind_scaling)      
                
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
                mrs = self.mrs_by_index([i,j,k])
                ax.plot(mrs.getAxes(ppmlim=ppmlim),np.real(mrs.get_spec(ppmlim=ppmlim)))
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
            plt.show()        
    
    def __str__(self):
        return f'MRSI with shape {self.data.shape}\nNumber of voxels = {self.num_voxels}\nNumber of masked voxels = {self.num_masked_voxels}'
    def __repr__(self):
        return str(self)
        
    def set_mask(self,mask):
        """ Load mask as numpy array."""
        if mask is None:
            mask = np.full(self.data.shape,True)
        elif mask.shape[0:3]==self.data.shape[0:3]:
            mask = mask!=0.0
        else:
            raise ValueError(f'Mask must be None or numpy array of the same shape as FID. Mask {mask.shape[0:3]}, FID {self.data.shape[0:3]}.')
        
        self.mask = mask
        self.num_masked_voxels = np.sum(self.mask)
    
    def set_tissue_seg(self,csf,wm,gm):
        """ Load tissue segmentation as numpy arrays."""
        if (csf.shape != self.spatial_shape) or (wm.shape != self.spatial_shape) or (gm.shape != self.spatial_shape):
            raise ValueError(f'Tissue segmentation arrays have wrong shape (CSF:{csf.shape}, GM:{gm.shape}, WM:{wm.shape}). Must match FID ({self.spatial_shape}).')

        self.csf = csf
        self.wm = wm
        self.gm = gm
        self.tissue_seg_loaded = True

    def write_output(self,data_list,file_path_name,indicies=None,cleanup=True,dtype=float):

        if indicies==None:
            indicies = self.get_indicies_in_order()

        nt       = data_list[0].size
        if nt>1:
            data     = np.zeros(self.spatial_shape+(nt,),dtype=dtype)
        else:
            data     = np.zeros(self.spatial_shape,dtype=dtype)

        for d,ind in zip(data_list,indicies):
            data[ind] = d
        
        if cleanup:
            data[np.isnan(data)] = 0
            data[np.isinf(data)] = 0
            data[data<1e-10]     = 0
            data[data>1e10]      = 0
        
        if nt == self.FID_points:
            saveNIFTI(file_path_name, data, self.header)
        else:            
            img      = nib.Nifti1Image(data,self.header['nifti'].affine)
            nib.save(img, file_path_name)

    @classmethod
    def from_files(cls,data_file,mask_file=None,basis_file=None,H2O_file=None,csf_file=None,gm_file=None,wm_file=None):
        
        data,hdr = mrs_io.read_FID(data_file)
        if mask_file is not None:
            mask,_ = mrs_io.fsl_io.readNIFTI(mask_file)
        else:
            mask = None
        
        if basis_file is not None:
            basis,names,basisHdr = mrs_io.read_basis(basis_file)
        else:
            basis,names,basisHdr = None,None,[None,]
            
        if H2O_file is not None:
            data_w,hdr_w = mrs_io.read_FID(H2O_file)
        else:
            data_w = None

        out = cls(data,hdr,mask=mask,basis=basis,names=names,basis_hdr=basisHdr[0],H2O=data_w)

        if (csf_file is not None) and (gm_file is not None) and (wm_file is not None):
            csf,_ = mrs_io.fsl_io.readNIFTI(csf_file)
            gm,_ = mrs_io.fsl_io.readNIFTI(gm_file)
            wm,_ = mrs_io.fsl_io.readNIFTI(wm_file)
            out.set_tissue_seg(csf,wm,gm)                  
        
        return out
