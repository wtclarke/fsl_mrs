.. _spatial:
Spatial Information
===================

Single voxel mask
-----------------
FSL-MRS tools provide an easy way to generate a mask of the SVS voxel in T1 space. For this FSL-MRS uses the :code:`svs_segment` command.::

    svs_segment -t T1.nii.gz --mask_only svs_data.nii.gz

