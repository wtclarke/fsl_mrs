Quantitation
============

Tissue Segmentation
-------------------
For FSL-MRS to produce accurate water scaled molarity or molality concentrations from the fitting results, it must be provided with estimates of the tissue (GM, WM, CSF) fractions in each voxel.

For this FSL-MRS provides the :code:`svs_segment` and :code:`mrsi_segment` commands.::

    svs_segment -t T1.nii.gz -f tissue_frac svs_data.nii.gz
    mrsi_segment -t T1.nii.gz -f tissue_frac mrsi_data.nii.gz

:code:`svs_segment` creates a small JSON file which can be passed to the fitting routines. :code:`mrsi_segment` creates NIfTI files of the fractional tissue volumes registered to the MRSI volume.
:code:`svs_segment` and :code:`mrsi_segment` both rely on fsl_anat to run FSL FAST tissue segmentation. If fsl_anat has already been run then the :code:`-t T1.nii.gz` can be substituted with :code:`-a T1.anat`. 