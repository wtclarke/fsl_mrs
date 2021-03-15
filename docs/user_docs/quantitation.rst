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


Water Reference Scaling Model
-----------------------------
If provided with an unsuppressed water reference FSL-MRS can generate metabolite concentrations in units of molarity (moles of substance per unit-volume, mol/L), or molality (moles of substance per unit-weight, mol/kg). Scaling is performed alongside relaxation correction as described by equations 4 & 6 of [NEAR20]_. 

Referencing to water is carried out by comparing the integrated water resonance in the unsuppressed water (between 1.65 and 7.65 ppm) to the integrated area of a reference metabolite. The raw unsuppressed signal is first fitted using to a simple model (a single peak with Voigt lineshape), and integration is carried out on the fitted data after residual phase has been removed. This is to ensure the corruption of the first few FID points doesn't result in integration of broad, negative-valued wings of the water peak. Similarly the integration of the reference metabolite is carried out on the scaled, broadened basis with the influence of phase and baseline removed.

The integrated areas are shown in the final plot of the html report if a reference dataset is provided.

References
----------

.. [NEAR20] `Near J et al. Preprocessing, analysis and quantification in single‐voxel magnetic resonance spectroscopy: experts' consensus recommendations. NMR in Biomed 2020.  <https://pubmed.ncbi.nlm.nih.gov/32084297>`_