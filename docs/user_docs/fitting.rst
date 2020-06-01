Fitting
=======

SVS
---
::

    fsl_mrs --data metab.nii.gz --basis my_basis_spectra --output example_fit --algo MH --overwrite --report --h2o wref.nii.gz --TE 11 --tissue_frac tissue_frac.json

Output
~~~~~~
CSV files.


MRSI
----

::

    fsl_mrsi --data mrsi.nii.gz --basis my_basis_spectra --output example_fit --overwrite --mask mask.nii.gz --h2o wref.nii.gz --TE 32 --tissue_frac WM.nii.gz GM.nii.gz CSF.nii.gz

Output
~~~~~~
Results from *fsl_mrsi* are output as directories containing NIfTI files with the same spatial size as the original data.

Python & Interactive Interface
------------------------------
fit_FSLModel in the fsl_mrs.utils.fitting module