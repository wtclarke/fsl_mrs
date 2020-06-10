.. _quick_start:

Quick Start Guide
=================

Summary
-------

1. Convert your data
~~~~~~~~~~~~~~~~~~~~
Before running FSL-MRS time domain data must be prepared in a complex 4D-NIFTI or ASCII based format.  The formats accepted are described below. The recommended format is NIfTI + json which can be created by running the accompanying spec2nii tool (see :ref:`Data Conversion <data_conversion>`).

Example conversion::

    spec2nii -f my_metab_file dicom -j metab.dcm

This will convert the dicom file (metab.dcm) to a NIfTI file named my_metab_file.nii and because the -j option was specified, a JSON file called my_metab_file.json.

For a list of supported formats see :ref:`Data Conversion <data_conversion>`.

2. Process your raw data
~~~~~~~~~~~~~~~~~~~~~~~~
Some data requires pre-processing. Often MRSI data will have gone through appropriate pre-processing during reconstruction, if so skip to step 3. For unprocessed single voxel (SVS) data, read on.

Use the :code:`fsl_mrs_proc` commands to pre-process your raw data. :code:`fsl_mrs_proc` contains routines for many common steps (e.g. coil combination, phase-frequency alignment, residual water removal). E.g.::

    fsl_mrs_proc -r --filename combined coilcombine --file my_metab_file*.nii.gz --reference my_wref_file.nii.gz 
    fsl_mrs_proc -r --filename aligned align --file combined*.nii.gz --ppm 1.8 3.5
    fsl_mrs_proc -r --filename avg average --file aligned*.nii.gz --avgfiles
    fsl_mrs_proc -r --filename water_removed remove --file avg.nii.gz
    fsl_mrs_proc -r --filename metab phase --file water_removed.nii.gz

The -r requests a HTML report to be generated for each stage of the processing. The different HTML reports can be merged using::

    merge_mrs_reports -d example_processing -o . *.html

If your data is unedited single voxel (SVS) try out the prepackaged processing pipeline :code:`fsl_mrs_preproc`. You will need to identify the water suppressed and water unsuppressed files to pass to the script.

::

    fsl_mrs_preproc --output my_subj --data metab*.nii.gz --reference wref*.nii.gz --report 

Have a look at the source code for :code:`fsl_mrs_preproc` to see how you can construct your own python script using the processing modules. You can always prototype using Jupyter/IPython (see :ref:`Demos <demos>`)

3. Create Basis Spectra
~~~~~~~~~~~~~~~~~~~~~~~
The fitting in FSL-MRS requires the user to provide basis spectra. Basis spectra are the simulated responses of the in vivo metabolites to the pulse sequence. FSL-MRS provides a simulator to create basis sets :code:`fsl_mrs_sim`::

    fsl_mrs_sim -b metabs.txt my_sequence_description.json

my_sequence_description.json contains a description of the sequence broken down into blocks of RF pulses and gradients. Much more information on constructing a suitable sequence description JSON file can be found on the :ref:`Basis Spectra Simulation <simulation>` page. 

Have a quick check of your basis set using mrs_vis::

    mrs_vis my_basis_spectra/

4. Tissue Segmentation
~~~~~~~~~~~~~~~~~~~~~~
For FSL-MRS to produce accurate water scaled molarity or molality concentrations from the fitting results, it must be provided with estimates of the tissue (GM, WM, CSF) fractions in each voxel.

For this FSL-MRS provides the *svs_segment* and :code:`mrsi_segment` commands.::

    svs_segment -t T1.nii.gz svs_data.nii.gz
    mrsi_segment -t T1.nii.gz mrsi_data.nii.gz

:code:`svs_segment` creates a small JSON file which can be passed to the fitting routines. :code:`mrsi_segment` creates NIfTI files of the fractional tissue volumes registered to the MRSI volume.
:code:`svs_segment` and :code:`mrsi_segment` both rely on `fsl_anat <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/fsl_anat>`_ to run FSL FAST tissue segmentation. If fsl_anat has already been run, then the  :code:`-t T1.nii.gz` option can be substituted with :code:`-a T1.anat`. 


5. Fitting
~~~~~~~~~~
FSL-MRS provides two wrapper scripts for fitting: :code:`fsl_mrs` (for SVS data) and :code:`fsl_mrsi` (for MRSI data).

::

    fsl_mrs  --data metab.nii.gz --basis my_basis_spectra --output example_fit --h2o wref.nii.gz --TE 11 --tissue_frac tissue_frac.json --report 

    fsl_mrsi --data mrsi.nii.gz  --basis my_basis_spectra --output example_fit --h2o wref.nii.gz --mask mask.nii.gz --TE 32 --tissue_frac WM.nii.gz GM.nii.gz CSF.nii.gz --report

6. Visualise
~~~~~~~~~~~~
HTML processing reports merged using :code:`merge_mrs_reports` and fitting reports made using :code:`fsl_mrs` and :code:`fsl_mrsi` can be viewed in your browser.

For visualising MRSI data, fits, and fitting results, `FSLeyes
<https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`_ is recommended. 


.. _demos:

Demos
-----
Two demo Jupyter notebooks are provided alongside some sample data in the example_usage directory. These notebooks show an example processing pipeline implemented both on the command-line and in interactive python. 

