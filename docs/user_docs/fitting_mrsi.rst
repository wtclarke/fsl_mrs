Fitting MRSI
============

FSL-MRS fitting is performed using a linear combination model where a spectral basis is shifted, 
broadened, and scaled to fit the FID in the spectral domain. 
Additional nuisance parameters are 0th and 1st order phase, as well as a polynomial or p-spline complex baseline.

Wrapper scripts for command-line fitting are provided for MRSI as shown below. 
For more details on the fitting model, algorithms, and advanced options see :ref:`Details <mrsi_details>`.

Command-line script
-------------------

A basic call to :code:`fsl_mrsi` is given below:

::

    fsl_mrsi --data mrsi.nii.gz \
             --basis my_basis_spectra \
             --output example_fit \
             --mask mask.nii.gz \
             --h2o wref.nii.gz \
             --tissue_frac WM.nii.gz GM.nii.gz CSF.nii.gz

This will fit the linear combination model to each voxel independently. Many additional options are available. Type :code:`fsl_mrsi --help` for a list of all options. 


Output
~~~~~~
Results from :code:`fsl_mrsi` are stored in a single folder containing the following output:

- An interactive HTML report showing the fit to the average FID across all voxels in the mask.
- NIfTI files summarising parameters, concentrations, and QC measures (one such file per metabolite)
- Model prediction in the time domain (NIfTI)
- Residuals (NIfTI)
- Fitted Baseline (NIfTI)
- A file-tree file (mrsi.tree) that contains the folder and file structure information.

The above outputs can be visualised in FSLeyes alongside the original data. See `instructions <https://open.oxcin.ox.ac.uk/pages/wclarke/fsleyes-plugin-mrs/mrsi_results.html>`_ on how to best load MRSI results.

Python & Interactive Interface
------------------------------

Fitting for MRSI data can also be run in an interactive Python environment with `fslpy`.

In an IPython or Jupyter Notebook environment, run the following (the example data resides in the main :code:`fsl_mrs` package folder):

.. code-block:: python

    from fsl.wrappers import fsl_mrsi

    fsl_mrsi(
        data='metab.nii.gz',
        basis='3T_slaser_32vespa_1250_wmm',
        output='fit',
        metab_groups=['MM09', 'MM12', 'MM14', 'MM17', 'MM21'],
        h2o='wref.nii.gz',
        TE='30',
        TR='2.0',
        mask='mask.nii.gz',
        tissue_frac=['mrsi_seg_wm.nii.gz',
                     'mrsi_seg_gm.nii.gz',
                     'mrsi_seg_csf.nii.gz'],
        output_correlations=True,
        overwrite=True,
        combine=['Cr', 'PCr'],
    )

.. _mrsi_details:

Details
-------

Modelling, Algorithms and Wrapper options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The main options for modelling, optimisation algorithms and wrapper optional arguments are common between `fsl_mrsi` and `fsl_mrs` commands. See :doc:`fitting_svs` for details.
  
.. _advanced-fsl-dynmrs:

Advanced `fsl_dynmrs` options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common options for dynamic fitting can be found in :doc:`dynamic`. Below are detailed explanations of some additional MRSI-specific options. 

:code:`--spatial-mask SPATIAL_MASK`
    A NIfTI binary mask for selecting which MRSI voxels to fit. If not provided, all voxels will be fitted.
:code:`--spatial-index X Y Z`
    Spatial index of a single MRSI voxel to fit. If not provided, all voxels will be fitted.
:code:`--mean_mrsi`
    If provided, the mean FID across MRSI voxels will also be fitted.

Parallel processing
~~~~~~~~~~~~~~~~~~~
Both `fsl_mrsi` and `fsl_dynmrs` use Dask for parallel processing across MRSI voxels. The following options are available for both commands.

:code:`--parallel PARALLEL`
    Control parallelisation. Set to: 'off', 'local' (default), or 'cluster'. 'off' forces serial processing, 'local' parallelises over local CPUs, 'cluster' distributes over HPC SLURM nodes. See documentation for cluster configuration.
:code:`--parallel-workers PARALLEL_WORKERS`
    Number of cores (local), or workers (cluster) to use.

References
----------

.. [CLAR21] `Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package. Magnetic Resonance in Medicine 2021;85:2950-2964 <https://doi.org/10.1002/mrm.28630>`_