Fitting
=======

FSL-MRS fitting is performed using a linear combination model where a spectral basis is shifted, broadened, and scaled to fit the FID in the spectral domain. Additional nuisance parameters are 0th and 1st order phase, as well as a polynomial complex baseline.

Wrapper scripts for command-line fitting are provided for SVS and MRSI. 


SVS
---

A basic call to :code:`fsl_mrs`, the SVS wrapper script, is as follows:

::

    fsl_mrs --data metab.nii.gz \
            --basis my_basis_spectra \
            --output example_fit \
            --h2o wref.nii.gz \
            --TE 11 \
            --tissue_frac tissue_frac.json \
            --report 

This will run nonlinear optimisation using the Truncated Newton algorithm, as implemented in Scipy, and will produce an interactive HTML report including absolute quantification to the water reference and tissue fraction correction. The script offers many additional options, including MCMC fitting (more robust but slower). Type: :code:`fsl_mrs --help` to see all available options.


Output
~~~~~~
Results from :code:`fsl_mrs` are stored in a single folder that contains the following:

- Interactive HTML Report (if the :code:`--report` option was used.
- CSV files summarising the metabolite concentrations (and uncertainties), fitted parameters, and some QC measures.
- PNG files with summary of the fitting and (optionally) voxel location.



MRSI
----

A basic call to :code:`fsl_mrsi` is given below:

::

    fsl_mrsi --data mrsi.nii.gz \
             --basis my_basis_spectra \
             --output example_fit \
             --mask mask.nii.gz \
             --h2o wref.nii.gz \
             --TE 32 \
             --tissue_frac WM.nii.gz GM.nii.gz CSF.nii.gz

This will fit the linear combination model to each voxel independently. Many additional options are available. Type :code:`fsl_mrsi --help` for a list of all options. 


Output
~~~~~~
Results from :code:`fsl_mrsi` are stored in a single folder containing the following output:

- An interactive HTML report showing the fit to the average FID across all voxels in the mask.
- NIFTI files summarising parameters, concentrations, and QC measures (one such file per metabolite)
- Model prediction in the time domain (NIFTI)
- Residuals (NIFTI)
- Fitted Baseline (NIFTI)

The above NIFTI output can all be visualied in FSLeyes alongside the original data.

Python & Interactive Interface
------------------------------

FSL-MRS can also be used in an interactive Python environment. The following is an example fitting and visualisation of data that has already been processed (e.g. with :code:`fsl_mrs_proc`). 

In an IPython or Jupyter Notebook environment, run the follwing (the example data resides in the main :code:`fsl_mrs` package folder):

Loading and preparing the data:
::
    from fsl_mrs import MRS

    FID_file     = 'example_usage/example_data/metab.nii'
    basis_folder = 'example_usage/example_data/steam_11ms'    

    mrs = MRS()
    mrs.from_files(FID_file,basis_folder)
    mrs.processForFitting()

Fitting the model to the data:
::
    from fsl_mrs.utils import fitting
    results = fitting.fit_FSLModel(mrs)

Visualising the fit:
::
    from fsl_mrs.utils import plotting
    plotting.plotly_fit(mrs,results)



