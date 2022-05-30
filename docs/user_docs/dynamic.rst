Dynamic Fitting
===============

The dynamic fitting module :code:`fsl_mrs.dynamic` and command-line tool :code:`fsl_dynmrs` can fit multiple spectra simultaneously, linked by an arbitrary model.

"Dynamic" MRS is classed as spectra acquired under changing experimental and/or acquisition conditions. For example functional (fMRS), diffusion weighted MRS (dwMRS), or edited MRS. The FSL-MRS dynamic fitting tools are suitable for fitting all of these types of data.

Requirements
~~~~~~~~~~~~

Three things are needed to use the dynamic fitting tools:

1. A 2D (or higher) dataset - This contains a spectral dimension and at least one other dimension across which spectra change.
2. A model configuration file - specifying how each spectra is related to the others by describing a dynamic model.
3. A description of the changing parameters - for example a file or files with time values or b-values / gradient directions.
4. One or more sets of basis spectra - Used to perform the linear combination fitting. Different spectra can be fitted using different basis sets.

To run the fitting use the :code:`fsl_dynmrs` interface. An example using the packaged dwMRS data is::

    fsl_dynmrs\
    --data dynamic_data.nii.gz\
    --basis basis\
    --config config.py\
    --time_variables \
    --output results

Using the interactive python interface is demonstrated in the packaged :file:`example_dynamic_fit.ipynb`


Dataset
-------

Configuration file
------------------
Purpose
Sections
Syntax of parameters and limits

Other requirements
------------------


Command line Interface Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
