Fitting
=======

FSL-MRS fitting is performed using a linear combination model where a spectral basis is shifted, broadened, and scaled to fit the FID in the spectral domain. Additional nuisance parameters are 0th and 1st order phase, as well as a polynomial complex baseline.

Wrapper scripts for command-line fitting are provided for SVS and MRSI as shown below. For more details on the fitting model, algorithms, and advanced options see :ref:`Details <details>`.


SVS
---

A basic call to :code:`fsl_mrs`, the SVS wrapper script, is as follows:

::

    fsl_mrs --data metab.nii.gz \
            --basis my_basis_spectra \
            --output example_fit

This will run nonlinear optimisation using the Truncated Newton algorithm, as implemented in Scipy, and will produce a simple PNG file summarising the fit, and several CSV files containing concentrations, uncertainties, and QC parameters for further analysis. 

A more complete call to :code:`fsl_mrs` may be as follows.

::

    fsl_mrs --data metab.nii.gz \
            --basis my_basis_spectra \
            --output example_fit \
            --h2o wref.nii.gz \
            --tissue_frac tissue_frac.json \
            --report 


This will additionally run absolute quantification w.r.t the water reference (with partial volume adjustments) and will produce an interactive HTML report. Type: :code:`fsl_mrs --help` to see all available options.

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


.. _details:

Details
-------

Modelling
~~~~~~~~~

At the core of FSL-MRS is a linear combination model. For more details on the modelling refer to [CLAR20]_. 

The signal in the spectral domain :math:`\mathrm{Y}(v)` is modelled as a linear combination of (shifted and broadened) metabolite basis spectra :math:`\mathrm{M}_{l,g}` (metab = :math:`l`, metab group = :math:`g`) plus a complex polynomial baseline :math:`\mathrm{B}(v)`. The signal model is as follows:

.. math::

    \begin{array}{c}
        \mathrm{Y}(v)=\mathrm{B}(v)+\exp \left[i\left(\phi_{0}+v \phi_{1}\right)\right] \sum_{g=1}^{N_{G}} \sum_{l=1}^{N_{g}} C_{l, g} M_{l, g}\left(v ; \gamma_{g}, \sigma_{g}, \epsilon_{g}\right) \\
        M_{l, g}\left(v ; \gamma_{g}, \epsilon_{g}\right)=\mathcal{FFT}\left\{m_{l, g}(t) \exp \left[-\left(\left(\gamma_{g}+\sigma_{g}^{2} t\right)+i \epsilon_{g}\right) t\right]\right\}
    \end{array}

Model parameters are summarised in the below table:

========================== ============================================================ ============
 Symbol                     Name                                                         Units  
========================== ============================================================ ============ 
 :math:`\phi_0`             zero-th order global phase                                    rad
 :math:`\phi_1`             first order global phase                                      rad/Hz
 :math:`\epsilon_g`         line shift for metab group :math:`g`                          rad/sec
 :math:`\gamma_g`           line broadening (Lorentizian) for metab group :math:`g`       Hz
 :math:`\sigma_g`           line broadening (Gaussian) for metab group :math:`g`          Hz
 :math:`\mathrm{C}_{l,g}`   concentration for metabolite :math:`l` in group :math:`g`     A.U.
========================== ============================================================ ============


Wrapper options
~~~~~~~~~~~~~~~

Below are detailed explanations of some of the optional arguments in the wrapper scripts. Type :code:`fsl_mrs --help` or :code:`fsl_mrsi --help` to get the full set of available options. 


:code:`--algo ALGO`         
    Algorithm to be used in the fitting. Either *Newton* (default) or *MH*. if *MH* is selected, the Metropolis hastings algorithm is run, initialised using the Newton algorithm (Truncated Newton as implemented in Scipy).
:code:`--ignore`            
    List of metabolites to be removed from the basis file prior to fitting.
:code:`--keep`              
    List of metabolites to include in the fitting, all other metabolites are excluded from the fitting
:code:`--combine`           
    Combine sets of metabolites (not in the fitting, only in the quantification/display) - this option is repeatable.
:code:`--ppmlim`            
    Only calculate the loss function within this ppm range.
:code:`--baseline_order`    
    Polynomial baseline order. Set to -1 to remove the baseline altogether.
:code:`--metab_groups`      
    Group metaboites into sub-groups that get their own lineshape parameters (shift and broadening). This can either be a list of integers (one per metabolite) from 0 to the max number of groups minus one. Or it could be a list of metabolites to be grouped. E.g. using the flag :code:`--metab_groups Mac NAA+NAAG+Cr` then the Mac spectrum will have its own group, the NAA, NAAG, and Cr will be in a different group, and all other metabolites in a 3rd group. Other possibilities are combine_all and separate_all, where metabs are combined into a single group or separated into distinct groups respectively.
:code:`--add_MM`            
    Add macromolecule peaks at the following frequencies: 0.9, 1.2, 1.4, 1.7 ppm and a doublet at 2.08 & 3.0 ppm
:code:`--lorentzian`        
    By default the lineshape is a Voigt (lorentizian+gaussian). Use this flag to set to Lorentzian.
:code:`--ind_scale`        
    Allow independent scaling of specified basis spectra before fitting. For example this can be used to independently scale empirically measured macromolecules combined with simulated metabolite spectra.
:code:`--disable_MH_priors`        
    Disable the priors on the MH fitting. The priors are tuned for *in vivo* human brain spectroscopy. Use this option if your spectra has significantly different line widths, phases or large shifts. E.g. in liquid phase phantom or (potentially) pre-clinical systems. Priors can be fine tuned by altering the values in :code:`fsl_mrs.utils.constants`.
:code:`--internal_ref`
    Set alternative metabolites for internal reference scaling (default is tCr = Cr + PCr). Multiple arguments can be specified for a combined internal reference.
:code:`--wref_metabolite`
    Set alternative water scaling reference (default is Cr). Must be used if none of Cr, PCr and NAA are present in the basis set.
:code:`--ref_protons`
    Number of protons that the water scaling reference is equivalent to (between defined integration limits). E.g. Cr is equivalent to 5 between 2 and 5 ppm. Only active when --wref_metabolite is used.
:code:`--ref_int_limits`
    Integration limits for water scaling reference. Only active when --wref_metabolite is used.

The wrapper scripts can also take a configuration file as an input. For example, say we have a text file called :code:`config.txt` which contains the below:

::

    # Any line beginning with this is ignored
    ppmlim       = [0.3,4.1]
    metab_groups = combine_all
    TE           = 11
    add_MM
    report

The the following calls to :code:`fsl_mrs` or :code:`fsl_mrsi` are equivalent:
::

    fsl_mrs --config config.txt

::

    fsl_mrs --ppmlim .3 4.1 --metab_groups combine_all --TE 11 --add_MM --report




References
----------

.. [CLAR20] Clarke WT, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package. Biorxiv 2020.