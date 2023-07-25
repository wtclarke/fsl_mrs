Dynamic Fitting
===============

The dynamic fitting module :code:`fsl_mrs.dynamic` and command-line tool :code:`fsl_dynmrs` can simultaneously fit multiple spectra which are linked by an arbitrary model.

"Dynamic" MRS is defined as spectra acquired under changing experimental and/or acquisition conditions. For example functional (fMRS), diffusion weighted MRS (dwMRS), or edited MRS. The FSL-MRS dynamic fitting tools are suitable for fitting all of these types of data.

Requirements
~~~~~~~~~~~~

Three things are needed to use the dynamic fitting tools:

1. A 2D (or higher) dataset - This contains a spectral dimension and at least one other dimension across which spectra change.
2. A model configuration file - This specifies how each spectrum is related to the others by describing a dynamic model.
3. A description of the modulating input variables - For example, a file or files with time values or b-values / gradient directions.
4. One or more sets of basis spectra - Used to perform the linear combination fitting. Different spectra can be fitted using different basis sets.

To run the fitting use the :code:`fsl_dynmrs` interface. An example using the packaged dwMRS data (found in example_usage/example_data/example_dwmrs) is::

    fsl_dynmrs\
    --data metab.nii.gz\
    --basis basis\
    --dyn_config config.py\
    --time_variables bvals\
    --output dyn_results

The same fit, but run using the interactive python interface is demonstrated in the packaged :file:`Example Dynamic Fitting.ipynb`


Dataset
-------
The dynamic data should be formatted as `NIfTI-MRS <https://wtclarke.github.io/mrs_nifti_standard/>`_ with a single 'higher dimension' in use. I.e. the fifth dimension should contain the dynamically varying spectra (with the first three dimensions encoding spatial extent and the fourth, the spectral time domain.)

Requisite pre-processing (e.g. coil combination, phase & frequency alignment, etc.) should already have been carried out.

Shaping of the NIfTI-MRS file can be carried out using the :code:`mrs_tools` command-line tool.

Configuration file
------------------
Purpose
*******
The configuration file is a python-formatted (*.py*) file that describes how the parameters of the dynamic model, known here as *free parameters*, correspond to the parameters of spectral fitting (e.g., concentrations, lineshapes, phases etc.), known as *mapped parameters*.

The file contains the explicit mappings, any bounds on the free parameters, and finally a functional description of the dynamic model.

Sections
********
The file come in three parts:

:code:`Parameters` variable - :code:`dict`:
    For each spectral fitting parameter type (e.g. concentration, or line-shift `eps`) this dict parameter defines whether that parameter:

       - Takes a different (unconstrained) value for each b value - **variable**
       - Has a fixed value across all b values - **fixed**
       - Or is described by a function of the varying acquisition (time, b-value, etc.) - **dynamic**

:code:`Bounds` variable - :code:`dict`:
    This dictionary provides lower and upper bounds for free parameters. 

Dynamic models and gradients - function definitions:
    If a mapped parameter has been identified as `dynamic` then a functional relationship between the mapped parameter and the time variable and free parameters must be given.

    These relationships are described using python functions.
    Each function listed in the `Parameters` dict must be defined.
    In addition a function providing the gradient of that function must be defined with a name suffix :code:`_grad`.
    Custom initialisation functions can be defined for each dynamic function. These must return estimates of the free parameters and named with a suffix :code:`_init`.

Example Syntax
**************

.. code-block::
   :caption: Example Parameters dict

   Parameters = {
    'conc'     : {'dynamic': 'model_exp', 'params':['c_amp','c_adc']},
    'gamma'    : 'fixed',
    'eps'      : 'fixed',
    'Phi_0'    : 'variable',
    'Phi_1'    : 'fixed'
    }

Here we see a simple example of a :code:`Parameters` dict. It defines a (exponentially decaying) dynamic model for metabolite concentration mapped parameters. In doing so it requires a function :code:`model_exp`, and defines two free parameters :code:`c_amp, c_adc`. Zero-order phase :code:`Phi_0` is free to vary across all spectra, whilst all other parameters are fixed (including the not listed :code:`baseline`, which defaults to *fixed*.)

The above listing of :code:`model_exp` requires a definition and a definition of the gradient function.

.. code-block::
    :caption: Example function definitions

    from numpy import exp, asarray

    def model_exp(p, t):
        # p = [amp,adc]
        return p[0] * exp(-p[1] * t)

    def model_exp_grad(p, t):
        e1 = exp(-p[1] * t)
        g0 = e1
        g1 = -t * p[0] * e1
        return asarray([g0, g1], dtype=object)

We may also wish to place bounds on the new free parameters. Below we limit the metabolite amplitudes, decay time constants and the line broadening to positive (but otherwise unbounded) values.

.. code-block::
    :caption: Example free parameter bounds

    Bounds = {
    'c_amp'       : (0, None),
    'c_adc'       : (0, None),
    'gamma'       : (0,None)
    }

More complex models can be defined, with different dynamic models defined per-metabolite (for concentrations) or per-metabolite-group for line widths (:code:`sigma, gamma`) and shifts (:code:`eps`).

.. code-block::
   :caption: Example multi-model Parameters dict

    Parameters = {
    'conc' : {'other': {'dynamic': 'model_exp', 'params': ['c_amp', 'c_adc']},
              'Mac':   {'dynamic': 'model_lin', 'params': ['c_amp', 'c_slope']}
              'H2O':   'variable'}}

In the above we provide a nested dict entry for the metabolite concentrations :code:`conc` entry. This defines that all metabolites except water (H2O) and the macromolecules (Mac) should follow the above exponential decay model. Macromolecules follow a linear decay, and water is free to vary unconstrained by a particular model. Note the names of water and macromolecules would be linked to the specific basis spectra set. Additional function definitions for :code:`model_lin` and :code:`model_lin_grad` would be needed. All other parameters would take the default *fixed* profile.

Other requirements
------------------

Two further items are needed:

A set of basis spectra:
    In the standard FSL-MRS format (directory of :code:`.json` files). A different set of basis spectra can be used for each dynamically linked spectra, though all metabolites must appear in each set.

A file (or files) defining the dynamically changing variable(s):
    Each file contains a list of one dynamically varying acquisition parameters, one value per spectrum. These values are passed to the functions defined in the configuration file.


Command line Interface Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are detailed explanations of some of the optional arguments in the wrapper script. Type :code:`fsl_dynmrs --help` to get the full set of available options. 


:code:`--ppmlim LOW HIGH`         
    Only calculate the loss function within this ppm range.
:code:`--baseline_order`            
    Polynomial baseline order. Set to -1 to remove the baseline altogether.
:code:`--metab_groups`      
    Group metabolites into sub-groups that get their own lineshape parameters (shift and broadening). This can either be a list of integers (one per metabolite) from 0 to the max number of groups minus one. Or it could be a list of metabolites to be grouped. E.g. using the flag :code:`--metab_groups Mac NAA+NAAG+Cr` then the Mac spectrum will have its own group, the NAA, NAAG, and Cr will be in a different group, and all other metabolites in a 3rd group. Other possibilities are combine_all and separate_all, where metabs are combined into a single group or separated into distinct groups respectively.
:code:`--lorentzian`        
    By default the lineshape is a Voigt (lorentizian+gaussian). Use this flag to set to Lorentzian.
:code:`--report`        
    Generate an HTML report of the fitting.
:code:`--no_rescale`        
    Do not rescale the input data before fitting. By default all spectra are rescaled using a single scaling factor.
