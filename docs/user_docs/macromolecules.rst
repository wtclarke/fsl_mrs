.. _macromolecules:

Macromolecules
==============

It is recommended to use empirically derived macromolecular basis spectra for the fitting step. A measured MM spectrum described in JSON format can be added to the basis spectra by following the instructions on the :ref:`basis spectra simulation <sim_mm>` page. 

A collection of MM basis spectra is available for a variety of field strengths, species and anatomies at `MRSHub <https://mrshub.org/datasets_mm/>`_.

In situations where an empirically measured macromolecular spectrum is not available FSL-MRS includes methods for quickly generating synthetic MM basis spectra. For details see `Synthetic MM`_. Both methods should not be used simultaneously.

For an in depth discussion of the effects of MM basis spectra choice on fitting performance see [CUDA12]_ and [GIAP19]_.

Synthetic MM
~~~~~~~~~~~~
Synthetic MM basis spectra can be added to a basis set using :code:`basis_tools add_set --add_MM basis_in basis_out`. In the interactive environment the same can be achieved by calling the method :code:`add_default_MM_peaks` in a basis object (of type :code:`fsl_mrs.core.Basis`).

By default this option will add the following basis spectra (in separate metabolite groups) to the basis sets.

.. csv-table::
    :header: Peak name,	Peak location(s) (ppm),	Peak relative amplitude(s),	Peak broadening (gamma/sigma) 
    :widths: 10, 10, 10, 10

    MM09,	0.9,	3,	10/20
    MM12,	1.2,	2,	10/20
    MM14,	1.4,	2,	10/20
    MM17,	1.7,	2,	10/20
    MM21,	"2.08, 2.25, 1.95, 3.0",	"1.33, 0.22, 0.33, 0.4",	10/20

Additional peaks may be added in using :code:`basis_tools add_set --gamma ...`, or in the interactive environment by calling :code:`basis.add_MM_peaks`.

Note that when fitting using the synthetic macromolecular peaks the :code:`--metab_groups` option should be used to assign each synthetic MM peak to its own group. E.g., :code:`--metab_groups MM09 MM12 MM14 MM17 MM21`

References
~~~~~~~~~~

.. [CUDA12] Cudalbu C, Mlynárik V, Gruetter R. Handling Macromolecule Signals in the Quantification of the Neurochemical Profile. Journal of Alzheimer’s Disease 2012;31:S101–S115 doi: 10.3233/JAD-2012-120100.

.. [GIAP19] Giapitzakis I-A, Borbath T, Murali‐Manohar S, Avdievich N, Henning A. Investigation of the influence of macromolecules and spline baseline in the fitting model of human brain spectra at 9.4T. Magnetic Resonance in Medicine 2019;81:746–758 doi: 10.1002/mrm.27467.
