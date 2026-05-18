The NIfTI-MRS class
===================

FSL-MRS inherits and extends the :class:`NIFTI_MRS` from `nifti-mrs-tools`.

The parent class loads and represents NIfTI-MRS formatted data. It provides access to key acquisition and image parameters, along with useful utilities for reading and updating them:
- Resonant nucleus (`NIFTI_MRS.nucleus`)
- Spectrometer frequency (`NIFTI_MRS.spectrometer_frequency`)
- Dwell time (`NIFTI_MRS.dwelltime`)
- Bandwidth (`NIFTI_MRS.spectralwidth` or `NIFTI_MRS.bandwidth`)
- Field strength (`NIFTI_MRS.field_strength`)
- NIfTI header (`NIFTI_MRS.header`)
- MRS JSON header extension (`NIFTI_MRS.hdr_ext`)

The main extension in FSL-MRS is a method to generate an `MRS` object directly from a `NIfTI-MRS` object (`generate_mrs()` or `mrs()`, see :ref:`mrs-section` for more details).


Loading data
------------

Valid NIfTI-MRS files can be loaded like this:

.. code-block:: python

   from fsl_mrs.core.nifti_mrs import NIFTI_MRS

   nmrs = NIFTI_MRS('path/to/data.nii.gz')

The above command, takes a boolean argument `validate_on_creation`. If `True` (default), then the code will validate the header extension and `SpectralWidth` fields upon creation.

Independent of this option, the `NIFTI_MRS` constructor will test if the input file is a valid NIfTI-MRS:
1) The header contains a non-empty intent code.
2) The intent code is of the format `mrs_v<number>_<number>` and not older than 0.2 version.
3) The header contains a header extension with code '44'.

If the first two reasons cause a failure in loading the file as `NIfTI-MRS`, then you can use `spec2nii clean` command to update the file:

::

    spec2nii clean <file> [-f <output_file> -o <output_folder>]

If the header extension causes a failure, then you can read and write the file as follows:

.. code-block:: python

    from fsl_mrs.utils.mrs_io import fsl_io

    nmrs = fsl_io.readNIFTI('path/to/data.nii.gz')
    nmrs.save('path/to/fixed_data.nii.gz')


Axes class
----------

A NIfTI-MRS object can be used to generate an `Axes` object:

.. code-block:: python

    from fsl_mrs.core.nifti_mrs import NIFTI_MRS
    from nifti_mrs.axes import Axes

    nmrs = NIFTI_MRS('path/to/data.nii.gz')
    axes = Axes.from_nifti_mrs(nmrs,
                               SpecFreqChemShift=4.65
                               RxOffset=-3.0)


An `Axes` object stores information for plotting and key spectroscopy parameters, like:
- Resonant nucleus (`Axes.ResonantNucleus`)
- Spectrometer frequency in MHz (`Axes.SpectrometerFrequency`)
- Dwell time in seconds (`Axes.dwelltime`)
- Bandwidth (`Axes.SpectralWidth`)
- Chemical shift (`Axes.ppmshift`)
- Number of points (`Axes.npoints`)
- Time axis (`Axes.timeAxis`)
- Frequency axis (`Axes.frequencyAxis`)
- PPM axis with no shift (`Axes.ppmAxis`)
- PPM axis with shift (`Axes.ppmAxisShift`)

Further, the Axes object allows for direct access to indices per axis:
- Time indices (`Axes.timeIndices(limits)`)
- Frequency indices (`Axes.frequencyIndices(limits)`)
- PPM indices with no shift (`Axes.ppmIndices(limits)`)
- PPM indices with shift (`Axes.ppmShiftIndices(limits)`)

.. note::

    PPM axis and indices with no shift mean the values are zero-centred, whereas with shift mean the values are centred around the chemical shift value.


.. _mrs-section:

MRS and MRSI classes
--------------------

An `MRS` object is the basic unit for fitting and encapsulates a single spectrum, the basis spectra, and water reference information required to carry out fitting. Similarly, an `MRSI` object encapsulates the same information but for multiple voxels.

A NIfTI-MRS object can be used to generate an `MRS` or `MRSI` object. 

.. code-block:: python

    from fsl_mrs.core.nifti_mrs import NIFTI_MRS

    nmrs = NIFTI_MRS('path/to/data.nii.gz')
    mrs = nmrs.mrs()

The `NIFTI_MRS.mrs()` (and `NIFTI_MRS.generate_mrs()`) methods may also take `dim`, `basis_file`, `basis`, `ref_data`, `spatial_index`, `axes`, `chemShift`, and `RxOffset` arguments. See the code docstring for more details.

An `MRS` object can also be created from `FID` and `Axes` variables:

.. code-block:: python

    from fsl_mrs.core import MRS

    mrs = MRS.from_axes(fid, axes)

Once created, an `MRS` object gives you access to key properties and methods, like:
- Axes object (`MRS.axes`)
- FID (`MRS.FID`)
- Basis object (`MRS.basis`)
- Reference data (`MRS.H20`)
- Create spectrum within given ppm limits (`MRS.get_spec(ppmlim)`)
- Return axis within given limits (`MRS.getAxes('ppmshift', ppmlim)`)
- Process before fitting (`MRS.processForFitting()`)
- Fitting wrapper (`MRS.fit()`)

Similarly, an `MRSI` object gives you access to key properties and methods, like:
- Axes object (`MRSI.axes`)
- FID (`MRSI.data`)
- Basis object (`MRSI._basis`)
- Reference data (`MRSI.H20`)
- Average of all voxels (`MRSI.mrs_from_average()`)


-------------

:class:`NIFTI_MRS` class provides a plotting method for MR spectra:

.. code-block:: python

   nmrs = NIFTI_MRS('path/to/data.nii.gz')
   mrs = nmrs.plot()


This method may also take `display_dim` (Dim tag), `ppmlim` (ppm range), `plot_avg` (boolean), `mask` (filename), and `legend` (boolean) arguments.

.. note::

    For MRSI data, this is equivalent to calling `MRSI.plot()` method.

Additional plotting methods exist for an `MRS` object:
- `plot(ppmlim)`: plots the spectrum within the given ppm range (default: `ppmlim=None`)
- `plot_ref(ppmlim)`: plots the reference (i.e. H20) spectrum within the given ppm range
- `plot_fid(tlim)`: plots the time-domain data (FID) within the given time range
- `plot_basis(add_spec, ppmlim)`: plots the formatted basis along with the spectrum (if `add_spec=True`)

.. note::
    For SVS data, `MRS.plot()` is not equivalent to `NIFTI_MRS.plot()`, as the latter plots multiple spectra on the same axes.

.. note::
    NIfTI-MRS data can also be viewed in FSLeyes with `the MRS plugin <https://open.oxcin.ox.ac.uk/pages/wclarke/fsleyes-plugin-mrs/viewing.html>`_
