Manipulating MRS Data
=====================

Handling NIfTI-MRS
------------------

MRS data stored in NIfTI-MRS format can contain multiple higher dimensions. For example it might contain dimensions encoding multiple receive coils, multiple temporal averages, or even a spectral editing dimension.

Data might need to be manipulated within the NIfTI-MRS storage framework before, after, or during preprocessing. For this, FLS-MRS provides the :code:`mrs_tools` command line script. :code:`mrs_tools` has the ability to merge and split NIfTI-MRS files along the higher encoding dimensions. It can also reorder the higher dimensions, or create a new singleton dimension for further manipulation.

:code:`mrs_tools` also contains the :code:`mrs_tools vis` and :code:`mrs_tools info` options to provide quick visualisation and information on the command line. See the :ref:`Visualisation <visualisation>` page for more information on :code:`mrs_tools vis/info`.

:code:`mrs_tools split` takes a single file and splits it along a specified dimension e.g. :code:`--dim DIM_DYN`, at a single point (:code:`--index 8`) or extracting multiple elements into a second file (:code:`--indices 8 9 10`).

:code:`mrs_tools merge` takes two or more files and merges them along a specified dimension e.g. :code:`--dim DIM_EDIT`.

:code:`mrs_tools reorder` permutes the dimensions of an existing NIfTI-MRS file. For example, the 5th through 7th dimensions can be changed from :code:`DIM_COIL, DIM_DYN, DIM_EDIT` to :code:`DIM_DYN, DIM_EDIT, DIM_COIL` using :code:`--dim_order DIM_DYN DIM_EDIT DIM_COIL`.