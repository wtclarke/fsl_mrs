.. _conda:

:orphan:

===========
Conda Guide
===========

This is a short guide on setting up conda for the first time.

1. Download and install a python 3.7 version of Miniconda from the `package website <https://docs.conda.io/en/latest/miniconda.html>`_.
2. Create a conda environment.

::

    conda create --name fsl_mrs -c conda-forge python=3.7

3. Activate the environment.

::

    conda activate fsl_mrs

4. Optionally install JupyterLab to access notebooks. This is required to run the example Jupyter notebooks.

::

    conda install -c conda-forge jupyterlab

5. Follow the FSL-MRS & spec2nii install instructions on the :ref:`Installation 
<install>` page.


See the Conda `documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ for more information.