.. _conda:

:orphan:

===========
Conda Guide
===========

This is a short guide on setting up conda for the first time.

1. Download and install Miniconda (python version 3.9 or newer) from the `package website <https://docs.conda.io/en/latest/miniconda.html>`_.
2. Create a conda environment for installing fsl-mrs. Note the name you specifiy with `--name` is arbitrary and does not need to relate to the package name. 

::

    conda create --name fsl_mrs_env -c conda-forge python=3.12

3. Activate the environment.

::

    conda activate fsl_mrs_env

4. Optionally install JupyterLab to access notebooks. This is required to run the example Jupyter notebooks.

::

    conda install -c conda-forge jupyterlab

5. Follow the FSL-MRS & spec2nii install instructions on the :ref:`Installation 
<install>` page.


See the Conda `documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ for more information.