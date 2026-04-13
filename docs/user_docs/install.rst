.. _install:

Installation Instructions
=========================

FSL-MRS should be installed via the main FSL installer (:ref:`Option 1 <mac-fsl-option>`) or using the conda package manager (:ref:`Option 2 <mac-conda-option>`). Building from source code is possible using the instructions in :ref:`Option 3 <mac-git-option>`. For installation on MS Windows machines please see :ref:`the specific setup instructions below <win-instructions>`.


.. _mac-fsl-option:

Option 1: FSL install script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The primary installation method is via the *FSL installer*. Follow the instructions on the main `FSL wiki installation page <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_. If the *fsl_mrs* version is not the latest, then run :code:`update_fsl_package fsl_mrs` to update it.

.. _mac-conda-option:

Option 2: Using Conda
~~~~~~~~~~~~~~~~~~~~~
The alternative installation method is via *conda*. After installing conda and creating or activating a suitable environment you can install FSL-MRS from the FSL conda channel.
See our page on setting up a :ref:`conda enviroment <conda>` for a step by step guide. Do not install FSL-MRS into the `base` environment (this is the environment that starts activated).

::

    conda install -c conda-forge \
                  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
                  fsl_mrs

Set *FSLDIR* and *FSLOUTPUTTYPE* environment variables within your activated environment - Do not set them in the `base` environment!
::

    export FSLDIR="${CONDA_PREFIX}"
    export FSLOUTPUTTYPE="NIFTI_GZ"

To check the version installed run :code:`fsl_mrs --version`. FSL-MRS can be updated by running :code:`conda update`:

::

    conda update -c conda-forge \
                  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
                  fsl_mrs

Example data with conda
-----------------------
Installation with conda is easy, but you won't get the packaged example data and notebooks. This can be downloaded separately here: |fslmrs_pkg_data_notebooks|_.

.. _mac-git-option:

Option 3: From GitLab
~~~~~~~~~~~~~~~~~~~~~
Download or clone from |fslmrs_gitlab|_. To get FSL-MRS with example data and example Jupyter notebooks, download the full package from gitlab. `Git LFS <https://git-lfs.github.com/>`_ must be installed to download package data.

::

    git clone --recurse-submodules https://git.fmrib.ox.ac.uk/fsl/fsl_mrs.git
    cd fsl_mrs
    conda install -c conda-forge -c defaults \
            -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
            --file requirements.txt
    pip install --no-deps .

To access all features (those which rely on other FSL tools, e.g. the :code:`svs_segment` scripts), you should either install FSL or run the following commands within your conda environment:
::

    conda install -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ -c conda-forge fsl-flirt fsl-flameo fsl-avwutils fsl-fugue
    export FSLDIR="${CONDA_PREFIX}"
    export FSLOUTPUTTYPE="NIFTI_GZ"

**Note**: You may need to re-activate your conda environment to access the CLI tools.


.. _win-instructions:

Windows Operating System
~~~~~~~~~~~~~~~~~~~~~~~~
FSL-MRS has been tested thoroughly on Mac and Linux operating systems, but it is limited tested on MS Windows. However there are three routes for using FSL-MRS on Windows.

Option A: Using WSL (recommended)
---------------------------------
The first option is to install FSL-MRS and the complete FSL package using `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (or WSL2). This offers an easy way of running a Linux environment on a Windows machine. To install the full FSL package in WSL, follow the `instructions online <https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/windows>`_.

Option B: Native windows FSL-MRS + FSL on WSL
---------------------------------------------
Alternatively, as of V1.1.13 of FSL-MRS the Python-only FSL-MRS package can be run in native Windows alongside a WSL FSL installation. This can be achieved as follows:

1. Enable WSL and install FSL into WSL as described in the `FSL install instructions <https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/windows>`_.

2. Add an :code:`FSLDIR` environment variable on the host Windows machine. This should be set to :code:`\\\\wsl$\<Distro>\home\<wsl-user>\fsl`, where <Distro> is the Linux distribution installed in WSL, and <wsl-user> is the WSL user account that you created when you set up WSL.
   a. In PowerShell this can be done for a single session with the command :code:`$env:FSLDIR = "\\\\wsl$\<Distro>\home\<wsl-user>\fsl"`.
   b. In Command Prompt this can be done for a single session with the command :code:`set FSLDIR="\\\\wsl$\<Distro>\home\<wsl-user>\fsl"`.
   c. To set it permanently, in either shell run the command :code:`setx FSLDIR "\\\\wsl$\<Distro>\home\<wsl-user>\fsl"`. Restart your terminal window for the variable to be accessible.

3. Install FSL-MRS on the native Windows machine by following the conda installation guide in :ref:`Option 2 <mac-conda-option>`.

**Note**: You may need to run FSL-MRS in a terminal with administrator rights as certain functions (e.g. symlink) might not have the required privileges to execute correctly.

Option C: Native windows FSL-MRS (partial function)
---------------------------------------------------
Install FSL-MRS on the native Windows machine by following the conda installation guide in :ref:`Option 2 <mac-conda-option>`. Certain features which rely on an FSL installation (e.g. the :code:`svs_segment` scripts) won't function.

**Note**: You may need to run FSL-MRS in a terminal with administrator rights as certain functions (e.g. symlink) might not have the required privileges to execute correctly.

Verifying the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Please run the packaged :code:`fsl_mrs_verify` script to confirm that installation has successfully completed.