.. _install:

Installation Instructions
=========================

FSL-MRS should be installed using the conda package manager (option 1) or via the main FSL installer (option 2). Building from source code is possible using the instructions in option 3. For installation on MS Windows machines please see the specific setup instructions below.


Option 1: Using Conda
~~~~~~~~~~~~~~~~~~~~~

The primary installation method is via *conda*. After installing conda and creating or activating a suitable environment you can install FSL-MRS from the FSL conda channel. See our page on setting up a :ref:`conda enviroment 
<conda>` for a step by step guide. Do not install _FSL-MRS_ into the `base` environment (this is the environment that starts activated).

::

    conda install -c conda-forge \
                  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
                  fsl_mrs

To check the version installed run :code:`fsl_mrs --version`. FSL-MRS can be updated by running :code:`conda update`:

::

    conda update -c conda-forge \
                  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
                  fsl_mrs

Example data with conda
-----------------------
Installation with conda is easy, but you won't get the packaged example data and notebooks. This can be downloaded separately here: |fslmrs_pkg_data_notebooks|_.

Option 2: FSL install script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Follow the instructions on the main `FSL wiki installation page <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_.

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


Windows Operating System
~~~~~~~~~~~~~~~~~~~~~~~~
FSL-MRS has been tested thoroughly on Mac and Linux operating systems but is not currently tested on MS Windows. However there are three routes for using FSL-MRS on Windows.

Option A: Using WSL (recommended)
---------------------------------
The first option is to install (as above)  FSL-MRS and the complete FSL package using `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ (or WSL2). This offers an easy way of running a linux environment on a Windows machine. To install the full FSL package in WSL, follow the `instructions online <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows#Windows_Subsystem_for_Linux>`_

Option B: Native windows FSL-MRS + FSL on WSL
---------------------------------------------
Alternatively, as of V1.1.13 of FSL-MRS the python-only FSL-MRS package can be run in native Windows alongside a WSL FSL installation. This can be achieved as follows:

1. Enable WSL and install FSL into WSL as described in the `FSL install instructions <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows#Windows_Subsystem_for_Linux>`_.

2. Add an :code:`FSLDIR` enviroment variable on the host Windows machine. This should be set to :code:`\\\\wsl$\\usr\\local\\fsl` assuming the default install directory for FSL on the WSL guest. In Powershell this can be done with the command :code:`$env:FSLDIR = "\\\\wsl$\\usr\\local\\fsl"` to set it for a single session or :code:`[System.Environment]::SetEnvironmentVariable("FSLDIR", "\\wsl$\usr\local\fsl", [System.EnvironmentVariableTarget]::User)` to set it permanently.

3. Install FSL-MRS on the native Windows machine by following the conda installation guide in Option 1.

For FSL-MRS to access the FSL scripts installed on the WSL machine, it must be running.

Option C: Native windows FSL-MRS (partial function)
---------------------------------------------------
Install FSL-MRS on the native Windows machine by following the conda installation guide in Option 1. Certain features which rely on an FSL installation (e.g. the :code:`svs_segment` scripts) won't function.


Verifying the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Please run the packaged :code:`fsl_mrs_verify` script to confirm that installation has successfully completed.