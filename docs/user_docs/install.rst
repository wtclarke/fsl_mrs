.. _install:

Installation Instructions
=========================

FSL-MRS can currently be installed using one of two methods.

Option 1: From GitLab
~~~~~~~~~~~~~~~~~~~~~
Download or clone from |fslmrs_gitlab|_. To get FSL-MRS with example data and example Jupyter notebooks, download the full package from gitlab. `Git LFS <https://git-lfs.github.com/>`_ must be installed to download package data.

::

    git clone --recurse-submodules https://git.fmrib.ox.ac.uk/fsl/fsl_mrs.git
    cd fsl_mrs
    pip install .

Option 2: From Conda
~~~~~~~~~~~~~~~~~~~~

The primary installation method is via *conda*. After installing conda and creating or activating a suitable you can install FSL-MRS from the FSL conda channel. See our page on setting up a :ref:`conda enviroment 
<conda>` for a step by step guide.

::

    conda install -c conda-forge \
                  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/channel/ \
                  fsl_mrs

Example data with conda
-----------------------
Installation with conda is easy, but you won't get the packaged example data and notebooks. This can be downloaded separately here: |fslmrs_pkg_data|_.


spec2nii
~~~~~~~~
To convert data to NIfTI install the spec2nii program from conda.

::

    conda install -c conda-forge spec2nii


Operating systems
~~~~~~~~~~~~~~~~~
FSL-MRS has been tested thoroughly on Mac and Linux operating systems. FSL-MRS dependencies and FSL-MRS is available on native Windows installations, but has not currently been tested. `Windows Subsytem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ (or WSL2) offers a Linux interface on Windows. FSL-MRS has been tested on WSL.