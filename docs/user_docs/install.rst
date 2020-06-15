Installation Instructions
=========================

FSL-MRS can currently be installed using two methods

1. From GitLab
~~~~~~~~~~~~~~
Download or clone from |fslmrs_gitlab|_. `Git LFS <https://git-lfs.github.com/>`_ must be installed to download package data.

::

    git clone --recurse-submodules https://git.fmrib.ox.ac.uk/saad/fsl_mrs.git
    cd fsl_mrs
    pip install .

2. From Conda
~~~~~~~~~~~~~

The primary installation method is via *conda*. After installing conda and creating or activating a suitable `enviroment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ you can install FSL-MRS from the FSL conda channel.

::

    conda install -c conda-forge \
                  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/channel/ \
                  fsl_mrs

spec2nii
~~~~~~~~
To convert data to NIfTI install the spec2nii program from conda.

::

    conda install -c conda-forge spec2nii
