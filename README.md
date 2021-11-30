# FSL-MRS


## Description

FSL-MRS is a collection of python modules and wrapper scripts for pre-processing and model fitting of Magnetic Resonance Spectroscopy (MRS) data.

---
## Installation 

### Conda package
The primary installation method is via _conda_. First you should install conda and creating  a suitable [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). For example, in the base conda environment execute: 

    conda create --name fsl_mrs -c conda-forge python=3.8

Then activate the environment:
    
    conda activate fsl_mrs


Finally install FSL-MRS and its dependencies from the FSL conda channel.

    conda install -c conda-forge -c defaults \
                  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
                  fsl_mrs

### Source code

To get the source code with the packaged example data, make sure [git-lfs](https://git-lfs.github.com/) is installed.

    git clone --recurse-submodules https://git.fmrib.ox.ac.uk/fsl/fsl_mrs.git
    cd fsl_mrs
    pip install .


After installation see the [quick start guide](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/quick_start.html).

---

## Content

### Scripts:

- **fsl\_mrs**
: fit a single voxel spectroscopy (SVS) spectrum 
- **fsl\_mrsi**
: fit a 3D volume of 1D spectra 
- **fsl\_mrs\_proc**
: pre-processing (coil combination, averaging, phase correction, outlier rejection, water suppression, eddy-current correction) 
- **fsl\_mrs\_preproc**
: Pre-packaged processing for non-edited SVS. 
- **fsl\_mrs\_sim**
: simulate basis spectra
- **mrs_tools**
: Collection of tools for NIfTI-MRS. Includes quick visualisation and information.
- **basis_tools**
: Collection of tools for manipulating basis sets.
- **svs_segment & mrsi_segment**
: Run tissue segmentation for SVS/MRSI from T1 image.
- **results_to_spectrum**
: Generate spectrum representation of a fit from *fsl_mrs* results.
---

## Documentation

Documentation can be found online on the [WIN open science website](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/).

For each of the wrapper scripts above, simply type `<name_of_script> --help` to get the usage.

Example command-line usage is demonstrated in the packaged [Jupyter Notebook](https://git.fmrib.ox.ac.uk/saad/fsl_mrs/-/blob/master/example_usage/Example%20SVS%20processing%20-%20command-line.ipynb.).

## Getting help
Please seek help via the [FSL JISC email list](mailto:FSL@JISCMAIL.AC.UK) or by submitting an issue on the [FSL-MRS Github mirror](https://github.com/wexeee/fsl_mrs/issues).

## File types

FSL-MRS accepts FID data in NIfTI-MRS format. Some scripts can also read .RAW (LCModel) and text (jMRUI).

Conversion to NIfTI-MRS is provided by [spec2nii](https://github.com/wexeee/spec2nii).

## Working in python

If you don't want to use the wrapper scripts, you can use the python modules directly in your own python scripts/programs. Or in an interactive Python environment (see example [notebook](https://git.fmrib.ox.ac.uk/saad/fsl_mrs/-/blob/master/example_usage/Example%20SVS%20processing%20-%20interactive%20notebook.ipynb)) 

---

## Permissions and citations

If you use FSL-MRS in your research please cite:

    Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package. Magnetic Resonance in Medicine 2021;85:2950–2964 doi: https://doi.org/10.1002/mrm.28630.


Please see the [LICENSE](https://git.fmrib.ox.ac.uk/saad/fsl_mrs/-/blob/master/LICENSE) file for licensing information.




