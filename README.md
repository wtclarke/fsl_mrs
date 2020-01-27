# FSL-MRS


### Description

FSL-MRS is a collection of python modules and wrapper scripts for pre-processing and model fitting of Magnetic Resonance Spectroscopy (MRS) data.

---

### Installation 

    git clone https://git.fmrib.ox.ac.uk/saad/fsl_mrs.git
    cd fsl_mrs
    pip install .

---

### Content

#### Scripts:

- **fsl\_mrs**
: fit a single spectrum 
- **fsl\_mrsi**
: fit a 4D volume of spectra 
- **fsl\_mrs\_preproc**
: pre-processing (coil combination, averaging, eddy-current correction) 
- **fsl\_mrs\_sim**
: simulate basis
- **mrs_vis**
: quick visualisation of the spectrum

---

#### Usage

For each of the wrapper scripts above, simply type `<name_of_script> --help` to get the usage.


#### File types

FSL-MRS accepts FID data in NIFTI format. It can also read .RAW format (like LCModel). 

#### Working in python

If you don't want to use the wrapper scripts, you can use the python modules directly in your own python scripts/programs. Here are some examples below:

- Pre-processing
- Model fitting - single voxel
- Model fitting - MRSImaging







