# FSL-MRS


### Description

FSL-MRS is a collection of python modules and wrapper scripts for pre-processing and model fitting of Magnetic Resonance Spectroscopy (MRS) data.

---
### Installation 

    git clone https://git.fmrib.ox.ac.uk/saad/fsl_mrs.git
    cd fsl_mrs
    pip install .

#### Dependencies

    pip install spec2nii

After installation see the [quick start guide](https://users.fmrib.ox.ac.uk/~wclarke/fsl_mrs/userdoc/quick_start.html).

---

### Content

#### Scripts:

- **fsl\_mrs**
: fit a single spectrum 
- **fsl\_mrsi**
: fit a 3D volume of 1D spectra 
- **fsl\_mrs\_proc**
: pre-processing (coil combination, averaging, eddy-current correction) 
- **fsl\_mrs\_preproc**
: Pre-packaged processing for non-edited SVS. 
- **fsl\_mrs\_sim**
: simulate basis spectra
- **mrs_vis**
: quick visualisation of the spectra or basis spectra
- **svs_segment & mrsi_segment**
: Run tissue segmentation for SVS/MRSI from T1 image.

---

### Usage

For each of the wrapper scripts above, simply type `<name_of_script> --help` to get the usage.

Example command-line usage is demonstrated in the packaged [Jupyter Notebook](https://git.fmrib.ox.ac.uk/saad/fsl_mrs/-/blob/master/example_usage/Example%20SVS%20processing%20-%20command-line.ipynb.).

Documentation can be found online [here](https://users.fmrib.ox.ac.uk/~wclarke/fsl_mrs/userdoc/index.html).

### File types

FSL-MRS accepts FID data in NIfTI + JSON format. Some scripts can also read .RAW (LCModel) and text (jMRUI).

Conversion to NIfTI is provided by [spec2nii](https://github.com/wexeee/spec2nii).

### Working in python

If you don't want to use the wrapper scripts, you can use the python modules directly in your own python scripts/programs. Or in an interactive Python environment (see example [notebook](https://git.fmrib.ox.ac.uk/saad/fsl_mrs/-/blob/master/example_usage/Example%20SVS%20processing%20-%20interactive%20notebook.ipynb)) 

---

### Permissions and citations

If you use FSL-MRS in your research please cite:

Please see the LICENSE file for licensing information.




