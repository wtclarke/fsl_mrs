.. _data_conversion:

Data Conversion
===============
There is a plethora of spectroscopy data formats in existence. Many are vendor specific and proprietary. The fitting capabilities of FSL-MRS can be used with either NIfTI or an ASCII file format, but to access the full features of the post-processing tools data must be in the NIfTI + JSON format. To facilitate the conversion of data to this format FSL-MRS is distributed with an additional conversion tool :code:`spec2nii`. Spec2nii currently converts SVS and MRSI data to NIfTI from the following formats. 

=============== ================ ===== ===== =======================
 Format          File extension   SVS   CSI   Automatic orientation  
=============== ================ ===== ===== ======================= 
 Siemens Twix    .dat             Yes   No    Yes                    
 Siemens DICOM   .ima / .dcm      Yes   Yes   Yes                    
 Philips         .SPAR/.SDAT      Yes   No    No                     
 GE              .7 (pfile)       Yes   No    No                     
 LCModel         .RAW             Yes   No    No                     
 jMRUI           .txt             Yes   No    No                     
 ASCII           .txt             Yes   No    No                     
=============== ================ ===== ===== =======================

The authors of the tool are happy to provide additional conversion routines if sample data and a thorough description of the format is provided. Please see the spec2nii `project page on Github <https://github.com/wexeee/spec2nii>`_.

Use of spec2nii 
---------------

To specify the creation of a JSON metadata file alongside a NIfTI file (recommended) use the :code:`-j` option with any other inputs. 

File names can be specified with the :code:`-f` option and output directories with the :code:`-o` option.  

Twix
~~~~

Spec2nii can be used to inspect the contents of a twix file before conversion.
::

    spec2nii twix -v <file>

This will produce a list of the data with associated ‘evalinfo’ flags and data dimensions.

Spec2nii can then be run to convert all data from a single ‘evalinfo’ flag.  
::

    spec2nii twix -e <flag> <file>

‘Image’ is the most typically used for the main dataset. Other flags might be used for noise data, water reference data or any other use specified by the sequence programmer. 

DICOM
~~~~~

Spec2nii can be passed a single file or directory of DICOM files for conversion. 
::

    spec2nii dicom <file_or_dir>
 
Philips
~~~~~~~
Conversion for Philips SDAT/SPAR files. Orientation information not validated.
::

    spec2nii philips <SDAT> <SPAR>

GE
~~
Conversion for GE pfiles (.7). Orientation information not validated.
::

    spec2nii GE <file>


Plain text format
~~~~~~~~~~~~~~~~~

Conversion for text files containing one FID specified as two columns of data (real and imaginary). Required inputs are central frequency (-i, --imagingfreq; MHz), bandwidth (-b, --bandwidth; Hz). Optionally specify location with a 4x4 affine matrix passed as a plain text file (-a, --affine) 
 
::

    spec2nii text -i <imaging_freq> -b <bandwidth> [-a <affine>]  <file>

LCModel RAW format
~~~~~~~~~~~~~~~~~~

Conversion for .RAW text files containing one FID specified as two columns of data (real and imaginary). Optionally specify location with a 4x4 affine matrix passed as a plain text file (-a, --affine) 

::

    spec2nii raw [-a <affine>] <file>


jMRUI text format
~~~~~~~~~~~~~~~~~

Conversion for text files in the jMRUI format containing one FID specified as two columns of data (real and imaginary). Optionally specify location with a 4x4 affine matrix passed as a plain text file (-a, --affine) 
 
::

    spec2nii jmrui [-a <affine>] <file>