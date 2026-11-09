.. _data_conversion:

Data Conversion
===============
There is a plethora of spectroscopy data formats in existence. Many are vendor specific and proprietary. The fitting capabilities of FSL-MRS can be used with either NIfTI or an ASCII file format, but to access the full features of the post-processing tools data must be in the NIfTI-MRS format. To facilitate the conversion of data to this format FSL-MRS is distributed with an additional conversion tool :code:`spec2nii`. Spec2nii currently converts SVS and MRSI data to NIfTI-MRS from the following formats. 

=============== ================ ===== ===== =======================
 Format          File extension   SVS   CSI   Automatic orientation  
=============== ================ ===== ===== ======================= 
 Siemens Twix    .dat             Yes   †     Yes                    
 Siemens DICOM   .ima / .dcm      Yes   Yes   Yes                    
 Siemens RDA     .rda             Yes   Yes   Yes                    
 Philips         .SPAR/.SDAT      Yes   No    Yes                    
 Philips         .data/.list      Yes   No    Yes                    
 Philips DICOM   .dcm             Yes   No    Yes                    
 GE              .7 (pfile)       Yes   Yes   Yes (WIP)              
 UIH DICOM       .dcm             Yes   Yes   Yes                    
 Bruker          2dseq            Yes   Yes   Yes                    
 Bruker          fid              Yes   Yes   Yes (WIP)              
 Varian          fid              Yes   No    No (WIP)               
 LCModel         .RAW             Yes   No    No                     
 jMRUI           .txt             Yes   No    No                     
 jMRUI           .mrui            Yes   No    No                     
 ASCII           .txt             Yes   No    No                     
=============== ================ ===== ===== =======================
† Partial handling

The authors of the tool are happy to provide additional conversion routines if sample data and a thorough description of the format is provided. 
Please see the spec2nii `project page on Github <https://github.com/wexeee/spec2nii>`_.

Use of spec2nii 
---------------

The specific sub command (e.g. `twix` or `dicom`) must be the first argument.

To specify the creation of a JSON metadata file alongside a NIfTI file (recommended) use the :code:`-j` option with any other inputs. 

File names can be specified with the :code:`-f` option and output directories with the :code:`-o` option.  

Twix
~~~~
Spec2nii can be used to inspect the contents of a twix file before conversion.
::

    spec2nii twix -v FILE

This will produce a list of the data with associated ‘evalinfo’ flags and data dimensions.

Spec2nii can then be run to convert all data from a single ‘evalinfo’ flag.  
::

    spec2nii twix -e FLAG FILE

‘Image’ is the most typically used for the main dataset. Other flags might be used for noise data, water reference data or any other use specified by the sequence programmer. 

Twix format loop variables (e.g. `Ave` or `ida`) can be assigned to specific NIfTI-MRS dimensions using the :code:`-d{5,6,7}` command line options. NIfTI MRS dimension tags (e.g. `DIM_COIL`) can be specified using the :code:`-t{5,6,7}` command line options.

Siemens DICOM
~~~~~~~~~~~~~
Spec2nii can be passed a single file or directory of DICOM files for conversion. 
::

    spec2nii dicom DCM_FILE_or_DIR

NIfTI-MRS dimension tags (e.g. `DIM_COIL`) can be specified using the `-t` command line argument.

Siemens RDA
~~~~~~~~~~~
Spec2nii can be passed a single RDA file for conversion. 
::

    spec2nii rda RDA_FILE

Validated to be the same data and orientation information as DICOM output on VE baselines.

UIH DICOM
~~~~~~~~~
Conversion for UIH DICOM format SVS and CSI spectroscopy.
::

    spec2nii uih DCM_FILE_or_DIR

NIfTI MRS dimension tags (e.g. `DIM_COIL`) can be specified using the `-t` command line argument.

GE
~~
Conversion for GE pfiles (.7).
::

    spec2nii ge FILE

Philips (SPAR/SDAT)
~~~~~~~~~~~~~~~~~~~
Conversion for Philips SDAT/SPAR files.
::

    spec2nii philips SDAT_FILE SPAR_FILE

Optional arguments:

-t/--tags allows the user to specify the dimension tags for each of the higher dimensions (up to three).
-s/--shape allows the user to perform Numpy style reshaping of multiple transients. By default (without specifying a shape) all transients will be listed in a single 5th dimension.

Philips (data/list)
~~~~~~~~~~~~~~~~~~~
Conversion for Philips data/list files. Must be provided along side a matching SPAR file.
::

    spec2nii philips_dl DATA_FILE LIST_FILE SPAR_FILE

Philips DICOM
~~~~~~~~~~~~~
Spec2nii can be passed a single file or directory of DICOM files for conversion. 
::
    spec2nii philips_dcm DCM_FILE_or_DIR

NIfTI MRS dimension tags (e.g. DIM_COIL) can be specified using the -t command line argument.

Generates separate reference file if present.

Both classic and enhanced DICOM is handled. Well tested on the vendors' own PRESS and MEGA-PRESS sequence.

Bruker (2dseq/fid)
~~~~~~~~~~~~~~~~~~
Spec2nii can be passed a single file or directory of 2dseq or fid files for conversion. 
::
    spec2nii bruker -m 2DSEQ 2DSEQ_FILE_or_DIR
    spec2nii bruker -m FID FID_FILE_or_DIR

Use the --inspect FOLDER option to run a textual App and navigate through a folder's scans and available data formats.

Use the -d option to dump the header files (method and acqp for fid, visu_pars for 2dseq) into the header extension.

Additional filters can be added by defining additional queries using the -q flag.

Varian
~~~~~~
Spec2nii can be passed a Varian fid directory containing a fid and procpar file for conversion. 
::
    spec2nii varian FILE

Use the -d option to dump the procpar header file contents into the header extension. 
Use the -t option to set an alternative dimension tag for the 6th dimension (default = DIM_DYN).

Plain text format
~~~~~~~~~~~~~~~~~
Conversion for text files containing one FID specified as two columns of data (real and imaginary). Required inputs are central frequency (-i, --imagingfreq; MHz), bandwidth (-b, --bandwidth; Hz). Optionally specify location with a 4x4 affine matrix passed as a plain text file (-a, --affine) 
 
::

    spec2nii text -i imaging_freq -b bandwidth [-a AFFINE_FILE] FILE

LCModel RAW format
~~~~~~~~~~~~~~~~~~
Conversion for .RAW text files containing one FID specified as two columns of data (real and imaginary). Optionally specify location with a 4x4 affine matrix passed as a plain text file (-a, --affine) 

::

    spec2nii raw [-a AFFINE_FILE] FILE


jMRUI text format
~~~~~~~~~~~~~~~~~
Conversion for text files in the jMRUI format containing one FID specified as two columns of data (real and imaginary). Optionally specify location with a 4x4 affine matrix passed as a plain text file (-a, --affine) 
 
::

    spec2nii jmrui [-a AFFINE_FILE] FILE