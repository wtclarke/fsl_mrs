Processing
==========
**For SVS**


Processed SVS data comprises a time domain signal localised to a single spatial volume. This is Fourier transformed to give a frequency domain spectrum. In an un-processed state, the data might not be coil-combined, might have multiple transients needing averaging and might have other acquisition loops requiring specific processing. Typically, SVS data requires no “reconstruction” per-se, but several steps must be followed to achieve the highest quality data possible from an acquisition. 

For a complete overview of pre-processing we recommend this `reference <https://onlinelibrary.wiley.com/doi/full/10.1002/nbm.4257>`_. In short, the data must be coil combined, processed to remove small frequency drifts and corrupted transients, averaged, corrected for eddy currents and phased. Optionally large residual water signals can be removed. This is summarised in the table below: 

**For MRSI**

MRSI comprises an “image” of spectroscopy data, each voxel contains time or frequency domain data. MRSI data will require reconstruction from the raw (k-space) data collected by the scanner. This may be carried out either online (on the scanner) or offline. Typically, this reconstruction incorporates some of the steps described above for SVS data (e.g. coil combination or averaging). Other steps used for SVS processing would not be commonly used for MRSI data (e.g. Bad average removal). However, the majority of the fsl_mrs_proc commands can be run on MRSI data stored in NIfTI format where the processing will be applied independently per-voxel. 

Due to the complexity and specialism of MRSI reconstruction FSL-MRS does not provide MRSI reconstruction. Nor do we advise application of pre-processing beyond that the data should be coil-combined and repetitions averaged before fitting. 


fsl_mrs_proc
------------

Processing in FSL-MRS is primarily accessed through the commandline program :code:`fsl_mrs_proc`.
:code:`fsl_mrs_proc` has a series of 

Subcommands
~~~~~~~~~~~
======================= ==============================================================
fsl_mrs_proc operation	 Description	
======================= ==============================================================
coilcombine	             Combine individual coils of receiver phased array.
average             	 Average FIDs, with optional complex weighting.	
align               	 Phase and frequency align FIDs using spectral registration.
align-diff	             Phase and frequency align sub-spectra for differencing.
ecc  	                 Eddy current correction using a water phase reference scan.
remove	                 Remove peak (typically residual water) using HLSVD.
tshift	                 shift/re-sample in time domain.	
truncate            	 Truncate/pad time-domain data by an integer number of points.	
apodize             	 Apply choice of apodization function to the data.	
fshift              	 Frequency domain shift.	
unlike              	 Identify outlier FIDs and remove.	
phase               	 Zero-order phase spectrum by phase of maximum point in range	
subtract            	 Subtract two FIDs	
add                 	 Add two FIDs	
======================= ==============================================================

Specific help for each subcommand can be accessed using :code:`fsl_mrs_proc [subcmd] --help`


Generic commands
~~~~~~~~~~~~~~~~
:code:`fsl_mrs_proc` has a few arguments that are generic to each subcommand. These should still be specified after the subcommand argument.
::

    fsl_mrs_proc [subcmd] [specific arguments] [generic arguments]

Some common commands are:

- `-output` (required)    : Output directory
- `-r, --generateReports` : Generate HTML report for this step
- `-filename`             : Output file name.

Merging processing HTML reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    merge_mrs_reports -d [description] -o [output folder] -f [report name] *.html

fsl_mrs_preproc
---------------

:code:`fsl_mrs_preproc` combines a number of processing steps to provide a one step processing of non-edited SVS data.
The script requires a list of transients to be averaged (--data), water reference data (--reference) and an output location (--output). The data can be coil combined or un-combined but must be consistent. 
::

    fsl_mrs_preproc --output my_subj --data metab*.nii.gz --reference wref*.nii.gz --report 

Optionally the user may specify any of the following:

- `--quant`: Data to be used for quantitation
- `--ecc`: Data to be used for eddy current correction
- `--hlsvd`: Apply HLSVD to remove residual water
- `--leftshift POINTS`: Truncate FID at start by POINTS.

Python & Interactive Interface
------------------------------

To access the processing methods in either a python or interactive python enviroment load the `preproc` module
::

    import fsl_mrs.utils.preproc

Reports can be generated using the associated [subcmd]_report functions in the preproc submodules.

fsl_mrs_proc subcommand specifics
---------------------------------

1. coilcombine (Coil combination) 
        Takes a list of files (--file) and runs wSVD coil combination on them optionally using a single water reference dataset (-r/--reference) to calculate the complex weightings of each coil. The function expects data to be stored as 5D data, with the last dimension storing individual coil data. Each file is treated separately. Pre-whitening can be disabled (--noprewhiten). 

2. average (averaging) 
        Takes either a single file or list of files (--file) and takes the mean across the list of files (--avgfiles) or across a certain dimension (--dim, indexes from 0). 

3. align (phase-frequency alignment) 
        Takes a list of files (--file) and aligns each FID to the FID nearest to the mean, or to a single passed reference FID (--reference). The ppm range can be defined (--ppm, default = 0.2->4.2 ppm). 

4. ecc (eddy current correction) 
        Takes either a single file or list of files (--file) and applies eddy current correction based on the phase of a water reference scan (--reference, supplied either as a single reference or list of same length as --files). The reference must have experienced the same eddy current effects (i.e. same gradients). 

5. remove (residual water removal - HLSVD) 
        Takes either a single file or list of files (--file) and applies HLSVD peak removal over the specified ppm limits (--ppm, default = 4.5->4.8 ppm) 

6. tshift (time domain resampling) 
        Takes either a single file or list of files (--file) and resamples in the time domain to achieve a different number of points (--samples), and/or a different start time (--tshiftStart, in ms), and/or a different end time (--tshiftEnd, in ms). 

7. truncate (truncation or zero padding) 
        Takes either a single file or list of files (--file) and adds or removes points (--points, positive to add, negative to remove) from the start or end (--pos, default end) of the FID. Points added are zeros. 

8. apodize (filtering of data) 
        Takes either a single file or list of files (--file) and applies either an exponential or Lorentzian to Gaussian window (--filter) to the time domain data. The window parameters may be specified (--amount). 

9. fshift (frequency shift) 
        Takes either a single file or list of files (--file) and shifts the data in the frequency domain by an amount specified in hertz (--shifthz) or in ppm (--shiftppm). 

10. unlike (bad average removal) 
        Takes a list of files (--file) and returns files containing FIDS that are within N standard deviations (--sd) from the median. The ppm range over which the spectra are compared can be set (--ppm, default = 0.2->4.2 ppm) and the number of iterations of the algorithm can be controlled (--iter). Optionally the FIDs which are identified as failing the criterion can be output (--outputbad) 

11. phase (zero order phasing) 
        Takes either a single file or list of files (--file) and applies zero-order phase to the FID/spectrum based on the phase at the maximum in a specified chemical shift range (--ppm) 