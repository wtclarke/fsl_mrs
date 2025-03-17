.. _processing:
Processing
==========
**For SVS**


Processed SVS data comprises a time domain signal localised to a single spatial volume. This is Fourier transformed to give a frequency domain spectrum. In an un-processed state, the data might not be coil-combined, might have multiple transients needing averaging, and might have other acquisition loops requiring specific processing. Typically, SVS data requires no “reconstruction” per-se, but several steps must be followed to achieve the highest quality data possible from an acquisition. 

For a complete overview of pre-processing we recommend [NEAR20]_. In short, the data must be coil combined, processed to remove small frequency drifts and corrupted transients, averaged, corrected for eddy currents, and finally phased. Optionally, large residual water signals can be removed. This is summarised in the table below.

**For MRSI**

MRSI comprises an “image” of spectroscopy data, each voxel contains time or frequency domain data. MRSI data will require reconstruction from the raw (k-space) data collected by the scanner. This may be carried out either online (on the scanner) or offline. Typically, this reconstruction incorporates some of the steps described above for SVS data (e.g. coil combination or averaging). Other steps used for SVS processing would not be commonly used for MRSI data (e.g. bad average removal). However, the majority of the fsl_mrs_proc commands can be run on MRSI data stored in NIfTI format where the processing will be applied independently per-voxel. 

Due to the complexity and specialism of MRSI reconstruction FSL-MRS does not provide MRSI reconstruction tools. 
For preprocessing we advise that the data should be coil-combined and repetitions averaged before fitting.
In addition we provide a few specific preprocessing tools for MRSI: inter voxel alignment, phase correction, and lipid removal.


fsl_mrs_proc
------------

Processing in FSL-MRS is primarily accessed through the commandline program :code:`fsl_mrs_proc`. 

Subcommands
~~~~~~~~~~~
======================= ==============================================================
fsl_mrs_proc operation	 Description	
======================= ==============================================================
coilcombine	         Combine individual coils of receiver phased array.
average             	 Average FIDs, with optional complex weighting.	
align               	 Phase and frequency align FIDs using spectral registration.
align-diff	         Phase and frequency align sub-spectra for differencing.
ecc  	                 Eddy current correction using a water phase reference scan.
remove	                 Remove peak (typically residual water) using HLSVD.
model                    Model peaks with HLSVD.
tshift	                 shift/re-sample in time domain.	
truncate            	 Truncate/pad time-domain data by an integer number of points.	
apodize             	 Apply choice of apodization function to the data.	
fshift              	 Frequency domain shift.
unlike              	 Identify outlier FIDs and remove.	
phase               	 Zero-order phase spectrum by phase of maximum point in range.
fixed_phase              Apply fixed phase to spectrum	
subtract            	 Subtract two FIDs	
add                 	 Add two FIDs
conj                     Conjugate fids
mrsi-align               Phase and/or frequency align across voxels.
mrsi-lipid               Remove lipids from MRSI by L2 regularisation.
======================= ==============================================================

Specific help for each subcommand can be accessed using :code:`fsl_mrs_proc <subcmd> --help`


Generic commands
~~~~~~~~~~~~~~~~
:code:`fsl_mrs_proc` has a few arguments that are generic to each subcommand. These should still be specified after the subcommand argument.
::

    fsl_mrs_proc [subcmd] [specific arguments] [generic arguments]

Some common commands are:

- `--output` (required)    : Output directory
- `-r, --generateReports`  : Generate HTML report for this step
- `--filename`             : Output file name.

Merging processing HTML reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    merge_mrs_reports -d [description] -o [output folder] -f [report name] *.html

.. _fsl_mrs_preproc:
fsl_mrs_preproc
---------------

:code:`fsl_mrs_preproc` combines a number of processing steps to provide a one step processing of non-edited SVS data.
The script requires the user to provide unsuppressed data (--data), water reference data (--reference) and an output location (--output). The data can be coil combined or un-combined but must be consistent. 
::

    fsl_mrs_preproc --output my_subj --data metab*.nii.gz --reference wref*.nii.gz --report 

Optionally the user may specify any of the following:

- `--quant`: Data to be used for quantitation
- `--ecc`: Data to be used for eddy current correction
- `--remove-water`: Remove residual water using HLSVD.
- `--truncate-fid POINTS`: Truncate FID at start by POINTS. This can remove first-order phase.
- `--align_limits`: Select spectral window (in ppm) to run phase/frequency alignment over.
- `--align_window {N}`: Enables iterative alignment using windowed averages of N transients. Useful for low SNR data or if there are strong artefacts normally cancelled by averaging across phase cycles.
- `--noremoval`, `--noaverage`, `--noalign`, and `--fmrs`: Disables respective part of the processing pipeline. `--fmrs` sets --noremoval and --noaverage arguments.

The :code:`--ecc` option should be used to provide water reference data for eddy current correction. I.e. the data has experienced all gradients that the primary water suppressed data has. Conversely the :code:`--quant` option should be used to provide water reference data purely for final water reference scaling. The water reference data provided using the :code:`--reference` option will always be used for coil combination (if required) and if :code:`--quant` or :code:`--ecc` haven't been specified it will be used for quantification and ECC respectively.

:code:`fsl_mrs_preproc_edit` performs the same purpose for edited SVS data.


Python & Interactive Interface
------------------------------

To access the processing methods in either a python or interactive python enviroment load the `preproc` module
::

    from fsl_mrs.utils.preproc import nifti_mrs_proc

Reports and figures can be generated using the :code:`figure` and :code:`report` keyword arguments.

fsl_mrs_proc subcommand specifics
---------------------------------

1. coilcombine (Coil combination) 
        Takes a list of files (:code:`--file`) and runs a weighted SVD [RODG10]_ coil combination on them optionally using a single water reference dataset (:code:`--reference`) to calculate the complex weightings of each coil. The function expects data to be stored as 5D data, with the last dimension storing individual coil data. Each file is treated separately. Pre-whitening is carried out by supplying a covariance matrix (:code:`--covariance`), noise samples (:code:`--noise`), or estimated from the data (default). Pre-whitening can be disabled (:code:`--noprewhiten`). 

2. average (averaging) 
        Takes a file as input (:code:`--file`) and takes the mean across across a certain dimension (:code:`--dim`, either a NIfTI-MRS tag or dim index (5, 6, 7).

3. align (phase-frequency alignment) 
        Takes a list of files (:code:`--file`) and aligns each FID to the FID nearest to the mean, or to a single passed reference FID (:code:`--reference`). The ppm range can be defined (:code:`--ppm`, default = 0.2->4.2 ppm). 

4. ecc (eddy current correction) 
        Takes either a single file or list of files (:code:`--file`) and applies eddy current correction based on the phase of a water reference scan (:code:`--reference`, supplied either as a single reference or list of same length as :code:`--files`). The reference must have experienced the same eddy current effects (i.e. same gradients). 

5. remove (residual water removal - HLSVD) 
        Takes either a single file or list of files (:code:`--file`) and applies HLSVD peak removal ([LAUD02]_) over the specified ppm limits (:code:`--ppm`, default = 4.5->4.8 ppm) 

6. model (model peaks from data using HLSVD)
        Takes either a single file (:code:`--file`) and applies HLSVD to model peaks within a ppm range (:code:`--ppm`). Returns noiseless representation of these peaks as a NIfTI-MRS file.

7. tshift (time domain resampling) 
        Takes either a single file or list of files (:code:`--file`) and resamples in the time domain to achieve a different number of points (:code:`--samples`), and/or a different start time (:code:`--tshiftStart`, in ms), and/or a different end time (:code:`--tshiftEnd`, in ms). 

8. truncate (truncation or zero padding) 
        Takes either a single file or list of files (:code:`--file`) and adds or removes points (:code:`--points`, positive to add, negative to remove) from the start or end (:code:`--pos`, default end) of the FID. Points added are zeros. 

9. apodize (filtering of data) 
        Takes either a single file or list of files (:code:`--file`) and applies either an exponential or Lorentzian to Gaussian window (:code:`--filter`) to the time domain data. The window parameters may be specified (:code:`--amount`). 

10. fshift (frequency shift) 
        Takes either a single file or list of files (:code:`--file`) and shifts the data in the frequency domain by an amount specified in hertz (:code:`--shifthz`) or in ppm (:code:`--shiftppm`). 

11. unlike (bad average removal) 
        Takes a list of files (:code:`--file`) and returns files containing FIDS that are within N standard deviations (:code:`--sd`) from the median. The ppm range over which the spectra are compared can be set (:code:`--ppm`, default = 0.2->4.2 ppm) and the number of iterations of the algorithm can be controlled (:code:`--iter`). Optionally the FIDs which are identified as failing the criterion can be output (:code:`--outputbad`) 

12. phase (zero order phasing) 
        Takes either a single file or list of files (:code:`--file`) and applies zero-order phase to the FID/spectrum based on the phase at the maximum in a specified chemical shift range (:code:`--ppm`) 

13. fixed_phase (Apply fixed phase to spectrum)
        Applies a fixed phase to data, either zero order (in degrees) (:code:`--p0`), or first order (in seconds) (:code:`--p1`). Can be applied as a timeshift in the time domain or linear phase int he frequency domain :code:`--p1_type {shift,linphase}`.

14. subtract (Subtract two FIDs)
        Subtracts two elements in a :code:`--dim` of one :code:`--file`, or subtracts a :code:`--reference` file.

15. add (Add two FIDs)
        Adds elements in a :code:`--dim` of one :code:`--file`, or adds a :code:`--reference` file.

16. conj (Conjugate fids)
        Applied conjugation (reverses frequency flip) to :code:`--file`.

17. mrsi-align (Phase and/or frequency align across voxels)
        Can frequency align voxels :code:`--freq-align` and/or perform zero-order phase correction :code:`--phase-correct`. Phase correction can be limited to peaks in a range :code:`--ppm`. THe detected shifts and phases can be output into NIfTI files :code:`--save-params`.

18. mrsi-lipid (Remove lipids from MRSI by L2 regularisation)
        Uses a NIfTI :code:`--mask` file to identify lipid source voxels to remove lipids from other voxels using L2 regularisation method ([BILG13]_). :code:`--beta` must be adjusted for different cases.

References
----------

.. [NEAR20] `Near J et al. Preprocessing, analysis and quantification in single‐voxel magnetic resonance spectroscopy: experts' consensus recommendations. NMR in Biomed 2020.  <https://pubmed.ncbi.nlm.nih.gov/32084297>`_

.. [RODG10] `Rodgers CT, Robson MD. Receive array magnetic resonance spectroscopy: Whitened singular value decomposition (WSVD) gives optimal Bayesian solution. Magn Reson Med 2010. <https://pubmed.ncbi.nlm.nih.gov/20373389>`_

.. [LAUD02] `Laudadio T et al. Improved Lanczos Algorithms for Blackbox MRS Data Quantitation. J Magn Reson 2010. <https://pubmed.ncbi.nlm.nih.gov/12323148/>`_

.. [BILG13]  `Bilgic B et al. Fast image reconstruction with L2-regularization. jMRI 2013. <https://doi.org/10.1002/jmri.24365>_`