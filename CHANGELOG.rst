This document contains the FSL-MRS release history in reverse chronological order.

2.3.3 (WIP)
---------------------------------
- Improved handling of trailing singleton dimensions of NIfTI-MRS data through various processing scripts.
- Fixed bug in unlike scan removal (e.g. `fsl_mrs_proc unlike`) when there was a header associated with the dimension being operated on.

2.3.2 (Monday 18th November 2024)
---------------------------------
- Fixed bug in `fsl_mrs_summarise` that caused existing data symlinks to not be found. Thanks to Sean Tan for reporting.

2.3.1 (Thursday 31st October 2024)
----------------------------------
- Recommended python version incremented to 3.12.
- Python testing and validation versions incremented to 3.12

2.3.0 (Friday 25th October 2024)
--------------------------------
- New baseline options - penalised splines have been added.
- In `fsl_mrs` and `fsl_mrsi`, baseline options should now be specified using the `--baseline` argument. Legacy `--baseline_order` is still available.
- New python interface api for fitting to accompany the new baseline options
- Dask now integrated for multi-voxel fitting across local cores and HPC clusters.
- Fix bug in coil combination with trailing singleton dimensions
- Fix bug for numpy versions < 2 calculating quantifiaction areas

2.2.0 (Tuesday 9th July 2024)
--------------------------------
- Increment minimum python version to 3.9, testing now takes place on python 3.11, matching main FSL.
- Numpy 2.0 compatibility

2.1.21 (Tuesday 16th April 2024)
--------------------------------
- Fix automatic deployment validation pipeline

2.1.20 (Monday 15th April 2024)
-------------------------------
- Minor improvements in MRSI plotting and preprocessing routines using average spectra.
- Added default relaxation values for 1.5 tesla.
- You can now specify custom T1 and T2 values to `fsl_mrs` using the `--t1-values` and `--t2-values`.
- The (t1/t2/tissue fraction) values used in the qunatification stage are now output as a `quantification_info.csv` file.
- `quantification_info.csv` can be passed to any of `--tissue_frac`, `--t1-values` and `--t2-values` as inputs.

2.1.19 (Wednesday 7th February 2024)
------------------------------------
- Multiple metabolites can now be specified as the water scaling metabolite
- Default water scaling metabolite is now PCr+Cr to add robustness to one of PCr or Cr not being fit.
- Improved error logging in `fsl_mrsi` when parallel processing enabled.

2.1.18 (Thursday 7th December 2023)
-----------------------------------
- Fixed bug in `fsl_mrs_summarise` that resulted in hang if single level of listed directories weren't distinguishable.
- NIfTI-MRS tools v1.1.1 required. Fixes some visualisation bugs.

2.1.17 (Tuesday 5th December 2023)
----------------------------------
- Fixed bug in windowed averaging introduced in v2.1.16

2.1.16 (Saturday 25th November 2023)
------------------------------------
- Better visualisation of dynamic fitting results.
- Added experimental windowed averaging to phase and frequency alignment. Improves results for low SNR data or that corrupted by phase cycled artifacts.

2.1.15 (Thursday 9th November 2023)
-----------------------------------
- No changes to FSL-MRS functionality, only under-the-hood fixes.
- Further (to `2.1.14`) changes to Docker CI images. Added build files and instructions to FSL-MRS repository.
- Fixes to version (versioneer) tracking. 

2.1.14 (Wednesday 8th November 2023)
------------------------------------
- Under the hood fixes to CI pipeline.

2.1.13 (Tuesday 5th August 2023)
--------------------------------
- Add group level f-tests to the `fmrs_stats` tool. First level f-contrasts are not yet implemented.
- Removed / retained indices after `fsl_mrs_proc unlike` (and related functions) are now listed in the NIfTI-MRS headers under the key "DIM_DYN Indices".

2.1.12 (Thursday 10th August 2023)
----------------------------------
- Implemented more testing of `fsl_mrs_proc` routines.
- Fixed further bugs in `fsl_mrs_proc unlike`.
- Fixed issues in the example notebooks distributed with FSL-MRS.

2.1.11 (Tuesday 8th August 2023)
--------------------------------
- Fixed bug in `fsl_mrs_proc unlike`.

2.1.10 (Tuesday 1st August 2023)
--------------------------------
- Coil covariance estimation is now common across all inputs to `fsl_mrs_preproc` and `fsl_mrs_preproc_edit`.
- Fallback option to disable coil pre-whitening when it isn't possible to calculate.

2.1.9 (Friday 28th July 2023)
-----------------------------
- Coil covariance estimation can now use multiple spectra held in higher NIfTI-MRS dimensions.

2.1.8 (Tuesday 25th July 2023)
------------------------------
- `fsl_dynmrs` can now handle MRSI data. This requires the use of `fsl_sub`, which is now a dependency.
- Custom initialisation functions can now be defined in dynamic fitting configuration files.

2.1.7 (Friday 7th July 2023)
----------------------------
- Added `fsl_mrs_proc mrsi-align` which can perform frequency and phase alignment across voxels using cross correlation.
- Added `fsl_mrs_proc mrsi-lipid` which can perform lipid removal using Bilgic et al's L2-regularised method.
- `fsl_mrs_proc fshift` can now take NIfTI images of matched shape witht he `--shifthz` anf `--shiftppm` to apply per-voxel shifts.
- Fixed bug with incorrectly calculated zero order phase when given in degrees.
- Improved interface and options for dynamic fitting driven preprocessing.
- Fixed minor bug in `fsl_mrs_summarise`
- Fixed bug where quantification information wasn't generated if no tissue fractions were given.
- Added warnings when quantification reference or water has zero integral.
- Moved to nifti-mrs 1.0.0 API

2.1.6 (Friday 5th May 2023)
---------------------------
- Add model performance outputs to dynamic fitting.
- Refined parameter-derived linewidth estimates of (default) voigt model. Previously the width of the Gaussian component was overestimated (by 50%), this did not affect per-metabolite QC measurement of FWHM reported in the main HTML remport and results CSV.
- fMRS contrasts formed from individual betas (e.g. using `fmrs_stats`) now operate over all parameter classes, not just concentrations (i.e. sigma, gamma, eps, baseline, phase, and concentrations).
- `fmrs_stats` can now be used to express metabolite concentrations as ratios to another contrast (e.g. for scaling to another metabolite, for example tCr).
- Added better help and error text for `fsl_dyn_mrs`.

2.1.5 (Wednesday 26th April 2023)
---------------------------------
- Improved speed of coil combination for MRSI
- Noise or covariance matrix may be supplied directly for pre-whitening in coil combination (`fsl_mrs_proc`, `fsl_mrs_preproc`, `fsl_mrs_preproc_edit`).
- More robust tests for coil combination.
- Fix 1D projection method for simulation of edited sequences

2.1.4 (Thursday 30th March 2023)
--------------------------------
- Improved `fsl_mrs_summarise` identification of dataset names.
- Fixed bug in `fsl_mrs_proc apodize`

2.1.3 (Wednesday 22nd March 2023)
---------------------------------
- Improved noise baseline de-trending in QC estimation.
- Resolved dash/flask dependency issues for python 3.7.

2.1.2 (Tuesday 21st March 2023)
-------------------------------
- Fixed bug in `fsl_mrs_summarise` showing some fits inverted along x.
- Updated installation instructions.
- Updated `fsl_mrs` report to include basis spectra (removed real/imag view).

2.1.1 (Monday 20th March 2023)
------------------------------
- Better handling of x-nuclei, including specific constants, ppm ranges, offsets etc.
- More appropriate xlabels for x-nuclei
- Fix issues around resolving symlinks, `fsl_mrs` results generate symlinks with appropriate extensions.
- Fixed plotting bug (depreciation of grid_b parameter in matplotlib).

2.1.0 (Thursday 19th January 2023)
----------------------------------
- FSL-MRS now uses the nifti-mrs tools python package. mrs_tools superseded by nifti-mrs implementation.
- Fixed bug in fmrs_stats design/contrast matrix specification
- Fixed bug in fmrs statistics module for metabolites with an underscore in their name.

2.0.9 (Wednesday 23rd November 2022)
------------------------------------
- `svs_segment` and `mrsi_segment` now normalise all voxel sums to 1.0
- QC now correctly estimates FWHM of inverted peaks.

2.0.8 (Monday 7th November 2022)
--------------------------------
- Added check (and printed suggestion) in `fsl_mrs` and `fsl_mrsi` that default MM are added to appropriate metabolite groups.
- Added ppm range option to `fsl_mrs_preproc` alignment stage. Use option `--align_limits`.
- Added new initialisation options to dynamic fitting based on free (rather than mapped) parameters.
- Truncation step in `fsl_mrs_preproc` now happens earlier in series.

2.0.7 (Thursday 20th October 2022)
-----------------------------------
- Added linear phase modulation to phasing processing tool, alternative to (interpolated) time domain time-shift
- Added more linear phase metrics to static fitting report.
- Refactored fitting models. 'Freeshift' model added to enable unconstrained frequency shifts.
- Added `basis_tools shift_all` command that can be used to tweak basis set based on a `freeshift` fit on high SNR data.
- Fix bug introduced in 2.0.5 in `fsl_mrs_preproc` where there was exactly one bad/good transient removed/kept.

2.0.6 (Monday 17th October 2022)
--------------------------------
- New CI validation stage that publishes to https://wtclarke.github.io/lcm_validation/index.html
- Fix bug in simulator code that stopped spatially resolved simulations running.

2.0.5 (Wednesday 5th October 2022)
----------------------------------
- Added specific `--fmrs` flag to fsl_mrs_preproc. This causes more sensible handling of data that isn't averaged e.g. not using data driven phase correction for each transient.
- `fsl_mrs_proc phase` and `fsl_mrs_proc fshift` now have the `--use_avg` to mirror the changes in `fsl_mrs_preproc`.
- Fixed plotting issue in `fsl_dynmrs` report.
- Updated Windows installation instructions.

2.0.4 (Wednesday 28th September 2022)
-------------------------------------
- fsl_mrs results now create symlinks to original data objects
- Updated command line interface for fsl_mrs_summarise, a list of results directories can now be passed.
- mrs_tools split better identifies which file contains which indices.
- Added fit and plot utility methods to mrs and results objects in python API.

2.0.3 (Wednesday 21st September 2022)
-------------------------------------
- Fixed bug in results / QC that only ran QC over default ppm region.
- Improved loading and ID of file types.
- Fixed bug in packaged example data loader.
- Improved textual help in fsl_mrs_preproc and fsl_mrs_proc for inappropriate data.

2.0.2 (Monday 1st August 2022)
------------------------------
- Handle jMRUI basis sets generated in VESPA which lack the SignalNames field.
- Add utility function parse_metab_groups to MRS class.

2.0.1 (Thursday 28th July 2022)
-------------------------------
- Fix results list generation from metropolis hastings dynamic optimisation
- Fix bug in fsl_mrs_summarise that would stop data with a disabled baseline (order = -1) working.
- Added conversion of jMRUI basis sets to basis_tools convert.
- --mask argument no longer mandatory for fsl_mrsi.
- Fixed bugs in mrsi_segment, auto run fsl_anat now work and qform copied to sform for MRSI in cases with small differences.

2.0.0 (Wednesday 6th July 2022)
-------------------------------
**Major rework of basis and fitting script interaction. First release of dynamic MRS fitting.**  

*Static fitting*  

- Default macromolecules are now added through basis_tools script rather than fitting. Fitting does not alter basis at run time now.
- Fixed bug in calculation of concentration covariances. New MC tests included.
- Better and faster covariance estimation via analytical jacobian.
- Update to QC SNR calculation to improve stability.

*Dynamic fitting*

- Saved dynamic results now contain free parameter covariances.
- New documentation for dynamic fitting
- New fmrs_stats module and script for higher-level GLM analysis.

*Other new features*  

- Experimental SVS results dashboard - view the results of multiple SVS fits together in a single summary.
- New documentation for dynamic fitting and all new features.
- Refactored imports to improve CLI startup times
- Conversion of LCModel raw formatted basis sets using basis_tools convert.

1.1.14 (Wednesday 29th June 2022)
---------------------------------
- Fixed variability in HLSVD by moving to Scipy dense svd.
- Fix for -ve ISHIFT in LCModel basis read. Also throws helpful error for encrypted basis.
- Fixed incorrect plotting of svs voxel orientation in fitting report.
- Fix issue in results_to_spectrum for disabled baseline.

1.1.13 (Wednesday 1st June 2022)
--------------------------------
- Updated setup script to allow command line scripts to run on MS Windows.
- Any FSL cmd-line scripts used operate through fslpy wrappers (including WSL interface).
- Updated install instructions for Windows.
- Added the fsl_mrs_verify script which can be run to verify correct function of FSL-MRS.

1.1.12 (Wednesday 20th April 2022)
----------------------------------
- Update to fslpy version (to 3.9.0) to substantially speed up MRSI preprocessing.
- Fixes to NIFTI_MRS class for compatibility with new fslpy version.
- Previous versions of FSL-MRS will not be compatible with fslpy >= 3.9.0

1.1.11 (Monday 4th April 2022)
------------------------------
- Now able to choose the number of workers in fsl_mrs_sim.
- Basis conversion now can remove reference peaks in a single step.
- Peak removal in basis set now defaults to zeroing rather than HLSVD for greater numerical stability. Mimics LCModel approach.
- Updates and corrections to documentation, references to new FSL Course MRS section added.
- Fixed bugs in LCModel basis set handling.
- Removed divide by zero warnings in quantification of voxels where fitting has failed.
- New outputs from fsl_mrsi script: parameter correlation matrices, group mappings and parameter names

1.1.10 (Thursday 12 January 2022)
---------------------------------
- Updates to fsl_mrs_preproc_edit
- Updated install documentation.
- Implemented new fft based interpolation of basis sets. Improves suppression of interpolation aliasing.

1.1.9 (Tuesday 30th November 2021)
----------------------------------
- Fixed typos in fsl_mrs_proc help.
- Fixed simulator bug for edited sequence coherence filters.
- Modified API of syntheticFromBasis function.
- Dynamic fitting now handles multiple different basis sets.
- Fix mapped parameter uncertainties in dynamic MRS results.
- Dynamic fitting results can now be saved to and loaded from a directory.
- Added MH sample option to fsl_mrs, matches interactive python interface.
- Changes to the dynamic fitting results API.
- Allow tissue fractions with larger errors, but normalise. Error thrown if sum < 0.9.
- Specialist phase & frequency alignment via dynamic fitting added.
- Added fsl_mrs_preproc_edit as a script for preprocessing edited data.
- Updated documentation with new install instructions.
- Updated interactive fitting documentation

1.1.8 (Tuesday 5th October 2021)
--------------------------------
- Fix bug in fsl_mrsi when default MM are added to a incorrectly conjugated basis set.
- Fix MRM reference in HTML report.

1.1.7 (Monday 4th October 2021)
-------------------------------
- Fixed commandline arguments for mrs_tools.
- mrs_tools now handles files with passed without extension.
- Fixed plotting orientation for preprocessing reports.
- CRLB are now reported in scaled absolute and percentage units.
- mrs_tools vis now handles DIM_COIL dimension appropriately with --display_dim command.
- Added a --no_mean command to mrs_tools vis to remove the average signal in multi dimensional data.

1.1.6 (Monday 20th September 2021)
----------------------------------
- Updates to dynamic MRS fitting in prep for 2021 dwMRS workshop.
- Dynamic MRS fitting beta: pending tests, documentation, and final features.

1.1.5 (Wednesday 11th August 2021)
----------------------------------
- Updated example MRSI data to conform to NIfTI-MRS standard.
- Quantification will not fail if volume fractions do not sum exactly to 1.0 (to within 1E-3).
- fixed bug in fsl_mrsi looking for TE in wrong header structure.
- New mrs_tools command 'conjugate' to help fix NIfTI-MRS data with the wrong phase/frequency convention.
- basis_tools remove has number of HLSVD components reduced to stop odd broad resonance behaviour.
- fsl_mrs_proc align can now align across all higher dimension FIDs. Pass 'all' as dimension tag.
- New command "fsl_mrs_proc model". HSLVD modelling of peaks in defined region. Number of components settable.
- Updates to basis set simulator. Non-uniform slice select gradients are now handled.

1.1.4 (Tuesday 3rd August 2021)
-------------------------------
- Fixed bug in calculation of molality concentration. Tissue mole fractions had been swapped for tissue volume fractions. Molar concentrations unaffected.
- Fixed bug in mrs_tools split
- Fixed bug in alignment of multi-dimensional data.
- Fixed bug in fsl_mrsi: data without a water reference now works.
- fsl_mrsi now outputs fitting nuisance parameters: phases, and shifts & linewidths for each metabolite group.
- Add NIfTI-MRS reshape command
- Add basis_tools remove_peak option to run HLSVD, typical usage for removing TMS peak.
- Added an add_water_peak method to MRS class.
- Updated fit_FSLModel defaults to match fsl_mrs command line defaults.

1.1.3 (Tuesday 29th June 2021)
------------------------------
- Added mrs_tools script. Replaces mrs_vis and mrs_info. Adds split/merge/reorder functionality.
- Added basis_tools script. Tools for manipulating (shifting, scaling, converting, differencing, conjugating, and adding to) basis sets.
- Improved display of basis sets using mrs_tools or basis_tools.
- Added 'default' MEGA-PRESS MM option to fsl_mrs and mrs class.
- Preprocessing tools now add processing provenance information to NIfTI-MRS files.
- Under the hood refactor of basis, MRS, and MRSI classes.
- Updated density matrix simulator. Added some automatic testing.
- Added documentation about the results_to_spectrum script.

1.1.2 (Friday 16th April 2021)
------------------------------
- Added 2H information
- Bug fixes
- Added documentation around installation from conda

1.1.1 (Monday 15th March 2021)
------------------------------
- SNR measurements should cope with negative peak amplitudes correctly
- New metabolites added to list of default water referencing metabolites (Cr, PCr and NAA)
- Quantification now takes into account T1 relaxation
- Quantification module now fits the water reference FID to deal with corruption of first FID points.
- Added plot in report to clarify referencing signals.
- Restructure of internal quantification code.

1.1.0 (Thursday 18th February 2021)
-----------------------------------
- Support for NIfTI-MRS format.
- Preprocessing scripts reoriented around NIfTI-MRS framework
- New script results_to_spectrum for generating full fits in NIfTI-MRS format from fsl_mrs results.
- Documentation and example data updated for move to NIfTI-MRS.
- Added mrs_info command to give quick text summary of NIfTI-MRS file contents.
- Updates to the WIP dynamic fitting module.

1.0.6 (Tuesday 12th January 2021)
---------------------------------
- Internal changes to core MRS class.
- New plotting functions added, utility functions for plotting added to MRS class.
- fsl_mrs/aux folder renamed for Windows compatibility.
- Moved online documentation to open.win.ox.ac.uk/pages/fsl/fsl_mrs/.
- Fixed small bugs in preprocessing display.
- Synthetic spectra now use fitting model directly.
- Bug fixes in the fsl_Mrs commandline interface. Thanks to Alex Craig-Craven.
- WIP: Dynamic fitting model and dynamic experiment simulation.
- spec2nii requirement pinned to 0.2.11 during NIfTI-MRS development.

1.0.5 (Friday 9th October 2020)
-------------------------------
- Extended documentation of hardcoded constants, including MCMC priors.
- Extended documentation of synthetic macromolecules.
- Added flag to MCMC optimise baseline parameters.

1.0.4 (Friday 14th August 2020)
-------------------------------
- Fixed bug in automatic conjugation facility of fsl_mrs_preproc
- jmrui text file reader now handles files with both FID and spectra

1.0.3 (Friday 10th July 2020)
-----------------------------
- Changed to pure python version of HLSVDPRO (hlsvdpropy). Slight speed penalty
  but hopefully reduced cross-compilation issues.
- fsl_mrs_preproc now outputs zipped NIFTI files to match the rest of the command-line   scripts.
- Apodisation option added to alignment in fsl_mrs_proc and fsl_mrs_preproc. Reduces effect of noise. Default value is 10 Hz of exponential apodisation.
- Fixed phasing subcommand added to fsl_mrs_proc allowing the user to apply a fixed 0th and 1st order phase.
- mrs_vis now handles folders as an input for MRS data (still handles folders of basis files).
- Conjugation command added to fsl_mrs_proc.
- fsl_mrs_preproc automatically conjugates input spectra if required.
- Typos and small bug fixes.
- Documentation expanded.

1.0.2 (Saturday 27th June 2020)
--------------------------------
- Add missing requirement (pillow)

1.0.1 (Friday 19th June 2020)
--------------------------------
- Output folder in fsl_mrs_proc will now be created if it does not exist.
- fsl_mrs_proc now handles data with a singleton coil dimension correctly.
- --ind_scale and --disable_MH_priors options added to fsl_mrs and fsl_mrsi.

1.0.0 (Wednesday 17th June 2020)
--------------------------------
- First public release of package.
