This document contains the FSL-MRS release history in reverse chronological order.

2.0.8 (WIP)
-----------
- Added check (and printed suggestion) in `fsl_mrs` and `fsl_mrsi` that default MM are added to appropriate metabolite groups.
- Added ppm range option to `fsl_mrs_preproc` alignment stage. Use option `--align_limits`.


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
