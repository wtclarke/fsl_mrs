Introduction
============

FSL-MRS is a python-based command-line software tool for the fitting and quantification of proton magnetic resonance spectroscopy data. FSL-MRS handles both single voxel and multi-voxel (MRSI) datasets. FSL-MRS is part of FSL (FMRIB Software Library). 

Alongside the core fitting features FSL-MRS contains tools for pre-processing, basis spectra simulation and data handling and conversion. FSL-MRS is designed to interface with existing FSL tools for data manipulation, quantification (FSL FAST) and display (Fsleyes). 

FSL-MRS can be used purely from the command line for bulk scripting of MRS analysis of large datasets, interactively in IPython style notebooks, or in a customisable way by modifying the open-source python libraries. 

At the core of FSL-MRS is a fitting tool that uses the linear combination of simulated basis spectra to estimate the relative or absolute concentrations of metabolites measured by the MRS sequence. The tool has been developed for 3T and 7T MRS and MRSI of teh human brain, but a number of advanced options are available providing increased flexibility for varied uses. 

In keeping with FSL’s tradition of favouring Bayesian inference approaches, the tool calculates full posterior distributions of the fitted metabolite concentrations to estimate concentration covariances and uncertainties using the Metropolis-Hastings algorithm.  

FSL-MRS typically takes a few seconds to analyse a single spectrum, producing an interactive HTML analysis report. In addition, we provide example post processing pipelines for an end-to-end solution for MRS processing.

Citing FSL-MRS
--------------
If you use FSL-MRS please cite: 

    Clarke, Near, Emir, Jbabdi - ISMRM 2020 