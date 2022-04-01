.. _simulation:

Basis Spectra Simulation
========================
The linear combination fitting method used by FSL-MRS requires the user to specify basis spectra. A basis spectrum must be supplied for each fitted metabolite. The basis spectra are specific to a sequence type, the precise sequence timings and RF pulses in the sequence. Each basis spectrum is effectively a “fingerprint” for a metabolite that will be scaled and manipulated simultaneously with all the other basis spectra during the fitting optimisation. Whilst the basis spectra can be generated from scanning phantoms, the recommended way is via simulation of the spectra using either a third-party software or FSL-MRS's own density matrix simulator. 

FSL-MRS's simulation software may be accessed through the :code:`fsl_mrs_sim` command line program. This section describes how to construct a description of your sequence, run the simulation and the format of the output basis spectra. Please see the dedicated simulation page for detailed information for the underlying simulation library. 

Getting started
---------------
Creation of basis spectra is a difficult step in the analysis of MRS data, with plenty of pitfalls even for experienced users. Please consult with local MRS experts or the technical community on the `MRSHub forums <https://forum.mrshub.org/>`_ for assistance and recommendations. The developers of FSL-MRS are aware that this area of the analysis pipeline remains a difficult stage and efforts are continuing to improve it for users. To that aim we have created two  resources for understanding basis set simulation in FSL-MRS:

1. The  section of the online |fsl_course|_ dedicated to MRS `basis set simulation <https://open.win.ox.ac.uk/pages/fslcourse/practicals/fsl_mrs/index.html#seq_description>`_.

2. The description in this documentation and the example python notebook `Example basis spectra creation.ipynb`, downloadable from this link: |fslmrs_pkg_data_notebooks|_.


Describing a sequence – the sequence file format 
------------------------------------------------

In FSL-MRS a sequence to be simulated is described in a JSON format file. The sequence description should comprise one transient (repetition time) of the sequence. The file breaks the sequence into a series of blocks that describe the preparation of the magnetization. Each block comprises an RF pulse, slice selection gradient, a delay with optional gradient rephasing/crushing and an optional coherence filter. The simulation does not simulate the read-out module of a sequence.

Phase cycling can in most cases be removed from the sequence and replaced by suitable coherence selection. If simulation of phase cycling is required, then summation of multiple runs of the simulator will be required with appropriate phase cycling gradients inserted into the delay period of each block. In a similar way edited sequences could be simulated. 

Coherence filters for some typical sequences are the following:  

========== ===============
Sequence    Filter 
========== ===============
STEAM      [1,0,-1] 
PRESS      [-1,1,-1] 
sLASER     [-1,1,-1,1,-1] 
MEGA‐PRESS [-1,1,1,-1,-1] 
========== ===============

For more information on coherence filters see this `reference <https://www.ncbi.nlm.nih.gov/pubmed/30390346>`_. The filter must end on –1. This is the only detected coherence in the simulator. 

For a description of the sequence file parameters see the :ref:`sequence file <seq_file_params>` page. Alternatively see the examples in the packaged in the `example_usage/example_data/example_sequences` directory (examplePRESS.json & exampleSTEAM.json).

Please note that delays in the sequence description file do not include the duration of the pulse. I.e., the time from the centroid to the start/end of the RF pulse must be added to the delay to calculate the time between pulse centroids.

Using fsl_mrs_sim 
-----------------

:code:`fsl_mrs_sim` provides a command line interface to the density matrix simulator in FSL-MRS. For examples on how to run the simulations in an interactive way see the API documentation (insert link here) and the packaged notebooks in the source code. 

On the command line the user must specify the sequence file to simulate, one of three metabolite options , any empirically measured macromolecule signal and the output folder (-o, --output). We recommend specifying an echo time (-e, --echotime; echo time in ms). 

Optionally: 
 - a 0 ppm reference can be added (:code:`-a, --addref`) 
 - Automatic phasing of the spectra can be achieved by simulating a singlet peak at a specified offset from the receiver centre (:code:`-p, --autophase`, offset in ppm). 
 - Different format outputs can be specified (:code:`--jmrui, --raw, --lcmIN`). 

Simulating basis spectra for j-difference editing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To simulate a basis spectrum for a j-difference edited sequence (e.g. MEGA-PRESS) simulate the basis for each condition and create the difference spectrum using the :code:`basis_tools diff` script.

Choosing metabolites
~~~~~~~~~~~~~~~~~~~~
The simulator is aware of the following metabolites. Spin systems are specified as in [SIMP17]_ and [GOVI00]_.  

=============================== =================== =========================== ===================
System                          Name in simulator   System                      Name in simulator 
=============================== =================== =========================== ===================
acetate                         Ace                 water                       H2O 
alanine                         Ala                 myo-inositol                Ins 
ascorbic acid                   Asc                 lactate                     Lac 
aspartic acid                   Asp                 N‐acetyl aspartate          NAA 
citrate                         Cit                 (NAA) glutamate             NAAG 
creatine                        Cr                  phosphocholine              PCh 
ethanol                         EtOH                phosphocreatine             PCr 
γ-aminobutyric acid\ :sup:`1` \ GABA                phosphorylethanolamine      PE 
γ-aminobutyric acid\ :sup:`2` \ GABA_gov            phenylalanine               Phenyl 
glycerophosphocholine           GPC                 scyllo-Inositol             Scyllo 
glutathione\ :sup:`2` \         GSH                 serine                      Ser 
glutathione\ :sup:`3` \         GSH_v2              taurine                     Tau 
glucose                         Glc                 tyrosine                    Tyros 
glutamine                       Gln                 beta-hydroxybutyrate        bHB 
glutamate                       Glu                 2-HG\ :sup:`4` \            bHG 
glycine                         Gly 
=============================== =================== =========================== ===================

- :sup:`1` Near et al. proc intl soc magn reson med 2012. p. 4386 
- :sup:`2` Corrigendum to Govindaraju et al. NMR Biomed . 2000; 13: 129–153
- :sup:`3` Tkac et al. proc intl soc magn reson med 2008: p1624 
- :sup:`4` Bal et al. Magnetic Resonance in Chemistry. 2002;40:533–36

Metabolites to simulate can be specified on the command line using the :code:`–m` option with a list typed on the command line, with the :code:`–b` option specifying a text file with one metabolite listed per line, or the :code:`–s` option pointing to a spin system json file for custom spin systems.

It is **not recommended** to simulate and use all of the metabolites. A typical list to start with for short echo time spectroscopy might be::

    Ala, Asp, GPC, PCh, Cr, PCr, GABA, Glc, Gln, Glu, GSH, Ins, Lac, NAA, NAAG, PE, Tau


.. _sim_mm:

Including macromolecules in your basis set 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An empirical description of the macromolecules can be included in the basis set by using the :code:`--MM` option. The macromolecules must be specified in a json format containing fields basis_re, basis_im, basis_dwell, basis_centre, basis_width and basis_name (see Output).

Example commands 
~~~~~~~~~~~~~~~~

To simulate the response of creatine, NAA and phosphocholine with the example TE = 11 ms STEAM sequence::

    fsl_mrs_sim -m Cr,NAA,PCh –o basis –e 11 exampleSTEAM.json 

To simulate the same using a batch metabolite list:: 

    fsl_mrs_sim -b mymetabs.txt –o basis –e 11 exampleSTEAM.json 

mymetabs.txt contains Cr,NAA,PCh each on a new line. 

Repeat the first example with automatic zero-order phasing. The automatic phasing simulates a peak at -2 ppm relative to the receiver set at 4.65 ppm (centralShift in table 1). 2.65 ppm is close to the centre of excitation.::

    fsl_mrs_sim -m Cr,NAA,PCh –o basis –e 11 -p -2.0 exampleSTEAM.json 

Speed of simulation 
~~~~~~~~~~~~~~~~~~~

The simulator takes advantage of the extended 1D projection method as implemented by [LAND19]_. The 1D projection method permits a dramatic time reduction during simulation but only where gradients occur in a non-reoccurring order e.g. x,y,z or x,y,y,z not x,y,z,y. In the latter case expect the simulation to take significantly longer. 

Where the 1D projection method can be used the simulation time will scale with the sum of the spatial resolution in each dimension. In other cases, the time will scale with the product.

If a large number of spatial points are specified (>30 in each dimension) then the simulation time for all metabolites can extend into a number of hours. We recommend ensuring that the sequence parameters are correctly defined using lower resolution simulations. 

Output – the basis spectra file format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`fsl_mrs_sim` will output a folder (location specified with :code:`-o`) containing a separate json file for each simulated metabolite. The json file contains a full description of:

- the simulated basis FID 
- the sequence used to simulate the basis. This allows the basis json to be used as a sequence file for new simulations. 
- meta-data 
- The density matrix state before readout. This allows regeneration of the basis FID at any resolution and bandwidth. 

Other basis spectra file formats supported by FSL-MRS: 

- LCModel (.BASIS) format: For interoperability FSL-MRS can read basis spectra in from LCModel .basis files. See the LCModel webpages for more information on this specific format.  
- JMRUI text format  
- LCModel Raw format (not recommended) 

Other recommended simulation environments include: MARSS, NMR ScopeB (jMRUI), VESPA 


References
----------

.. [SIMP17] `Simpson R et al. Advanced Processing and Simulation of MRS Data Using the FID Appliance (FID-A)-An Open Source, MATLAB-based Toolkit. Magn Reson Med 2017 <https://pubmed.ncbi.nlm.nih.gov/26715192/>`_

.. [GOVI00] `Govindaraju V et al. Proton NMR Chemical Shifts and Coupling Constants for Brain Metabolites. NMR Biomed. 2000 <https://pubmed.ncbi.nlm.nih.gov/26094860/>`_

.. [LAND19] `Landheer K et al. Magnetic Resonance Spectrum Simulator (MARSS), a Novel Software Package for Fast and Computationally Efficient Basis Set Simulation. NMR Biomed. 2019  <https://pubmed.ncbi.nlm.nih.gov/31313877>`_