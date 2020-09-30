.. _constants:

Constants
=========

The following constants are hardcoded into FSL-MRS and are used in the quantisation module.

Water to tetramethylsilane (TMS) chemical shift ::
    
    H2O_PPM_TO_TMS     = 4.65

Gyromagnetic ratio of proton::

    H1_gamma           = 42.576     # MHz/tesla

Concentration scaling parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tissue water density [0]_::

    TISSUE_WATER_DENSITY = {'GM':0.78,'WM':0.65,'CSF':0.97} 


Mollecular mass of water::

    H2O_MOLECULAR_MASS = 18.01528   # g/mol

Molality of pure water::

    H2O_MOLALITY = 55.51E3    # mmol/kg     

Number of protons in water::

    H2O_PROTONS = 2

Relaxation parameters
*********************
Values are derived from a survey of the literature. References listed below. Values for metabolites are derived from an average of NAA, Cr and Cho peaks.

T1 (all values in seconds)
__________________________
.. csv-table::
    :header: "Field strength (T)", "Water: WM", "Water: GM", "Water: CSF" , "Metabolites" 
    :widths: 10, 10, 10, 10, 10

    3,	        0.97,	1.50,	4.47,	1.29
    7,	        1.21,	2.05,	4.43,	1.43
    References,	1-6,	1-6,	4,	    "2, 7-9"


Code definitions::

    STANDARD_T1 = { '3T':{'H2O_WM':0.97, # Ref: 1-6
                        'H2O_GM':1.50, # Ref: 1-6
                        'H2O_CSF':4.47, # Ref: 4
                        'METAB':1.29}, # Ref: 2, 7-9
                    '7T':{'H2O_WM':1.21, # Ref: 1-6
                        'H2O_GM':2.05, # Ref: 1-6
                        'H2O_CSF':4.43, # Ref: 4
                        'METAB':1.43}} # Ref: 2, 7-9

    STANDARD_T2 = { '3T':{'H2O_WM':0.073, # Ref: 1,3,10-11
                        'H2O_GM':0.088, # Ref: 1,3,10-11
                        'H2O_CSF':2.030, # Ref: 12
                        'METAB':0.194}, # Ref: 7-9,13-15
                    '7T':{'H2O_WM':0.055, # Ref: 1,3,10-11
                        'H2O_GM':0.050, # Ref: 1,3,10-11
                        'H2O_CSF':1.050, # Ref: 12
                        'METAB':0.131}} # Ref: 7-9,13-15


References
----------

.. [0] Kreis R, Ernst T, Ross BD. Absolute quantitation of water and metabolites in the human brain. II. Metabolite concentrations. J Magn Reson B. 1993;102:9-19.
.. [1] Stanisz GJ et al. doi: 10.1002/mrm.20605.
.. [2] Ethofer T et al. doi: 10.1002/mrm.10640.
.. [3] Wansapura JP et al. doi: 10.1002/(SICI)1522-2586(199904)9:4<531::AID-JMRI4>3.0.CO;2-L.
.. [4] Rooney WD et al. doi: 10.1002/mrm.21122.
.. [5] Dieringer MA et al. doi: 10.1371/journal.pone.0091318.
.. [6] Wright PJ et al. doi: 10.1007/s10334-008-0104-8.
.. [7] Mlynárik V et al. doi: 10.1002/nbm.713.
.. [8] Li Y. doi: 10.4172/2155-9937.S1-002.
.. [9] An L et al. doi: 10.1002/mrm.26612.
.. [10] Gelman N et al. doi: 10.1148/radiology.210.3.r99fe41759.
.. [11] Bartha R et al. doi: 10.1002/mrm.10112.
.. [12] Spijkerman JM et al. doi: 10.1007/s10334-017-0659-3.
.. [13] Marjańska M et al. doi: 10.1002/nbm.1754.
.. [14] Träber F et al. doi: 10.1002/jmri.20053.
.. [15] Wyss PO et al. doi: 10.1002/mrm.27067.