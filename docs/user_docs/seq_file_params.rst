.. _seq_file_params:

:orphan:

Sequence description parameters
===============================

.. csv-table::
    :header: "Field", "Value type","Req?", "Description" , "Example value" 
    :widths: 15, 10, 5, 100, 30

    sequenceName,   Str,	        Y,        Sequence name,                              svs_slaser
    description,    Str,	        Y,        User description,	                        3T slaser 28 ms
    B0,	            float,	        Y,        Static field strength,	                    6.98
    centralShift,	float,	        N,         Receiver offset from ppm reference. For most 1H MRS sequences this will be 4.65 ppm (the shift of water at 37 °C from TMS/DSS). Default = 0.0,	4.65
    RX_Points,	    int,	        Y,        Number of points in the final spectrum,	    4096
    RX_SW,	        float,	        Y,        Receiver sweep-width (bandwidth) in Hz,     6000
    RX_LW,	        float,	        Y,        FWHM of peaks in the output spectra. In Hz., 2.0
    RX_Phase,	    float,	        Y,        Zero-order phase applied to final spectra. In radians.,	0.0
    x,	            float (1x2),	Y,        Spatial range simulated in x direction. In units specified by spaceUnits., [-25 25]
    y,	            float (1x2),	Y,        As above for y,                             [-25 25]
    z,	            float (1x2),	Y,        As above for z,	                            [-25 25]
    resolution,	    Int (1x3),	    Y,        Number of spatial points to simulate in each dimension., [25 25 1]
    CoherenceFilter,Int (1x Nblocks),Y,	    Coherence order NOT zeroed at end of block. Value of element can be ‘None’ for no coherence selection. See below for more information.,	[1 0 -1]
    RFUnits,	    Str,	        N,         Units of RF amplitude (options: ’Hz’ ’T’ ’mT’ ’uT’; default=’Hz’) ,‘Hz’
    GradUnits,	    Str,	        N,         Units of gradient amplitude per meter (options: ’Hz’ ’T’ ’mT’; default=’T’), ‘mT’
    spaceUnits,	    Str,	        N,         Units of spatial position (options: ’mm’ ’cm’ ’m’; default=’m’), ‘mm’
    RF,	            List of NBlocks,Y,        Sequence block description. Contains the fields listed below., [block1 … blockN]

Sequence block parameters
~~~~~~~~~~~~~~~~~~~~~~~~~
Parameters required for the nested sequence blocks of the RF array in the above table.
These parameters describe one block of the sequence RF (with or without gradient),  rephasing, delay.

.. csv-table::
    :header: "Field", "Value type","Req?", "Description" , "Example value" 
    :widths: 15, 10, 5, 30, 30
    
    time,	        float,	        Y,        Total pulse time. In seconds.,              0.003
    frequencyOffset,float,	        N,         Offset of pulse central frequency from receiver centre. In Hz., -246
    phaseOffset,	float,	        N,         Phase offset applied to whole pulse. In radians. Default = 0., 0.0
    amp,	        List of floats,	Y,        List of amplitudes in units specified by RFUnits. Must contain >=1 points., [0 1.0 … 1.0 0.0]
    phase,	        List of floats, Y,        List of pulse phase points. Must match the size of amp. In radians., [0 0.0 … 0.0 0.0]
    grad,	        Float (1x3),	Y,        Slice selection gradient amplitude (in GradUnits/m) for each of the three spatial axes., [3.5 0 0]
    ampScale,	    Float,	        N,         Optional scaling of the pulse amplitude. Default = 1.0.,	2.0
    delays,	        Float (1x Nblocks),Y,     Free evolution time after RF in each RF block. In seconds. Measured from end of RF to start of next RF., [0.005 0.001 0.005]
    rephaseAreas,	Float (Nblocksx3),Y,	    Area of gradients (in seconds.GradUnits/m) on each spatial axis during the delay time. Can be applied on more than one axis per block., [[-5.1e-3  0  0] [ 0  -5.1e-3  0] [ 0  0  -5.1e-3]]





