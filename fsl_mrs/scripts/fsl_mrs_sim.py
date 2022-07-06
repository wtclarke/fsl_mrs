#!/usr/bin/env python

# fsl_mrs_sim - wrapper script for MRS spectra simulations
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

# Quick imports
import argparse
from fsl_mrs import __version__
from fsl_mrs.utils.splash import splash
# Note there are imports after argparse


def main():

    p = argparse.ArgumentParser(
        description='FSL Magnetic Resonance Spectroscopy Tools')
    p.add_argument('-v', '--version', action='version', version=__version__)

    required = p.add_argument_group('required arguments')
    mutual = required.add_mutually_exclusive_group(required=True)
    optional = p.add_argument_group('optional arguments')

    # positional REQUIRED ARGUMENTS
    p.add_argument('seqFile', type=str, metavar='SEQUENCE',
                   help='pulse sequence file or previous results file')

    # Mutual required arguments
    mutual.add_argument('-m', '--metab', type=str, metavar='METAB',
                        help='Single metabolite string')

    mutual.add_argument('-b', '--batch', type=str, metavar='METAB_FILE',
                        help='Batch simulate metabolites. Path to file containing metabolite list.')

    mutual.add_argument('-s', '--spinsys', type=str, metavar='spinsysJSON',
                        help='Pass custom spin systems defined in json.')

    # OPTIONAL ARGUMENTS
    optional.add_argument('-o', '--output', default='.',
                          required=False, type=str, metavar='<str>',
                          help='output folder')

    optional.add_argument('-r', '--raw', action="store_true",
                          help='Output LCModel Style Raw files')

    optional.add_argument('-j', '--jmrui', action="store_true",
                          help='Output jMRUI Style txt files')

    optional.add_argument('-a', '--addref', action="store_true",
                          help='Add 0 ppm reference to all outputs')

    optional.add_argument('-p', '--autophase', type=float, metavar='<ppm>',
                          help='Simulate a singlet peak at <ppm> to auto-phase the basis spectra.'
                               ' Relative to reviever central frequency.')

    optional.add_argument('--lcmIN', type=str,
                          required=False, metavar='<str>',
                          help='Location to enter into lcm IN file.')

    optional.add_argument('--MM', type=str,
                          required=False, metavar='<str>',
                          help='JSON file containing MM FID to add to the basis output.')

    optional.add_argument('-e', '--outputTE', type=float,
                          required=False, metavar='ECHOTIME',
                          help='Echo time value in ms for output files (no effect on simulation).')

    optional.add_argument('--num_processes', type=int,
                          required=False, default=None,
                          help='Number of worker processes to use in simulation, defaults to os.cpu_count().')

    # optional.add_argument('--verbose',action="store_true",
    #                help='spit out verbose info')
    optional.add_argument('--overwrite', action="store_true",
                          help='overwrite existing output folder')
    optional.add_argument('--verbose', action="store_true",
                          help='Verbose output')

    # Parse command-line arguments
    args = p.parse_args()

    # Output kickass splash screen
    if args.verbose:
        splash(logo='mrs')

    # ######################################################
    # DO THE IMPORTS AFTER PARSING TO SPEED UP HELP DISPLAY
    import shutil
    import json
    from fsl_mrs.denmatsim import utils as simutils
    from fsl_mrs.utils.misc import FIDToSpec
    import numpy as np
    from fsl_mrs.denmatsim import simseq as sim
    from fsl_mrs.utils.mrs_io import fsl_io, lcm_io, jmrui_io
    import os
    import datetime
    # ######################################################

    # Check if output folder exists
    overwrite = args.overwrite
    if args.output != '.':
        if os.path.exists(args.output):
            if not overwrite:
                print("Folder '{}' exists. Are you sure you want to delete it? [Y,N]".format(args.output))
                response = input()
                overwrite = response.upper() == "Y"
            if not overwrite:
                print('Early stopping...')
                exit()
            else:
                shutil.rmtree(args.output)
                os.mkdir(args.output)
        else:
            os.mkdir(args.output)

    # Do the work
    # Load the sequence file
    with open(args.seqFile, 'r') as seqFile:
        # If exisitng results file then extract the relavent dict
        # Identify if file has a seq field
        jsonString = seqFile.read()
        seqFileParams = json.loads(jsonString)
        if 'seq' in seqFileParams:
            seqParams = seqFileParams['seq']
        else:  #  Assume that this is a plain sequence file and contains just the information we need
            seqParams = seqFileParams

    # Identify spin systems to run
    if args.spinsys is not None:  # Parse the file containing a spin system description
        with open(args.spinsys, 'r') as spinFile:
            jsonString = spinFile.read()
            spinsysIn = json.loads(jsonString)
            if 'shifts' in spinsysIn:  # assume single spin system
                spinsys = [spinsysIn, ]
                basename = os.path.basename(args.spinsys)
                prefix, suffix = os.path.splitext(basename)
                spinsToSim = [prefix, ]
            else:  # Assume multiple systems and create a list
                spinsys = []
                spinsToSim = []
                for systems in spinsysIn:
                    spinsys.append(spinsysIn[systems])
                    spinsToSim.append(systems)

    else:
        # Read built in spin system file
        spinSystems = simutils.readBuiltInSpins()

        if args.batch is not None:
            # Parse file containing list of the known metabolites. Extract these from that list
            if os.path.isfile(args.batch):
                with open(args.batch, 'r') as batchFile:
                    metabsRequested = batchFile.readlines()
                    metabsRequested = [lines.rstrip() for lines in metabsRequested if lines != '\n']
            else:
                raise ValueError('batch argument expects a file name.')

        elif args.metab is not None:  # Parse sting of metabolites that (hopefully) match known metabolites
            metabsRequested = args.metab.split(',')
        else:
            raise ValueError('Unknown spin system imput.')

        spinsys = []
        spinsToSim = []
        for metab in metabsRequested:
            sysmetab = 'sys' + metab
            if sysmetab in spinSystems:
                spinsys.append(spinSystems[sysmetab])
                spinsToSim.append(metab)
            else:
                print(f'{metab}({sysmetab}) not recognised.')
        print(f'Identified spinsystems: {spinsToSim}')

    # Auto phase adjustment
    # breakpoint()
    if args.autophase is not None:
        # Construct a single spin spin system
        if 'centralShift' in seqParams:
            apshift = args.autophase + seqParams['centralShift']
        else:
            apshift = args.autophase
        apsystem = {'shifts': np.array([apshift]),
                    'j': np.array([[0.0]]),
                    'scaleFactor': 1.0}
        # Simulate it
        FID, ax, pmat = sim.simseq(apsystem, seqParams)

        # Determine phase adj needed
        FID = np.pad(FID, (0, 10 * 8192))
        apspec = FIDToSpec(FID)
        maxindex = np.argmax(np.abs(apspec))
        apPhase = np.angle(apspec[maxindex])
        print(f'Auto-phase adjustment. Phasing peak position = {apshift:0.2f} ppm')
        print(f'Rx_Phase: {seqParams["Rx_Phase"]}')
        newRx_Phase = seqParams["Rx_Phase"] + apPhase
        seqParams["Rx_Phase"] = newRx_Phase
        print(f'Additional phase: {apPhase:0.3f}\nFinal Rx_Phase: {newRx_Phase:0.3f}')

    if args.addref:
        # Add an extra spin system to the list with a single resonance at 0 ppm
        # Yes, this isn't the most efficent way of doing this, but this is a one
        # spin system.
        zeroRefSS = {"shifts": [0], "j": [[0]], "scaleFactor": 1.0}
        for iDx, s in enumerate(spinsys):
            if isinstance(spinsys[iDx], dict):
                spinsys[iDx] = [spinsys[iDx], zeroRefSS]
            elif isinstance(spinsys[iDx], list):
                spinsys[iDx] = spinsys[iDx] + [zeroRefSS]
            else:
                raise ValueError('spinsys[iDx] should either be a dict or list of dicts')

    import multiprocessing as mp
    from functools import partial
    # global_counter = mp.Value('L')

    # Loop over the spin systems (list) in a parallel way
    poolFunc = partial(runSimForMetab, seqParams=seqParams, args=args)
    poolArgs = [(i, s, n) for i, (s, n) in enumerate(zip(spinsys, spinsToSim))]
    # handle number of processes
    pool = mp.Pool(args.num_processes)
    pool.starmap(poolFunc, poolArgs)

    # Additional write steps for MM and LCM basis generation
    if args.MM is not None:
        # Read in file
        MMBasisFID = fsl_io.readJSON(args.MM)  # Input should be the same as the format of basisDict above

        # Write MM FID to the correct format
        if args.raw:
            fileOut = os.path.join(args.output, 'Mac.RAW')
            combinedScaledFID = np.array(MMBasisFID['basis_re']) + 1j * np.array(MMBasisFID['basis_im'])
            info = {'ID': 'Mac', 'VOLUME': 1.0, 'TRAMP': 1.0}
            lcm_io.saveRAW(fileOut, combinedScaledFID, info=info, conj=True)
        if args.jmrui:
            fileOut = os.path.join(args.output, 'Mac.txt')
            combinedScaledFID = np.array(MMBasisFID['basis_re']) + 1j * np.array(MMBasisFID['basis_im'])
            jmruidict = {'dwelltime': 1 / seqParams['Rx_SW'],
                         'centralFrequency': ax['centreFreq']}
            jmrui_io.writejMRUItxt(fileOut, combinedScaledFID, jmruidict)

        # Always output fsl_mrs
        # Form results dictionary
        outputDict = {}
        outputDict.update({'seq': None})
        outputDict.update({'basis': MMBasisFID})
        outputDict.update({'MM': True})
        metaDict = {'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 'SimVersion': ''}
        outputDict.update({'meta': metaDict})
        # Save to json file.
        fileOut = os.path.join(args.output, 'Mac.json')
        fsl_io.writeJSON(fileOut, outputDict)

        # Add MM to list of metabs
        spinsToSim = spinsToSim + ['Mac']

    if args.lcmIN is not None:
        # Create .IN file that can be used to create LCModel .basis file from raw files
        inputSeqBaseName, _ = os.path.splitext(os.path.basename(args.seqFile))
        outputBaseName = args.lcmIN

        fileOut = os.path.join(args.output, 'lcm.IN')
        inParamDict = {'width': seqParams['Rx_LW'],
                       'centralFrequency': seqParams['B0'] * 42.5774,
                       'dwelltime': 1 / seqParams['Rx_SW'],
                       'points': seqParams['Rx_Points']}
        if args.outputTE is not None:
            lcm_io.writeLcmInFile(
                fileOut,
                spinsToSim,
                outputBaseName,
                inputSeqBaseName,
                inParamDict,
                echotime=args.outputTE)
        else:
            lcm_io.writeLcmInFile(fileOut, spinsToSim, outputBaseName, inputSeqBaseName, inParamDict)


def runSimForMetab(iDx, s, name, seqParams, args):
    """Run the simulation for a single metabolite"""
    import copy
    import numpy as np
    from fsl_mrs.utils.mrs_io import fsl_io, lcm_io, jmrui_io
    from fsl_mrs.denmatsim import simseq as sim
    import os
    import datetime

    print(f'Running simulation on {name}.')
    sToSave = copy.deepcopy(s)  # Make a copy here as some bits of s are converted to np.arrays inside simulator.
    # Run simulation
    densityMatriciesRe = []
    densityMatriciesIm = []
    if isinstance(s, list):
        combinedScaledFID = []
        for ss in s:
            FID, ax, pmat = sim.simseq(ss, seqParams)
            combinedScaledFID.append(FID * ss['scaleFactor'])
            densityMatriciesRe.append(np.real(pmat).tolist())
            densityMatriciesIm.append(np.imag(pmat).tolist())
        combinedScaledFID = np.sum(combinedScaledFID, axis=0)

    else:
        FID, ax, pmat = sim.simseq(s, seqParams)
        combinedScaledFID = FID * s['scaleFactor']
        densityMatriciesRe.append(np.real(pmat).tolist())
        densityMatriciesIm.append(np.imag(pmat).tolist())

    basisDict = {}
    basisDict.update({'basis_re': np.real(combinedScaledFID).tolist()})
    basisDict.update({'basis_im': np.imag(combinedScaledFID).tolist()})
    basisDict.update({'basis_dwell': 1 / seqParams['Rx_SW']})
    basisDict.update({'basis_centre': ax['centreFreq']})
    basisDict.update({'basis_width': seqParams['Rx_LW']})
    basisDict.update({'basis_name': name})

    if args.raw:
        fileOut = os.path.join(args.output, name + '.RAW')
        info = {'ID': name, 'VOLUME': 1.0, 'TRAMP': 1.0}
        lcm_io.saveRAW(fileOut, combinedScaledFID, info=info, conj=True)
    if args.jmrui:
        fileOut = os.path.join(args.output, name + '.txt')
        jmruidict = {'dwelltime': 1 / seqParams['Rx_SW'],
                     'centralFrequency': ax['centreFreq']}
        jmrui_io.writejMRUItxt(fileOut, combinedScaledFID, jmruidict)

    # Always output fsl_mrs
    # Form results dictionary
    outputDict = {}
    outputDict.update({'seq': seqParams})
    outputDict.update({'basis': basisDict})
    outputDict.update({'spinSys': sToSave})
    densmatdict = {'re': densityMatriciesRe, 'im': densityMatriciesIm}
    outputDict.update({'outputDensityMatrix': densmatdict})
    metaDict = {'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 'SimVersion': ''}
    if args.outputTE is not None:
        metaDict.update({'TE': args.outputTE})
    outputDict.update({'meta': metaDict})
    # Save to json file.
    fileOut = os.path.join(args.output, name + '.json')
    fsl_io.writeJSON(fileOut, outputDict)


if __name__ == '__main__':
    main()
