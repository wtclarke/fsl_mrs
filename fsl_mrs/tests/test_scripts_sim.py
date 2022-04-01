'''FSL-MRS test script

Test the simulation script in FSL-MRS

Copyright Will Clarke, University of Oxford, 2021'''


import pytest
import json
import numpy as np
import os.path as op
import subprocess
from fsl_mrs.utils.mrs_io import fsl_io, lcm_io
from fsl_mrs.denmatsim import simseq as sim
seqfile = op.join(op.dirname(__file__), 'testdata/sim/example.json')


# construct two simple (1 spin) test spin systems
@pytest.fixture
def spinsys(tmp_path):
    spinsystem = {"sys1": {"shifts": [1, 2], "j": [[0, 0], [0, 0]], "scaleFactor": 1.0},
                  "sys2": {"shifts": [3], "j": [[0]], "scaleFactor": 3.0}}
    with open(op.join(tmp_path, 'custom_ss.json'), 'w', encoding='utf-8') as f:
        json.dump(spinsystem, f, ensure_ascii=False, indent='\t')

    zeroRefSS = {"shifts": [0], "j": [[0]], "scaleFactor": 1.0}
    spinsystem.update({'0ref': zeroRefSS})
    return spinsystem


# Load an example dataset
@pytest.fixture
def seqparams(tmp_path):
    with open(seqfile, 'r') as seqFile:
        jsonString = seqFile.read()
        seqFileParams = json.loads(jsonString)
    return seqFileParams


def test_sim(spinsys, seqparams, tmp_path):
    # Test simulation by running on one of the included sequence files
    # and then comparing to a manually simulated data set here.

    # Run the sequence on a couple of metabolites, ask for all types of files, add the reference peak
    ssFile = op.join(tmp_path, 'custom_ss.json')
    outfile = op.join(tmp_path, 'simulated')
    subprocess.call(['fsl_mrs_sim',
                     '-s', ssFile,
                     '-o', outfile,
                     '-r',
                     '-a', seqfile])

    # Load the data
    basis_j, names_j, header = fsl_io.readFSLBasisFiles(outfile)
    rawfiles = [op.join(tmp_path, 'simulated', 'sys1.RAW'),
                op.join(tmp_path, 'simulated', 'sys2.RAW')]
    basis_r, names = lcm_io.read_basis_files(rawfiles)

    # Check that this matches what is simulated manually
    directSim = []
    for ss in spinsys:
        print(ss)
        FID, ax, pmat = sim.simseq(spinsys[ss], seqparams)
        FID *= spinsys[ss]['scaleFactor']
        directSim.append(FID.conj())

    directSim1 = directSim[0] + directSim[2]
    directSim2 = directSim[1] + directSim[2]

    assert np.allclose(basis_j[:, names_j.index('sys1')], directSim1)
    assert np.allclose(basis_j[:, names_j.index('sys2')], directSim2)


def test_sim_workers(spinsys, seqparams, tmp_path):
    # Test simulation number of workers
    # Run the sequence on a couple of metabolites, ask for all types of files, add the reference peak

    ssFile = op.join(tmp_path, 'custom_ss.json')
    outfile1 = op.join(tmp_path, 'simulated_single')
    outfile2 = op.join(tmp_path, 'simulated_multi')
    assert not subprocess.check_call(
        ['fsl_mrs_sim',
         '-s', ssFile,
         '-o', outfile1,
         '-r',
         '-a', seqfile,
         '--num_processes', '1'])

    assert not subprocess.check_call(
        ['fsl_mrs_sim',
         '-s', ssFile,
         '-o', outfile2,
         '-r',
         '-a', seqfile,
         '--num_processes', '2'])
