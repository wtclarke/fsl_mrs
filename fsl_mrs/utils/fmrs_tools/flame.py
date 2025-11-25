"""Module for wrapping FSL FLAME tool for higher level analysis of fMRS data.

    Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
            Saad Jbabdi <saad@fmrib.ox.ac.uk>

    Copyright (C) 2022 University of Oxford
    # SHBASECOPYRIGHT
    """

import tempfile
import subprocess
from pathlib import Path
import sys

import numpy as np
import scipy.stats

from fsl.data.vest import generateVest
from fsl.data.image import Image

os_name = sys.platform
win_platforms = {'win32', 'cygwin'}


def flameo_wrapper(cope, varcope, design_mat=None, contrast_mat=None, covariance_mat=None, ftests=None, verbose=False):
    """Wrapper around FSL FLAMEO for fMRS group analysis

    Apply FLAME stage 1 method (https://www.fmrib.ox.ac.uk/datasets/techrep/tr04ss2/tr04ss2/node4.html) to
    fMRS GLM betas calulated using FSL-MRS dynamic fitting. Optional kwargs allow the user to specify
    the design matrix, the t contrasts, and the groups the covariance is split into.

    :param cope: subjects x params numpy array of betas
    :type cope: np.array
    :param varcope: subjects x params numpy array of variances
    :type varcope: np.array
    :param design_mat: Group analysis design matrix, defaults to np.ones((nsubjects, 1))
    :type design_mat: np.array, optional
    :param contrast_mat: Group analysis contrasts matrix, defaults to np.ones((1, 1))
    :type contrast_mat: np.array, optional
    :param covariance_mat: Vector of covariance group assignments, defaults to np.ones((nsubjects, 1))
    :type covariance_mat: np.array, optional
    :param ftests: Vector of f-test selections, defaults to np.zeros((1, ncontrasts))
    :type ftests: np.array, optional
    :return: Output p values
    :rtype: np.array
    :return: Output z statistics
    :rtype: np.array
    :return: Output group-level COPEs
    :rtype: np.array
    :return: Output group-level VARCOPEs
    :rtype: np.array
    :return: Output group-level f-tests results. Dict of f, zf stats and p values
    :rtype: dict or None
    """

    nsubjects = cope.shape[0]
    nparams   = cope.shape[1]
    mats = {}

    def check_2d(mat, T=True):
        if mat.ndim == 1:
            if T:
                return np.atleast_2d(mat).T
            else:
                return np.atleast_2d(mat)
        else:
            return mat

    if design_mat is None:
        mats['desmat'] = np.ones((nsubjects, 1))
    else:
        mats['desmat'] = check_2d(design_mat)

    if contrast_mat is None:
        mats['conmat'] = np.ones((1, 1))
    else:
        mats['conmat'] = check_2d(contrast_mat)

    if covariance_mat is None:
        mats['covmat'] = np.ones((nsubjects, 1))
    else:
        mats['covmat'] = check_2d(covariance_mat)

    if ftests is not None:
        mats['fmat'] = check_2d(ftests, T=False)
        print(mats['fmat'])
        ncontrasts = mats['conmat'].shape[0]
        if mats['fmat'].shape[1] != ncontrasts:
            raise ValueError(
                f'The ftests matrix second dim ({mats["fmat"].shape[1]}) '
                f'must match the number of contrasts ({ncontrasts}).')

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp = Path(tmpdirname)
        # generate vest files
        for f, m in mats.items():
            with open(tmp / f, "w") as txtfile:
                txtfile.write(generateVest(m))

        # save image files
        shape = (nparams, 1, 1, nsubjects)
        cope    = np.reshape(cope.T, shape)
        varcope = np.reshape(varcope.T, shape)
        img = Image(cope)
        img.save(tmp / 'cope')
        img = Image(varcope)
        img.save(tmp / 'varcope')
        img = Image(1 + 0 * cope[:, :, :, 0])
        img.save(tmp / 'mask')

        flameo_cmd = [
            'flameo',
            f'--ld={str(tmp / "logs")}',
            f'--cope={str(tmp / "cope")}',
            f'--varcope={str(tmp / "varcope")}',
            f'--mask={str(tmp / "mask")}',
            f'--dm={str(tmp / "desmat")}',  # design matrix file
            f'--tc={str(tmp / "conmat")}',  # file containing matrix specifying the t contrasts
            f'--cs={str(tmp / "covmat")}',  # file containing matrix specifying the covariance groups
            '--runmode=flame1',  # (mixed effects - FLAME stage 1)
            '--sdof=-1']

        if ftests is not None:
            flameo_cmd.append(f'--fc={str(tmp / "fmat")}')  # file containing vector specifying the f-tests

        # run flameo
        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
        try:
            subprocess.run(
                flameo_cmd,
                stdout=stdout,
                check=True)
        except subprocess.CalledProcessError as exc:
            if os_name in win_platforms:
                msg = 'FSL tool flameo not found. It is not installable on Windows, unless you follow the WSL instructions in the FSL-MRS documentation.'
            else:
                msg = 'FSL tool flameo not found. Please install FSL or fsl-flameo using conda..'
            raise FileNotFoundError("\033[91m"+msg+"\033[0m") from exc

        # collect results
        p = []
        z = []
        out_cope = []
        out_varcope = []
        # Loop over Contrasts
        for idx in range(mats['conmat'].shape[0]):
            zs = Image(tmp / 'logs' / f'zstat{idx + 1}').data[:, 0, 0]
            p.append(scipy.stats.norm.sf(zs))
            z.append(zs)
            out_cope.append(Image(tmp / 'logs' / f'cope{idx + 1}').data[:, 0, 0])
            out_varcope.append(Image(tmp / 'logs' / f'varcope{idx + 1}').data[:, 0, 0])

        if ftests is not None:
            f = []
            zf = []
            pf = []
            # Loop over ftests
            for idx in range(mats['fmat'].shape[0]):
                fs = Image(tmp / 'logs' / f'fstat{idx + 1}').data[:, 0, 0]
                f.append(fs)
                zfs = Image(tmp / 'logs' / f'zfstat{idx + 1}').data[:, 0, 0]
                zf.append(zfs)
                pf.append(scipy.stats.norm.sf(zfs))

            f = np.stack(f)
            zf = np.stack(zf)
            pf = np.stack(pf)
            fout = {'f-stat': f, 'zf-stat': zf, 'p': pf}
        else:
            fout = None

    return np.stack(p), np.stack(z), np.stack(out_cope), np.stack(out_varcope), fout
