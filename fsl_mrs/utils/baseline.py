#!/usr/bin/env python

# baseline.py - Functions associated with the baseline description
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from fsl_mrs.utils.misc import regress_out


def prepare_baseline_regressor(mrs, baseline_order, ppmlim):
    """
       Complex baseline is polynomial

    Parameters:
    -----------
    mrs            : MRS object
    baseline_order : degree of polynomial (>=1)
    ppmlim         : interval over which baseline is non-zero

    Returns:
    --------

    2D numpy array
    """

    first, last = mrs.ppmlim_to_range(ppmlim)

    B = []
    x = np.zeros(mrs.numPoints, np.complex)
    x[first:last] = np.linspace(-1, 1, last - first)

    for i in range(baseline_order + 1):
        regressor  = x**i
        if i > 0:
            regressor  = regress_out(regressor, B, keep_mean=False)

        B.append(regressor.flatten())
        B.append(1j * regressor.flatten())
    B = np.asarray(B).T
    tmp = B.copy()
    B   = 0 * B
    B[first:last, :] = tmp[first:last, :].copy()

    return B
