#!/usr/bin/env python

# conftest.py - Test configuration
#
# Author: Vasilis Karlaftis <vasilis.karlaftis@ndcn.ox.ac.uk>
#
# Copyright (C) 2026 University of Oxford
# SHBASECOPYRIGHT

import os

# set plotting backend
os.environ["MPLBACKEND"] = "Agg"

# Keep threaded numeric libraries conservative during test runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
