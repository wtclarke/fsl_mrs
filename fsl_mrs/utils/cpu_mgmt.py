#!/usr/bin/env python

# cpu_mgmt.py - Automatic detection of available CPUs
#
# Author: Vasilis Karlaftis <vasilis.karlaftis@ndcn.ox.ac.uk>
#
# Copyright (C) 2026 University of Oxford
# SHBASECOPYRIGHT

import os


def _read_cgroup_v2_limit():
    path = "/sys/fs/cgroup/cpu.max"
    if not os.path.exists(path):
        return None

    with open(path, "r") as fh:
        data = fh.read().strip().split()

    if len(data) < 2:
        return None

    quota, period = data[0], data[1]
    if quota == "max":
        return None

    quota = int(quota)
    period = int(period)
    if quota <= 0 or period <= 0:
        return None

    return max(1, quota // period)


def _read_cgroup_v1_limit():
    qpath = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    ppath = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if not os.path.exists(qpath) or not os.path.exists(ppath):
        return None

    with open(qpath) as qf, open(ppath) as pf:
        quota = int(qf.read().strip())
        period = int(pf.read().strip())

    if quota <= 0 or period <= 0:
        return None

    return max(1, quota // period)


def get_effective_cpu_count():
    """
    Return the most restrictive CPU limit visible to the current process.

    Candidate limits are taken from:
      1) os.sched_getaffinity(0) if available
      2) cgroup v2: /sys/fs/cgroup/cpu.max
      3) cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us and cpu.cfs_period_us
      4) os.cpu_count()

    Always returns an int >= 1.
    """
    limits = []

    try:
        cpu_set = os.sched_getaffinity(0)
        if cpu_set:
            limits.append(len(cpu_set))
    except (AttributeError, OSError):
        pass

    try:
        limit = _read_cgroup_v2_limit()
        if limit is not None:
            limits.append(limit)
    except (OSError, ValueError):
        pass

    try:
        limit = _read_cgroup_v1_limit()
        if limit is not None:
            limits.append(limit)
    except (OSError, ValueError):
        pass

    cpu_count = os.cpu_count()
    if cpu_count is not None:
        limits.append(cpu_count)

    return max(1, min(limits)) if limits else 1
