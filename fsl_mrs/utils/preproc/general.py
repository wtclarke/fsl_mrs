"""General preprocessing functions"""
import numpy as np
from dataclasses import dataclass

@dataclass
class datacontainer:
    '''Class for keeping track of data and reference data together.'''
    data: np.array
    dataheader: dict
    datafilename: str
    reference: np.array = None
    refheader: dict = None
    reffilename: str = None

def get_target_FID(FIDlist,target='mean'):
    """
    target can be 'mean' or 'first' or 'nearest_to_mean' or 'median'
    """
    if target == 'mean':
        return sum(FIDlist) / len(FIDlist)
    elif target == 'first':
        return FIDlist[0].copy()
    elif target == 'nearest_to_mean':
        avg = sum(FIDlist) / len(FIDlist)
        d   = [np.linalg.norm(fid-avg) for fid in FIDlist]
        return FIDlist[np.argmin(d)].copy()
    elif target == 'median':
        return np.median(np.real(np.asarray(FIDlist)),axis=0)+1j*np.median(np.imag(np.asarray(FIDlist)),axis=0)
    else:
        raise(Exception('Unknown target type {}'.format(target)))

def subtract(FID1,FID2):
    """ Subtract FID2 from FID1."""
    return (FID1-FID2)/2.0

def add(FID1,FID2):
    """ Add FID2 to FID1."""
    return (FID1+FID2)/2.0
