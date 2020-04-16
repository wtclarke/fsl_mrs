#!/usr/bin/env python

# Distributions
#
# Authors: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT



import numpy as np
from scipy.special import gammaln


###############################################
#       -logp for common distributions        #
###############################################

def gauss_logpdf(x,loc=0.0,scale=1.0):
    """
    Gaussian prior

    prior     = exp( -(x-loc)^2/2/scale^2 )/scale

    Parameters
    ----------
    x: array-like
       
    loc: float, optional
         Location parameter (mean)
    scale: float
         Scale parameter (standard deviation)

    Returns
    -------
    array

    """
    
    return (x-loc)**2/2/scale/scale + np.log(scale);


def gamma_logpdf(x,shape,scale):

    """
    Gamma prior

    prior     = x^(shape-1)*exp(-x/scale)/scale^shape/Gamma(shape)

    Parameters
    ----------
    x: array-like
       
    shape: float, optional
         Shape parameter 
    scale: float
         Scale parameter 
    
    Returns
    -------
    array

    """

    return (1-shape)*np.log(x) + x/scale + shape*np.log(scale) + gammaln(shape);


def inverse_logpdf(x):

    """
    Inverse prior (Jeffrey's prior for scale param)

    prior = 1/x

    Returns
    -------
    array

    """
    return np.log(x)

def uniform_logpdf(x):

    """    

    prior = 1

    Returns
    -------
    array

    """
    return 0*x

