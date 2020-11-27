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
    
    return np.abs(x-loc)**2/2/scale/scale + np.log(scale)


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

    return (1-shape)*np.log(x) + x/scale + shape*np.log(scale) + gammaln(shape)


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


def lognorm2stats(loc,scale):
    mu = np.exp(loc+scale**2/2)
    sig = np.sqrt(mu**2*(np.exp(scale**2-1)))
    return mu,sig

# quick plotting
import matplotlib.pyplot as plt
def plot_gausspdf(loc,scale,interval=None,numpoints=100):
    if interval is None:
        interval = np.linspace(loc-3*scale,loc+3*scale,numpoints)
    value = -(interval-loc)**2/2/scale**2
    value = np.exp(value-np.max(value))
    value = value/np.sum(value)
    fig = plt.plot(interval,value)
    return fig

def plot_lognormpdf(loc,scale,interval=None,numpoints=100):
    if interval is None:
        interval = np.linspace(1E-10,np.exp(loc)+2*np.exp(loc),numpoints)
    value = -(np.log(interval)-loc)**2/2/scale**2
    value = np.exp(value-np.max(value))/interval
    value = value/np.sum(value)
    fig = plt.plot(interval,value)
    return fig
