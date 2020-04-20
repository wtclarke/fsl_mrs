#!/usr/bin/env python

# fitting.py - Fit MRS models
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import numpy as np

from fsl_mrs.utils import models, misc
from fsl_mrs.utils.stats import mh,vb
from fsl_mrs.utils.constants import *
from fsl_mrs.core import MRS
from fsl_mrs.utils.results import FitRes

from scipy.optimize import minimize,nnls

        
def print_params(x,mrs,metab_groups,ref_metab='Cr',scale_factor=1):
    """
       Print parameters 
    """
    g = max(metab_groups)+1
    con,gamma,eps,phi0,phi1,b=models.FSLModel_x2param(x,mrs.numBasis,g)
    print('-----------------------------------------------------------------')
    print('gamma  = {}'.format(gamma))
    print('eps    = {}'.format(eps))
    print('phi0   = {}'.format(phi0))
    print('phi1   = {}'.format(phi1))
    print('b      = {}'.format(b))
    dict_con = dict(zip(mrs.names,con))
    norm_con = [scale_factor*dict_con[i]/dict_con[ref_metab] for i in mrs.names]
    print(dict(zip(mrs.names,norm_con)))
    print('-----------------------------------------------------------------')


# New strategy for init
def init_params(mrs,baseline,ppmlim):
    first,last = mrs.ppmlim_to_range(ppmlim)
    y = mrs.getSpectrum(ppmlim=ppmlim)
    y = np.concatenate((np.real(y),np.imag(y)),axis=0).flatten()
    B = baseline[first:last,:].copy()
    B = np.concatenate((np.real(B),np.imag(B)),axis=0)
    
    def modify_basis(mrs,gamma,eps):
        bs = mrs.basis * np.exp(-(gamma+1j*eps)*mrs.timeAxis)        
        bs = misc.FIDToSpec(bs,axis=0)
        bs = bs[first:last,:]
        return np.concatenate((np.real(bs),np.imag(bs)),axis=0)
            
    def loss(p):
        gamma,eps = np.exp(p[0]),p[1]
        basis     = modify_basis(mrs,gamma,eps)
        desmat    = np.concatenate((basis,B),axis=1)
        beta      = np.real(np.linalg.pinv(desmat)@y)
        beta[:mrs.numBasis] = np.clip(beta[:mrs.numBasis],0,None) # project onto >0 concentration
        pred      = np.matmul(desmat,beta)
        val       = np.mean(np.abs(pred-y)**2)    
        return val
        
    x0  = np.array([np.log(1e-5),0])
    res = minimize(loss, x0, method='Powell')
    
    g,e = np.exp(res.x[0]),res.x[1]

    # get concentrations and baseline params 
    basis  = modify_basis(mrs,g,e)
    desmat = np.concatenate((basis,B),axis=1)
    beta   = np.real(np.linalg.pinv(desmat)@y)
    con    = np.clip(beta[:mrs.numBasis],0,None)
    #con    = beta[:mrs.numBasis]
    b      = beta[mrs.numBasis:]

    return g,e,con,b


def init_FSLModel(mrs,metab_groups,baseline,ppmlim):
    """
       Initialise params of FSLModel
    """

    gamma,eps,con,b0 = init_params(mrs,baseline,ppmlim)
    
    # Append 
    x0  = con                                    # concentrations
    g   = max(metab_groups)+1                    # number of metab groups
    x0  = np.append(x0,[gamma]*g)                # gamma[0]..
    x0  = np.append(x0,[eps]*g)                  # eps[0]..
    x0  = np.append(x0,[0,0])                    # phi0 and phi1
    x0  = np.append(x0,b0)                       # baseline
    
    return x0

def init_params_voigt(mrs,baseline,ppmlim):
    first,last = mrs.ppmlim_to_range(ppmlim)
    y = mrs.getSpectrum(ppmlim=ppmlim)
    y = np.concatenate((np.real(y),np.imag(y)),axis=0).flatten()
    B = baseline[first:last,:].copy()
    B = np.concatenate((np.real(B),np.imag(B)),axis=0)
    
    def modify_basis(mrs,gamma,sigma,eps):
        bs = mrs.basis * np.exp(-(gamma+(sigma**2*mrs.timeAxis)+1j*eps)*mrs.timeAxis)        
        bs = misc.FIDToSpec(bs,axis=0)
        bs = bs[first:last,:]
        return np.concatenate((np.real(bs),np.imag(bs)),axis=0)
            
    def loss(p):
        gamma,sigma,eps = np.exp(p[0]),np.exp(p[1]),p[2]
        basis     = modify_basis(mrs,gamma,sigma,eps)
        desmat    = np.concatenate((basis,B),axis=1)
        beta      = np.real(np.linalg.pinv(desmat)@y)
        beta[:mrs.numBasis] = np.clip(beta[:mrs.numBasis],0,None) # project onto >0 concentration
        pred      = np.matmul(desmat,beta)
        val       = np.mean(np.abs(pred-y)**2)    
        return val
        
    x0  = np.array([np.log(1e-5),np.log(1e-5),0])
    res = minimize(loss, x0, method='Powell')
    
    g,s,e = np.exp(res.x[0]),np.exp(res.x[1]),res.x[2]

    # get concentrations and baseline params 
    basis  = modify_basis(mrs,g,s,e)
    desmat = np.concatenate((basis,B),axis=1)
    beta   = np.real(np.linalg.pinv(desmat)@y)
    con    = np.clip(beta[:mrs.numBasis],0,None)
    #con    = beta[:mrs.numBasis]
    b      = beta[mrs.numBasis:]

    return g,s,e,con,b


def init_FSLModel_Voigt(mrs,metab_groups,baseline,ppmlim):
    """
       Initialise params of FSLModel for Voigt linesahapes
    """
    gamma,sigma,eps,con,b0 = init_params_voigt(mrs,baseline,ppmlim)

    # Append 
    x0 = con
    g   = max(metab_groups)+1                  # number of metab groups
    x0  = np.append(x0,[gamma]*g)              # gamma[0]..
    x0  = np.append(x0,[sigma]*g)              # sigma[0]..
    x0  = np.append(x0,[eps]*g)                # eps[0]..
    x0  = np.append(x0,[0,0])                  # phi0 and phi1
    x0  = np.append(x0,b0)                     # baseline
    
    return x0

# ####################################################################################


def prepare_baseline_regressor(mrs,baseline_order,ppmlim):
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

    first,last = mrs.ppmlim_to_range(ppmlim)
    
    B = []
    x = np.zeros(mrs.numPoints,np.complex) 
    x[first:last] = np.linspace(-1,1,last-first)
    
    for i in range(baseline_order+1):
        regressor  = x**i
        if i>0:
            #regressor  = regressor - np.mean(regressor)
            regressor  = misc.regress_out(regressor,B,keep_mean=False)
            
        B.append(regressor.flatten())
        B.append(1j*regressor.flatten())
    B = np.asarray(B).T
    tmp = B.copy()
    B   = 0*B
    B[first:last,:] = tmp[first:last,:].copy()
    
    return B


def get_bounds(num_basis,num_metab_groups,baseline_order,model,method,disableBaseline=False):
    if method == 'Newton':
        # conc
        bnds = [(0,None)]*num_basis
        # gamma/sigma/eps
        bnds.extend([(0,None)]*num_metab_groups)
        if model.lower() == 'voigt':
            bnds.extend([(0,None)]*num_metab_groups)
        bnds.extend([(None,None)]*num_metab_groups)
        # phi0,phi1
        bnds.extend([(None,None)]*2)
        # baseline
        n = (1+baseline_order)*2
        if disableBaseline:
            bnds.extend([(0.0,0.0)]*n)
        else:
            bnds.extend([(None,None)]*n)
        return bnds

    elif method == 'MH':
        MAX =  1e10
        MIN = -1e10
        # conc
        LB = [0]*num_basis
        UB = [MAX]*num_basis
        # gamma/sigma/eps
        LB.extend([0]*num_metab_groups)
        UB.extend([MAX]*num_metab_groups)        
        if model.lower() == 'voigt':
            LB.extend([0]*num_metab_groups)
            UB.extend([MAX]*num_metab_groups)
        LB.extend([MIN]*num_metab_groups)
        UB.extend([MAX]*num_metab_groups)        
        # phi0,phi1
        LB.extend([MIN]*2)
        UB.extend([MAX]*2)
        # baseline
        n = (1+baseline_order)*2
        if disableBaseline:
            LB.extend([0.0]*n)
            UB.extend([0.0]*n)
        else:
            LB.extend([MIN]*n)
            UB.extend([MAX]*n)

        return LB,UB


    else:
        raise(Exception(f'Unknown method {method}'))
            
def get_fitting_mask(num_basis,num_metab_groups,baseline_order,model,
                     fit_conc=True,fit_shape=True,fit_phase=True,fit_baseline=False):

    if fit_conc:
        mask = [1]*num_basis
    else:
        mask = [0]*num_basis
    n = 2*num_metab_groups
    if model.lower() == 'voigt':
        n += num_metab_groups
    if fit_shape:
        mask.extend([1]*n)
    else:
        mask.extend([0]*n)
    if fit_phase:
        mask.extend([1]*2)
    else:
        mask.extend([0]*2)
    n = (1+baseline_order)*2
    if fit_baseline:
        mask.extend([1]*n)
    else:        
        mask.extend([0]*n)
    return mask


def fit_FSLModel(mrs,
                 method='Newton',
                 ppmlim=None,
                 baseline_order=5,
                 metab_groups=None,
                 model='lorentzian',
                 x0=None,
                 MHSamples=500):
    """
        A simplified version of LCModel
    """
    err_func,grad_func,forward,x2p,p2x = models.getModelFunctions(model)
    if model.lower() == 'lorentzian':     
        init_func  = init_FSLModel         # initialisation of params
    elif model.lower() == 'voigt':        
        init_func  = init_FSLModel_Voigt    # initialisation of params

    data       = mrs.Spec.copy()              # data copied to keep it safe
    first,last = mrs.ppmlim_to_range(ppmlim)  # data range

    if metab_groups is None:
        metab_groups = [0]*len(mrs.names)

    # shorter names for some of the useful stuff
    freq,time,basis=mrs.frequencyAxis,mrs.timeAxis,mrs.basis

    # Handle completely disabling baseline
    if baseline_order < 0:
        baseline_order = 0 # Generate one order of baseline parameters
        disableBaseline = True # But disable by setting bounds to 0
    else:
        disableBaseline = False
   
    # Prepare baseline
    B                 = prepare_baseline_regressor(mrs,baseline_order,ppmlim)
   
    # Results object
    results = FitRes(model,method,mrs.names,metab_groups,baseline_order,B,ppmlim)

    # Constants
    g         = results.g
    constants = (freq,time,basis,B,metab_groups,g,data,first,last)    

    if x0 is None:
        # Initialise all params
        x0 = init_func(mrs,metab_groups,B,ppmlim)
        
        
    # Fitting
    if method == 'Newton':
        # Bounds
        bounds = get_bounds(mrs.numBasis,g,baseline_order,model,method,disableBaseline=disableBaseline)                
        res    = minimize(err_func, x0, args=constants,
                          method='TNC',jac=grad_func,bounds=bounds)
        # collect results
        results.loadResults(mrs,res.x)        

    elif method == 'init':
        results.loadResults(mrs,x0)
  
    elif method == 'MH':
        forward_mh = lambda p : forward(p,freq,time,basis,B,metab_groups,g)
        numPoints_over_2  = (last-first)/2.0
        y      = data[first:last]
        loglik = lambda  p : np.log(np.linalg.norm(y-forward_mh(p)[first:last]))*numPoints_over_2
        logpr  = lambda  p : 0 

        # Setup the fitting
        # Init with nonlinear fit
        res  = fit_FSLModel(mrs,method='Newton',ppmlim=ppmlim,
                            metab_groups=metab_groups,baseline_order=baseline_order,model=model)
        # Create maks and bounds for MH fit
        p0   = res.params

        LB,UB = get_bounds(mrs.numBasis,g,baseline_order,model,method)                
        mask  = get_fitting_mask(mrs.numBasis,g,baseline_order,model,fit_baseline=False)        

        # Check that the values initilised by the newton
        # method don't exceed these bounds (unlikely but possible with bad data)
        for i,(p, u, l) in enumerate(zip(p0, UB, LB)):
            if p>u:        
                p0[i]=u        
            elif p<l:
                p0[i]=l

        # Do the fitting
        mcmc    = mh.MH(loglik,logpr,burnin=100,njumps=MHSamples)
        samples = mcmc.fit(p0,LB=LB,UB=UB,verbose=False,mask=mask)

        # collect results
        results.loadResults(mrs,samples)

    elif method == 'VB':
        import warnings
        warnings.warn('VB method still under development!',UserWarning)

        # init with nonlinear fitting
        res  = fit_FSLModel(mrs,method='Newton',ppmlim=ppmlim,
                            metab_groups=metab_groups,baseline_order=baseline_order,model=model)
        x0   = res.params

        

        # run VB fitting
        if model.lower()=='lorentzian':
            # log-transform positive params
            con,gamma,eps,phi0,phi1,b = x2p(x0,mrs.numBasis,g)
            con[con<=0]     = 1e-10
            gamma[gamma<=0] = 1e-10
            logcon,loggamma = np.log(con),np.log(gamma)
            vbx0 = p2x(logcon,loggamma,eps,phi0,phi1,b)
            vbmodel   = vb.NonlinVB(models.FSLModel_forward_vb)
        elif model.lower()=='voigt':
            # log-transform positive params
            con,gamma,sigma,eps,phi0,phi1,b = x2p(x0,mrs.numBasis,g)
            con[con<=0]     = 1e-10
            gamma[gamma<=0] = 1e-10
            sigma[sigma<=0] = 1e-10
            logcon,loggamma,logsigma = np.log(con),np.log(gamma),np.log(sigma)
            vbx0 = p2x(logcon,loggamma,logsigma,eps,phi0,phi1,b)
            vbmodel   = vb.NonlinVB(models.FSLModel_forward_vb_voigt)

        datasplit = np.concatenate((np.real(data[first:last]),np.imag(data[first:last])))
        args      = [freq,time,basis,B,metab_groups,g,first,last]                
        
        res_vb    = vbmodel.fit(y=datasplit,
                                x0=vbx0,
                                verbose=False,
                                monitor=True,
                                args=args)
        results.optim_out = res_vb

        # de-log
        if model.lower()=='lorentzian':
            logcon,loggamma,eps,phi0,phi1,b = x2p(res_vb.x,mrs.numBasis,g)
            x = p2x(np.exp(logcon),np.exp(loggamma),eps,phi0,phi1,b)
        elif model.lower()=='voigt':
            logcon,loggamma,logsigma,eps,phi0,phi1,b = x2p(res_vb.x,mrs.numBasis,g)
            x = p2x(np.exp(logcon),np.exp(loggamma),np.exp(logsigma),eps,phi0,phi1,b)

        # collect results
        results.loadResults(mrs,x)
        
    else:
        raise Exception('Unknown optimisation method.')


    # End of fitting

    return results







# # Parallel fitting
# def parallel_fit(fid_list,MRSargs,Fitargs,verbose):
#     import multiprocessing as mp
#     from functools import partial
#     import time
#     global_counter = mp.Value('L')

#     # Define some ugly local functions for parallel processing
#     def runworker(FID,MRSargs,Fitargs):
#         mrs = MRS(FID=FID,**MRSargs)        
#         res = fit_FSLModel(mrs,**Fitargs)   
#         with global_counter.get_lock():
#             global_counter.value += 1
#         return res
#     def parallel_runs(data_list):
#         pool    = mp.Pool(processes=mp.cpu_count())
#         func    = partial(runworker,MRSargs=MRSargs,Fitargs=Fitargs) 
#         results = pool.map_async(func,data_list)
#         return results

#     # Fitting
#     if verbose:
#         print('    Parallelising over {} workers '.format(mp.cpu_count()))
#     t0  = time.time()
#     results = parallel_runs(fid_list)

#     while not results.ready():
#         if verbose:
#             print('{}/{} voxels completed'.format(global_counter.value,len(fid_list)), end='\r')
#         time.sleep(1)
#     if verbose:
#         print('{}/{} voxels completed'.format(global_counter.value,len(fid_list)), end='\r')
#         print('\n\nFitting done in {:0f} secs.'.format(time.time()-t0))


#     if not results.successful:
#         raise(Exception("Fitting unsuccessful :-(((((("))
#     return results.get()

