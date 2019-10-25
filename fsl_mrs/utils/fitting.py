#!/usr/bin/env python

# fitting.py - Fit MRS models
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT


from fsl_mrs.utils.models import *
from fsl_mrs.utils.misc import *
from fsl_mrs.utils.constants import *
from fsl_mrs.utils import mh

from scipy.optimize import minimize


class FitRes(object):
    """
       Collects fitting results
    """
    def __init__(self):
        self.params       = None
        self.crlb         = None
        self.mcmc_samples = None
        self.mcmc_cov     = None
        self.mcmc_cor     = None
        self.mcmc_var     = None
        self.perc_SD      = None
        self.conc_h2o     = None
        self.conc_cr      = None
        self.cov          = None
        self.params_names = None
        self.residuals    = None
        self.mse          = None
        self.base_poly    = None
        self.pred         = None
        self.g            = None
        self.metab_groups = None
        self.residuals    = None
        
    def __str__(self):
        out  = "----- Fitting Results ----\n"
        out += " names          = {}\n".format(self.params_names)
        out += " params         = {}\n".format(self.params)
        out += " CRLB           = {}\n".format(self.crlb)
        out += " MSE            = {}\n".format(self.mse)
        #out += " cov            = {}\n".format(self.cov)
        out += " phi0 (deg)     = {}\n".format(self.phi0_deg)
        out += " phi1 (deg/ppm) = {}\n".format(self.phi1_deg_per_ppm)
        out += "inv_gamma_ms    = {}\n".format(self.inv_gamma_sec)
        out += "eps_ppm         = {}\n".format(self.eps_ppm)
        out += "b_norm          = {}\n".format(self.b_norm)

        return out
        
        
    def fill_names(self,mrs,baseline_order=0,metab_groups=None):
        """
        mrs : MRS Object
        baseline_order : int
        metab_groups   : list (by default assumes single metab group)
        """
        self.metabs = mrs.names
        
        self.params_names = []
        self.params_names.extend(mrs.names)

        if metab_groups is None:
            g = 1
        else:
            g = max(metab_groups)+1

        self.g = g
        self.metab_groups = metab_groups
        
        for i in range(g):
            self.params_names.extend(["gamma_{}".format(i)])
        for i in range(g):
            self.params_names.extend(["eps_{}".format(i)])

        for i in range(baseline_order+1):
            self.params_names.extend(["B_{}".format(i)])

    def to_file(self,mrs,filename):
        import pandas as pd
        df = pd.DataFrame()
        df['Metab']          = mrs.names
        df['mMol/kg']        = self.conc_h2o
        df['%CRLB']          = self.perc_SD[:mrs.numBasis]
        df['/Cr']            = self.conc_cr
        df.to_csv(filename)

        

def print_params(x,mrs,metab_groups,ref_metab='Cr',scale_factor=1):
    """
       Print parameters 
    """
    g = max(metab_groups)+1
    con,gamma,eps,phi0,phi1,b=FSLModel_x2param(x,mrs.numBasis,g)
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


def calculate_area(mrs,FID,ppmlim=None):
    """
        Calculate area 
    """
    Spec = np.fft.fft(FID,axis=0)
    if ppmlim is not None:
        first,last = mrs.ppmlim_to_range(ppmlim)
        Spec = Spec[first:last]
    area = np.mean(np.abs(Spec),axis=0)
    return area

def quantify(mrs,concentrations,ref='Cr',to_h2o=False,scale=1):
    """
        Quantification
    """
    ref_con = concentrations[mrs.names.index(ref)]

    if to_h2o is not True:
        QuantifFactor = scale/ref_con
    else:
        ref_fid       = mrs.basis[:,mrs.names.index(ref)]
        ref_area      = calculate_area(mrs,ref_con*ref_fid)
        H2O_area      = calculate_area(mrs,mrs.H2O) 
        refH2O_ratio   = ref_area/H2O_area
        H2O_protons   = 2
        ref_protons   = num_protons[ref]
        QuantifFactor = scale*refH2O_ratio*H2O_Conc*H2O_protons/ref_protons/ref_con
    
    res = concentrations*QuantifFactor

    return res

def init_gamma_eps(mrs):
    """
       Initialise gamma/epsilon parameters
       This is done by summing all the basis FIDs and
       maximizing the correlation with the data FID
       after shifting and blurring
       correlation is calculated in the range [.2,4.2] ppm
    """
    target = mrs.FID[:,None]
    target = extract_spectrum(mrs,target)
    b      = np.sum(mrs.basis,axis=1)[:,None]
    def cf(p):
        gamma = p[0]
        eps   = p[1]
        bs = blur_FID(mrs,b,gamma)    
        bs = shift_FID(mrs,bs,eps)
        bs = extract_spectrum(mrs,bs)
        xx = 1-correlate(bs,target)
        return xx

    x0  = np.array([0,0])
    res = minimize(cf, x0, method='Powell')
    g   = res.x[0]
    e   = res.x[1]
    
    # shifts = np.linspace(-1e2,1e2,500)
    # t2blur = 50e-3    # seconds
    # gamma  = 1/t2blur # Hz
    
    # if basis is None:
    #     basis = np.sum(mrs.basis,axis=1)[:,None]
    
    # g = []
    # e = []
    
    # for i in range(basis.shape[1]):
    #     b = basis[:,i][:,None]
    #     x = []
    #     for shift in shifts:
    #         bs = blur_FID(mrs,b,gamma)    
    #         bs = shift_FID(mrs,bs,shift)
    #         bs = extract_spectrum(mrs,bs)
    #         xx = correlate(bs,target)
    #         x.append(xx)
    #     x = np.array(x)
    #     eps = shifts[np.argmax(x)]
    #     e.append(eps)
    #     g.append(gamma)
    # e = np.array(e)
    # g = np.array(g)
    # print('Initial value for gamma/eps = {},{}'.format(g,e))
    
    return g,e

def init_FSLModel(mrs,metab_groups,baseline_order):
    """
       Initialise params of FSLModel
    """
    # 1. Find gamma and eps
    # 2. Use those to init concentrations
    # 3. How about phi0 and phi1?
    
    # 1. gamma/eps
    gamma,eps = init_gamma_eps(mrs)
        
    new_basis = mrs.basis*np.exp(-(gamma+1j*eps)*mrs.timeAxis)    
    
    data   = np.append(np.real(mrs.FID),np.imag(mrs.FID),axis=0)
    desmat = np.append(np.real(new_basis),np.imag(new_basis),axis=0)            
    con    = np.real(np.linalg.pinv(desmat)@data)   
                
    # Append 
    x0 = con
        
    
    #  Zero for the rest        
    g   = max(metab_groups)+1                  # number of metab groups
    x0  = np.append(x0,[gamma]*g)                  # gamma[0]..
    x0  = np.append(x0,[eps]*g)                  # eps[0]..
    x0  = np.append(x0,[0,0])                  # phi0 and phi1
    x0  = np.append(x0,[0]*(baseline_order+1)) # baseline
    
    return x0

def prepare_baseline_regressor(mrs,baseline_order,first,last):
    """
       Should the baseline be complex (twice the number of parameters to fit)?
    """
    B = []
    x = 0*mrs.ppmAxisShift
    x[first:last] = mrs.ppmAxisShift[first:last]-np.mean(mrs.ppmAxisShift[first:last])
    
    for i in range(baseline_order+1):
        regressor  = x**i           
        if i>0:
            regressor = ztransform(regressor)
        B.append(regressor.flatten())
    B = np.asarray(B).T
    tmp = B.copy()
    B   = 0*B
    B[first:last,:] = tmp[first:last,:].copy()
    
    return B

def fit_FSLModel(mrs,method='Newton',ppmlim=None,baseline_order=5,metab_groups=None):
    """
        A simplified version of LCModel
    """
    err_func   = FSLModel_err          # error function
    grad_func  = FSLModel_grad         # gradient
    forward    = FSLModel_forward      # forward model
    data       = mrs.Spec.copy()              # data
    first,last = mrs.ppmlim_to_range(ppmlim)  # data range

    if metab_groups is None:
        metab_groups = [0]*len(mrs.names)

    # shorter names for some of the useful data
    freq,time,basis=mrs.frequencyAxis,mrs.timeAxis,mrs.basis


    # Results object
    results = FitRes()
    results.fill_names(mrs,baseline_order,metab_groups)
    
    # Initialise all params
    x0= init_FSLModel(mrs,metab_groups,baseline_order)

    # Prepare baseline
    B                 = prepare_baseline_regressor(mrs,baseline_order,first,last)
    results.base_poly = B

    # Constants
    g         = results.g
    constants = (freq,time,basis,B,metab_groups,g,data,first,last)    
    
    # Fitting
    if method == 'Newton':
        # Bounds        
        bnds = []
        for i in range(mrs.numBasis):
            bnds.append((0,None))
        for i in range(g):
            bnds.append((0,None))
        for i in range(g):
            bnds.append((None,None))
        bnds.append((None,None))
        bnds.append((None,None))        
        for i in range(baseline_order+1):
            bnds.append((None,None))
        
        res = minimize(err_func, x0, args=constants, method='TNC',jac=grad_func,bounds=bnds)
        # collect results
        results.params = res.x

  
    elif method == 'MH':
        forward_mh = lambda p : forward(p,freq,time,basis,B,metab_groups,g)
        numPoints_over_2  = (last-first)/2.0
        y      = data[first:last]
        loglik = lambda  p : np.log(np.linalg.norm(y-forward_mh(p)[first:last]))*numPoints_over_2
        logpr  = lambda  p : 0 

        # Setup the fitting
        # Init with nonlinear fit
        res  = fit_FSLModel(mrs,method='Newton',ppmlim=ppmlim,
                            metab_groups=metab_groups,baseline_order=baseline_order)
        p0   = res.params
        mask = np.ones(mrs.numBasis)
        LB   = np.zeros(mrs.numBasis)        
        for i in range(g):
            LB  = np.append(LB,0)
            mask = np.append(mask,1)
        for i in range(g):
            LB  = np.append(LB,-1e10)            
            mask = np.append(mask,1)
        LB  = np.append(LB,-1e10*np.ones(2+baseline_order+1))
        mask = np.append(mask,np.zeros(2+baseline_order+1))
        
        UB   = 1e10*np.ones(len(p0))
        # Do the fitting
        mcmc    = mh.MH(loglik,logpr,burnin=100,njumps=500)
        samples = mcmc.fit(p0,LB=LB,UB=UB,verbose=False,mask=mask)

        # collect results
        results.params       = samples.mean(axis=0)
        results.mcmc_samples = samples
    else:
        raise Exception('Unknown optimisation method.')


    # Collect more results
    results.pred = FSLModel_forward(results.params,freq,time,basis,results.base_poly,metab_groups,g)
    results.pred = np.fft.ifft(results.pred) # predict FID not Spec

    # baseline
    con,gamma,eps,phi0,phi1,b = FSLModel_x2param(results.params,mrs.numBasis,g)
    xx       = FSLModel_param2x(0*con,gamma,eps,phi0,phi1,b)
    baseline = FSLModel_forward(xx,mrs.frequencyAxis,mrs.timeAxis,mrs.basis,
                                results.base_poly,metab_groups,g)
    baseline = np.fft.ifft(baseline)
    results.baseline = baseline

    
    forward = lambda p : FSLModel_forward(p,freq,time,basis,B,metab_groups,g)[first:last]
    
    
    results.crlb      = calculate_crlb(results.params,forward,data[first:last])
    results.cov       = calculate_lap_cov(results.params,forward,data[first:last])
    results.residuals = FSLModel_forward(results.params,
                                         freq,time,basis,
                                         B,metab_groups,g) - data    
    
    results.mse       = np.mean(np.abs(results.residuals[first:last])**2)
    results.residuals = np.fft.ifft(results.residuals)
    
    if results.mcmc_samples is not None:
        results.mcmc_cov = np.ma.cov(results.mcmc_samples.T)
        results.mcmc_cor = np.ma.corrcoef(results.mcmc_samples.T)
        results.mcmc_var = np.var(results.mcmc_samples,axis=0)
    
    results.perc_SD = np.sqrt(results.crlb) / results.params*100

    # LCModel-style output
    #results.snr = np.max(np.fft(results.pred-results.baseline)[first:last]) / np.sqrt(results.mse)

    # Referencing
    results.conc_h2o = quantify(mrs,con,ref='Cr',to_h2o=True,scale=1)
    results.conc_cr  = quantify(mrs,con,ref='Cr',to_h2o=False,scale=1)

    # nuisance parameters in sensible units
    con,gamma,eps,phi0,phi1,b = FSLModel_x2param(results.params,mrs.numBasis,g)
    results.phi0_deg          = phi0*np.pi/180.0
    results.phi1_deg_per_ppm  = phi1*np.pi/180.0 * mrs.centralFrequency / 1E6
    results.inv_gamma_sec     = 1/gamma
    results.eps_ppm           = eps / mrs.centralFrequency * 1E6
    results.b_norm            = b/b[0]


    # QC parameters (like LCModel)
    results.snr  = np.max(np.abs(forward(results.params))) / np.sqrt(results.mse)
    #results.fwhm =  ????
    

    return results
