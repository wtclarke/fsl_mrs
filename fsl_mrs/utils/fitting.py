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
from fsl_mrs.core import MRS

from scipy.optimize import minimize,nnls


class FitRes(object):
    """
       Collects fitting results
    """
    def __init__(self,model):
        self.model        = model
        self.params       = None
        self.crlb         = None
        self.mcmc_samples = None
        self.mcmc_cov     = None
        self.mcmc_cor     = None
        self.mcmc_var     = None
        self.perc_SD      = None
        self.conc_h2o     = None
        self.conc_cr      = None
        self.conc_cr_pcr  = None
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
        out += "inv_gamma_sec   = {}\n".format(self.inv_gamma_sec)
        out += "eps_ppm         = {}\n".format(self.eps_ppm)
        out += "b_norm          = {}\n".format(self.b_norm)

        return out
        
        
    def fill_names(self,mrs,baseline_order=0,metab_groups=None):
        """
        mrs            : MRS Object
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

        self.params_names.extend(['Phi0','Phi1'])
        
        for i in range(baseline_order+1):
            self.params_names.extend(["B_real_{}".format(i)])

        for i in range(baseline_order+1):
            self.params_names.extend(["B_imag_{}".format(i)])

    def to_file(self,filename,mrs=None,what='concentrations'):
        """
        Save results to a csv file

        Parameters:
        -----------
        filename : str
        mrs      : MRS obj (only used if what = 'concentrations')
        what     : one of 'concentrations, 'qc', 'parameters'
        """
        import pandas as pd
        df                   = pd.DataFrame()

        if what is 'concentrations':
            df['Metab']          = mrs.names
            df['mMol/kg']        = self.conc_h2o
            df['%CRLB']          = self.perc_SD[:mrs.numBasis]
            df['/Cr']            = self.conc_cr
        elif what is 'qc':
            df['Measure'] = ['SNR']
            df['Value']   = [self.snr]
        elif what is 'parameters':
            df['Parameter'] = self.params_names
            df['Value']     = self.params

        df.to_csv(filename,index=False)

         

        
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
    Spec = FIDToSpec(FID,axis=0)
    if ppmlim is not None:
        first,last = mrs.ppmlim_to_range(ppmlim)
        Spec = Spec[first:last]
    area = np.mean(np.abs(Spec),axis=0)
    return area

def quantify(mrs,concentrations,ref='Cr',to_h2o=False,scale=1):
    """
        Quantification
    """
    if isinstance(ref,list):
        ref_con = 0
        for met in ref:
            ref_con += concentrations[mrs.names.index(met)]
    else:
        ref_con = concentrations[mrs.names.index(ref)]

    if to_h2o is not True or mrs.H2O is None:
        QuantifFactor = scale/ref_con
    else:
        ref_fid       = mrs.basis[:,mrs.names.index(ref)]
        ref_area      = calculate_area(mrs,ref_con*ref_fid)
        H2O_area      = calculate_area(mrs,mrs.H2O) 
        refH2O_ratio  = ref_area/H2O_area
        H2O_protons   = 2
        ref_protons   = num_protons[ref]
        QuantifFactor = scale*refH2O_ratio*H2O_Conc*H2O_protons/ref_protons/ref_con
    
    res = concentrations*QuantifFactor

    return res



def init_gamma_eps(mrs):
    """
       Initialise gamma/epsilon parameters
       
    """

    ppmlim = (.2,4.2)
    first,last = mrs.ppmlim_to_range(ppmlim)
    
    y = FIDToSpec(mrs.FID)[first:last]
    #y = np.real(y).flatten() #
    y = np.concatenate((np.real(y),np.imag(y)),axis=0).flatten()

    def modify_basis(mrs,gamma,eps):
        bs = mrs.basis * np.exp(-(gamma+1j*eps)*mrs.timeAxis)        
        bs = FIDToSpec(bs,axis=0)
        bs = bs[first:last,:]
        #return np.real(bs) #
        return np.concatenate((np.real(bs),np.imag(bs)),axis=0)
            
    def cf(p):
        gamma,eps = np.exp(p[0]),p[1]
        bs        = modify_basis(mrs,gamma,eps)
        #print(gamma)
        #print(eps)
        #beta,_    = nnls(bs,y)
        beta      = np.linalg.pinv(bs)@y
        pred      = bs@beta                
        loss      = np.mean((pred-y)**2)
        return loss

    x0     = np.array([-10,0])
    #bounds = [(0,1e3),(-1e3,1e3)]
    #res    = minimize(cf, x0, method='TNC',bounds=bounds)

    
    res    = minimize(cf, x0, method='Powell')
    res    = minimize(cf, x0=res.x, method='Nelder-Mead')
    res    = minimize(cf, x0=res.x, method='TNC')
    

    g,e    = np.exp(res.x[0]),res.x[1]
    
    #print([g,e])
    return g,e

def init_gamma_eps_old(mrs):
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
    res = minimize(cf, x0, method='TNC',bounds=[(0,1e5),(-np.pi,np.pi)])
    #res = minimize(cf, x0, method='Powell')
    g   = res.x[0]
    e   = res.x[1]
    
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
    #con    = np.real(np.linalg.pinv(desmat)@data)   
    con,_  = nnls(desmat,data.flatten())
    
    # Append 
    x0  = con   # concentrations
        
    g   = max(metab_groups)+1                    # number of metab groups
    x0  = np.append(x0,[gamma]*g)                # gamma[0]..
    x0  = np.append(x0,[eps]*g)                  # eps[0]..
    x0  = np.append(x0,[0,0])                    # phi0 and phi1
    x0  = np.append(x0,[0]*2*(baseline_order+1)) # baseline
    
    return x0

def init_gamma_sigma_eps(mrs):
    """
       Initialise gamma/sigma/epsilon parameters
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
        sigma = p[1]
        eps   = p[2]
        bs = blur_FID_Voigt(mrs,b,gamma,sigma)    
        bs = shift_FID(mrs,bs,eps)
        bs = extract_spectrum(mrs,bs)
        xx = 1-correlate(bs,target)
        return xx

    x0  = np.array([1,0,0])
    res = minimize(cf, x0, method='Powell')
    g   = res.x[0]
    s   = res.x[1]
    e   = res.x[2]
        
    return g,s,e

def init_FSLModel_Voigt(mrs,metab_groups,baseline_order):
    """
       Initialise params of FSLModel
    """
    # 1. Find theta, k and eps
    # 2. Use those to init concentrations
    # 3. How about phi0 and phi1?
    
    # 1. theta/k/eps
    gamma,sigma,eps = init_gamma_sigma_eps(mrs)
        
    new_basis = mrs.basis*np.exp(-(1j*eps+gamma+mrs.timeAxis*sigma**2)*mrs.timeAxis)

    data   = np.append(np.real(mrs.FID),np.imag(mrs.FID),axis=0)
    desmat = np.append(np.real(new_basis),np.imag(new_basis),axis=0)            
    con    = np.real(np.linalg.pinv(desmat)@data)   
                
    # Append 
    x0 = con
        
    g   = max(metab_groups)+1                  # number of metab groups
    x0  = np.append(x0,[gamma]*g)              # gamma[0]..
    x0  = np.append(x0,[sigma]*g)              # sigma[0]..
    x0  = np.append(x0,[eps]*g)                # eps[0]..
    x0  = np.append(x0,[0,0])                  # phi0 and phi1
    x0  = np.append(x0,[0]*2*(baseline_order+1)) # baseline
    
    return x0

def prepare_baseline_regressor(mrs,baseline_order,first,last):
    """
       Should the baseline be complex (twice the number of parameters to fit)?
    """

    B = []
    x = np.zeros(mrs.numPoints,np.complex) #0*mrs.ppmAxisShift
    x[first:last] = np.linspace(-1,1,last-first)
    
    for i in range(baseline_order+1):
        regressor  = x**i
        if i>0:
            regressor  = regressor - np.mean(regressor)
        B.append(regressor.flatten())
        B.append(1j*regressor.flatten())
    B = np.asarray(B).T
    tmp = B.copy()
    B   = 0*B
    B[first:last,:] = tmp[first:last,:].copy()
    
    return B

# OLD ONE - DELETE EVENTUALLY
# def prepare_baseline_regressor(mrs, baseline_order, first, last):
#     """
#        Should the baseline be complex (twice the number of parameters to fit)?
#     """
#     B = []
#     x = 0 * mrs.ppmAxisShift
#     x[first:last] = mrs.ppmAxisShift[first:last] - np.mean(mrs.ppmAxisShift[first:last])

#     for i in range(baseline_order + 1):
#         regressor = x ** i
#         if i > 0:
#             regressor = ztransform(regressor)
#         B.append(regressor.flatten())
#         B.append(1j * regressor.flatten())
#     B = np.asarray(B).T
#     tmp = B.copy()
#     B = 0 * B
#     B[first:last, :] = tmp[first:last, :].copy()

#     return B


def fit_FSLModel(mrs,method='Newton',ppmlim=None,baseline_order=5,metab_groups=None,model='lorentzian',x0=None):
    """
        A simplified version of LCModel
    """
    if model == 'lorentzian':
        err_func   = FSLModel_err          # error function
        grad_func  = FSLModel_grad         # gradient
        forward    = FSLModel_forward      # forward model        
        init_func  = init_FSLModel         # initilisation of params
    elif model == 'voigt':
        err_func   = FSLModel_err_Voigt     # error function
        grad_func  = FSLModel_grad_Voigt    # gradient
        forward    = FSLModel_forward_Voigt # forward model
        init_func  = init_FSLModel_Voigt    # initilisation of params
    else:
        raise Exception('Unknown model {}.'.format(model))

    data       = mrs.Spec.copy()              # data
    first,last = mrs.ppmlim_to_range(ppmlim)  # data range

    if metab_groups is None:
        metab_groups = [0]*len(mrs.names)

    # shorter names for some of the useful data
    freq,time,basis=mrs.frequencyAxis,mrs.timeAxis,mrs.basis


    # Results object
    results = FitRes(model)
    results.fill_names(mrs,baseline_order,metab_groups)

    # Prepare baseline
    B                 = prepare_baseline_regressor(mrs,baseline_order,first,last)
    results.base_poly = B
    
    # Constants
    g         = results.g
    constants = (freq,time,basis,B,metab_groups,g,data,first,last)    

    if x0 is None:
        # Initialise all params
        x0= init_func(mrs,metab_groups,baseline_order)
        # Update init baseline params
        pred = forward(x0,freq,time,basis,B,metab_groups,g)
        err  = data-pred
        br = np.linalg.pinv(np.real(B[:,0::2]))@np.real(err[:,None])
        bi = np.linalg.pinv(np.imag(B[:,1::2]))@np.imag(err[:,None])
        b0 = np.zeros(B.shape[1])
        b0[0::2] = br.flatten()
        b0[1::2] = bi.flatten()
        x0[mrs.numBasis+2*g+2:] = b0
        
    # Fitting
    if method == 'Newton':
        # Bounds        
        bnds = []
        for i in range(mrs.numBasis):
            bnds.append((0,None))
        if model == 'lorentzian':
            for i in range(g):
                bnds.append((0,None))
        elif model == 'voigt':
            for i in range(g):
                bnds.append((0,None))
            for i in range(g):
                bnds.append((0,None))
        else:
            raise Exception('Unknown model bounds.')

        for i in range(g):
            bnds.append((None,None))
        bnds.append((None,None))
        bnds.append((None,None))        
        for i in range(baseline_order+1):
            bnds.append((None,None))
            bnds.append((None,None))
        
        res = minimize(err_func, x0, args=constants, method='TNC',jac=grad_func,bounds=bnds)
        # collect results
        results.params = res.x

    elif method == 'init':
        results.params = x0
  
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
        p0   = res.params
        mask = np.ones(mrs.numBasis)
        LB   = np.zeros(mrs.numBasis)        
        for i in range(g):
            LB  = np.append(LB,0)
            mask = np.append(mask,1)
        if model == 'lorentzian':
            for i in range(g):
                LB  = np.append(LB,0)
                mask = np.append(mask,1)
        elif model == 'voigt':
            for i in range(g):
                LB  = np.append(LB,0)
                mask = np.append(mask,1)
            for i in range(g):
                LB  = np.append(LB,0)
                mask = np.append(mask,1)
        for i in range(g):
            LB  = np.append(LB,-1e10)            
            mask = np.append(mask,1)
        LB  = np.append(LB,-1e10*np.ones(2+2*(baseline_order+1)))
        mask = np.append(mask,np.zeros(2+2*(baseline_order+1)))
        
        UB   = 1e10*np.ones(len(p0))

        # Check that the values initilised by the newton method don't exceed these bounds (unlikely but possible with bad data)
        for i,(p, u, l) in enumerate(zip(p0, UB, LB)):
            if p>u:        
                p0[i]=u        
            elif p<l:
                p0[i]=l

        # Do the fitting
        mcmc    = mh.MH(loglik,logpr,burnin=100,njumps=500)
        samples = mcmc.fit(p0,LB=LB,UB=UB,verbose=False,mask=mask)

        # collect results
        results.params       = samples.mean(axis=0)
        results.mcmc_samples = samples
    else:
        raise Exception('Unknown optimisation method.')


    # Collect more results
    results.pred_spec = forward(results.params,freq,time,basis,results.base_poly,metab_groups,g)
    results.pred      = SpecToFID(results.pred_spec) # predict FID not Spec
    
    # baseline
    if model == 'lorentzian':
        con,gamma,eps,phi0,phi1,b = FSLModel_x2param(results.params,mrs.numBasis,g)
        xx       = FSLModel_param2x(0*con,gamma,eps,phi0,phi1,b)
    elif model == 'voigt':
        con,gamma,sigma,eps,phi0,phi1,b = FSLModel_x2param_Voigt(results.params,mrs.numBasis,g)
        xx       = FSLModel_param2x_Voigt(0*con,gamma,sigma,eps,phi0,phi1,b)

    baseline = forward(xx,mrs.frequencyAxis,mrs.timeAxis,mrs.basis,
                                results.base_poly,metab_groups,g)
    baseline = SpecToFID(baseline)
    results.baseline = baseline
    results.B        = b

    
    forward_lim = lambda p : forward(p,freq,time,basis,B,metab_groups,g)[first:last]
    
    
    results.crlb      = calculate_crlb(results.params,forward_lim,data[first:last])
    results.cov       = calculate_lap_cov(results.params,forward_lim,data[first:last])
    results.residuals = forward(results.params,
                                freq,time,basis,
                                B,metab_groups,g) - data    
    
    results.mse       = np.mean(np.abs(results.residuals[first:last])**2)
    results.residuals = SpecToFID(results.residuals)
    
    if results.mcmc_samples is not None:
        results.mcmc_cov = np.ma.cov(results.mcmc_samples.T)
        results.mcmc_cor = np.ma.corrcoef(results.mcmc_samples.T)
        results.mcmc_var = np.var(results.mcmc_samples,axis=0)

    
    results.perc_SD = np.sqrt(results.crlb) / results.params*100
    results.perc_SD[results.perc_SD>999]       = 999   # Like LCModel :)
    results.perc_SD[np.isnan(results.perc_SD)] = 999

    
    # LCModel-style output
    #results.snr = np.max(np.fft(results.pred-results.baseline)[first:last]) / np.sqrt(results.mse)

    # Referencing
    results.names    = mrs.names # keep metab names
    results.conc     = con
    if not mrs.H2O is None:
        results.conc_h2o = quantify(mrs,con,ref='Cr',to_h2o=True,scale=1)
    else:
        results.conc_h2o = con*0
    if 'Cr' in mrs.names:
        results.conc_cr  = quantify(mrs,con,ref='Cr',to_h2o=False,scale=1)
    if 'PCr' in mrs.names:
        results.conc_cr_pcr = quantify(mrs,con,ref=['Cr','PCr'],to_h2o=False,scale=1)
    
    # nuisance parameters in sensible units
    if model == 'lorentzian':
        con,gamma,eps,phi0,phi1,b = FSLModel_x2param(results.params,mrs.numBasis,g)
        results.inv_gamma_sec     = 1/gamma
        results.gamma             = gamma
    elif model == 'voigt':
        con,gamma,sigma,eps,phi0,phi1,b = FSLModel_x2param_Voigt(results.params,mrs.numBasis,g)
        results.inv_gamma_sec     = 1/gamma
        results.inv_sigma_sec     = 1/sigma
    results.phi0              = phi0
    results.phi1              = phi1

    results.phi0_deg          = phi0*np.pi/180.0
    results.phi1_deg_per_ppm  = phi1*np.pi/180.0 * mrs.centralFrequency / 1E6
    results.eps               = eps
    results.eps_ppm           = eps / mrs.centralFrequency * 1E6
    results.b_norm            = b/b[0]


    # QC parameters (like LCModel)
    results.snr  = np.max(np.abs(forward_lim(results.params))) / np.sqrt(results.mse)
    #results.fwhm =  ????
    
    # Save some input options as we want to know these later in the report
    results.model = model
    results.method = method

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

