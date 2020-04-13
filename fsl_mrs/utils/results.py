# results.py - Collate results from MRS fits
#
# Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2020 University of Oxford 
# SHBASECOPYRIGHT

import pandas as pd
import fsl_mrs.utils.models as models
import fsl_mrs.utils.quantify as quant
import fsl_mrs.utils.qc as qc
from fsl_mrs.utils.misc import FIDToSpec,SpecToFID,calculate_crlb,calculate_lap_cov
import numpy as np

class FitRes(object):
    """
       Collects fitting results
    """
    def __init__(self,model,method,names,metab_groups,baseline_order,B,ppmlim):
        """ Short class initilisation """ 
        # Initilise some basic parameters - includes fitting options
        # Populate parameter names        
        self.fill_names(names,baseline_order=baseline_order,metab_groups=metab_groups)
        self.model        = model
        self.method        = method
        self.ppmlim       = ppmlim
        self.baseline_order = baseline_order
        self.base_poly    = B

        self.concScalings={'internal':None,'molarity':None,'molality':None}

    def loadResults(self,mrs,fitResults):
        "Load fitting results and calculate some metrics"
        # Populate data frame
        if fitResults.ndim ==1:
            self.fitResults = pd.DataFrame(data=fitResults[np.newaxis,:], columns=self.params_names)
        else:
            self.fitResults = pd.DataFrame(data=fitResults, columns=self.params_names)
        self.params = self.fitResults.mean().values
        
        #Store prediction, baseline, residual
        self.pred = self.predictedFID(mrs,mode='Full')
        self.baseline = self.predictedFID(mrs,mode='Baseline')
        self.residuals = mrs.FID - self.pred

        # Calculate single point crlb and cov
        first,last = mrs.ppmlim_to_range(self.ppmlim)
        _,_,forward,_,_ = models.getModelFunctions(self.model)
        forward_lim = lambda p : forward(p,mrs.frequencyAxis,
                                        mrs.timeAxis,
                                        mrs.basis,
                                        self.base_poly,
                                        self.metab_groups,
                                        self.g)[first:last]
        data = mrs.getSpectrum(ppmlim=self.ppmlim)
        self.crlb      = calculate_crlb(self.params,forward_lim,data)
        self.std       = np.sqrt(self.crlb)
        self.cov       = calculate_lap_cov(self.params,forward_lim,data)
        self.corr      = self.cov/(self.std[:,np.newaxis] *self.std[np.newaxis,:] )
        self.mse       = np.mean(np.abs(FIDToSpec(self.residuals)[first:last])**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.perc_SD = np.sqrt(self.crlb) / self.params*100
        self.perc_SD[self.perc_SD>999]       = 999   # Like LCModel :)
        self.perc_SD[np.isnan(self.perc_SD)] = 999

        # Calculate mcmc metrics
        if self.method == 'MH':
            self.mcmc_cov = self.fitResults.cov().values
            self.mcmc_cor = self.fitResults.corr().values
            self.mcmc_var = self.fitResults.var().values

        self.hzperppm = mrs.centralFrequency/1E6

        # Calculate QC metrics
        fwhm,snrSpec,snrPeaks,snrResidual = qc.calcQC(mrs,self,ppmlim=(0.2,4.2))
        for m,width in zip(self.metabs,fwhm):
            self.fitResults[f'fwhm_{m}'] = pd.Series(width, index=self.fitResults.index)
        for m,snr in zip(self.metabs,snrPeaks):
            self.fitResults[f'SNR_{m}'] = pd.Series(snr, index=self.fitResults.index)
        self.SNRUnapodised = snrSpec
        self.SNRFromResidual = snrResidual




    def calculateConcScaling(self,mrs,referenceMetab=['Cr','PCr'],waterRefFID=None,tissueFractions=None,TE=None,T2='Default',waterReferenceMetab='Cr',wRefMetabProtons=5,reflimits=(2,5),verbose=False):
        self.referenceMetab = referenceMetab
        self.waterReferenceMetab = waterReferenceMetab
        self.waterReferenceMetabProtons = wRefMetabProtons
        
        internalRefScaling = quant.quantifyInternal(referenceMetab,self.getConc(),self.metabs)

        if  (waterRefFID is not None) and\
            (tissueFractions is not None) and\
            (TE is not None):
            refFID = self.predictedFID(mrs,mode=waterReferenceMetab,noBaseline=True)
            if T2 == 'Default':                
                Q = quant.loadDefaultQuantificationInfo(TE,tissueFractions,mrs.centralFrequency/1E6)
            else:
                Q = quant.QuantificationInfo(TE,T2,tissueFractions)
            molalityScaling,molarityScaling = quant.quantifyWater(mrs,
                                                            waterRefFID,
                                                            refFID,
                                                            waterReferenceMetab,
                                                            self.getConc(),
                                                            self.metabs,   
                                                            wRefMetabProtons,
                                                            Q,     
                                                            reflimits=reflimits,
                                                            verbose=verbose)

            self.concScalings={'internal':internalRefScaling,'molarity':molarityScaling,'molality':molalityScaling}
        else:
            self.concScalings={'internal':internalRefScaling,'molarity':None,'molality':None}


    def predictedFID(self,mrs,mode='Full',noBaseline=False):
        if mode.lower() == 'full':
            out = models.getFittedModel(self.model,self.params,self.base_poly,self.metab_groups,mrs,noBaseline=noBaseline)
        elif mode.lower() == 'baseline':
            out = models.getFittedModel(self.model,self.params,self.base_poly,self.metab_groups,mrs,baselineOnly= True)
        elif mode in self.metabs:
            out = models.getFittedModel(self.model,self.params,self.base_poly,self.metab_groups,mrs,baselineOnly= False,noBaseline=noBaseline)
        else:
            raise ValueError('Unknown mode, must be one of: Full, baseline or a metabolite name.')
        return SpecToFID(out)

        
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
        
        
    def fill_names(self,names,baseline_order=0,metab_groups=None):
        """
        mrs            : MRS Object
        baseline_order : int
        metab_groups   : list (by default assumes single metab group)
        """
        self.metabs = names
        
        self.params_names = []
        self.params_names.extend(names)

        if metab_groups is None:
            g = 1
        else:
            g = max(metab_groups)+1

        self.g = g
        self.metab_groups = metab_groups
        
        for i in range(g):
            self.params_names.append(f'gamma_{i}')
        for i in range(g):
            self.params_names.append(f"eps_{i}")

        self.params_names.extend(['Phi0','Phi1'])
        
        for i in range(0,baseline_order+1):
            self.params_names.append(f"B_real_{i}")
            self.params_names.append(f"B_imag_{i}")


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

        if what == 'concentrations':
            df['Metab']          = mrs.names
            df['mMol/kg']        = self.conc_h2o
            df['%CRLB']          = self.perc_SD[:mrs.numBasis]
            df['/Cr']            = self.conc_cr
        elif what == 'qc':
            df['Measure'] = ['SNR']
            df['Value']   = [self.snr]
        elif what == 'parameters':
            df['Parameter'] = self.params_names
            df['Value']     = self.params

        df.to_csv(filename,index=False)


    # Functions to return physically meaningful units from the fitting results
    def getConc(self,scaling='raw',metab=None,function='mean'):
        if function is None:
            dfFunc = lambda m : self.fitResults[m].values
        else:
            dfFunc = lambda m : self.fitResults[m].apply(function)

        # Extract concentrations from parameters.
        if metab is not None:
            if metab not in self.metabs:
                raise ValueError(f'{metab} is not a recognised metabolite.')
            rawConc = np.asarray(dfFunc(metab))
        else:
            rawConc = []
            for m in self.metabs:
                rawConc.append(dfFunc(m))
            rawConc = np.asarray(rawConc)

        if scaling == 'raw':            
            return rawConc
        elif scaling == 'internal':
            if self.concScalings['internal'] is None:
                raise ValueError('Internal concetration scaling not calculated, run calculateConcScaling method.')
            return rawConc * self.concScalings['internal']

        elif scaling == 'molality':
            if self.concScalings['molality'] is None:
                raise ValueError('Molality concetration scaling not calculated, run calculateConcScaling method.')
            return rawConc * self.concScalings['molality']

        elif scaling == 'molarity':
            if self.concScalings['molarity'] is None:
                raise ValueError('Molarity concetration scaling not calculated, run calculateConcScaling method.')
            return rawConc * self.concScalings['molarity']
        else:
            raise ValueError(f'Unrecognised scaling value {scaling}.')

    def getPhaseParams(self,phi0='degrees',phi1='seconds',function='mean'):
        """Return the two phase parameters in specified units"""
        if function is None:
            p0 = self.fitResults['Phi0'].values
            p1 = self.fitResults['Phi1'].values
        else:
            p0 = self.fitResults['Phi0'].apply(function)
            p1 = self.fitResults['Phi1'].apply(function)        

        if phi0.lower() == 'degrees':
            p0 *= np.pi/180.0
        elif phi0.lower() == 'radians':
            pass 
        else:
            raise ValueError('phi0 must be degrees or radians')
        
        if phi1.lower() == 'seconds':
            p1 *= 1/(2*np.pi)
        elif phi1.lower() == 'deg_per_ppm':
            p1 *= 180.0/np.pi * self.hzperppm
        elif phi1.lower() == 'deg_per_hz':
            p1 *= 180.0/np.pi * 1.0
        else:
            raise ValueError('phi1 must be seconds or deg_per_ppm or deg_per_hz ')

        return p0,p1

    def getShiftParams(self,units='ppm',function='mean'):
        """ Return shift parameters (eps) in specified units - default = ppm."""
        if function is None:
            shiftParams = np.zeros([self.fitResults.shape[0],self.g])
            for g in range(self.g):
                shiftParams[:,g] = self.fitResults[f'eps_{g}'].values
        else:
            shiftParams = []
            for g in range(self.g):
                shiftParams.append(self.fitResults[f'eps_{g}'].apply(function))
            shiftParams = np.asarray(shiftParams)        

        if units.lower() == 'ppm':
            shiftParams *= 1/(2*np.pi*self.hzperppm)
        elif units.lower() == 'hz':
            shiftParams *= 1/(2*np.pi)
        else:
            raise ValueError('Units must be Hz or ppm.')

        return shiftParams

    def getLineShapeParams(self,units='Hz',function='mean'):
        """ Return line broadening parameters (gamma and sigma) in specified units - default = Hz."""
        if self.model=='lorentzian':
            if function is None:
                gamma = np.zeros([self.fitResults.shape[0],self.g])
                for g in range(self.g):
                    gamma[:,g] = self.fitResults[f'gamma_{g}'].values
            else:
                gamma = []
                for g in range(self.g):
                    gamma.append(self.fitResults[f'gamma_{g}'].apply(function))
                gamma = np.asarray(gamma) 

            if units.lower() =='hz':
                gamma *= 1/(np.pi)
            elif units.lower() == 'ppm':
                gamma *= 1/(np.pi*self.hzperppm)
            else:
                raise ValueError('Units must be Hz or ppm.')
            return gamma
        elif self.model=='voigt':
            raise Exception('TO DO.')

            # return gamma,sigma
        
    def getBaselineParams(self):
        """ Return normalised complex baseline parameters."""
        bParams = []
        for g in range(self.g):
            bParams.append(self.fitResults[f'B_real_{g}'].mean()+1j*self.fitResults[f'B_imag_{g}'].mean())
        bParams = np.asarray(bParams)
        with np.errstate(divide='ignore', invalid='ignore'):
            return bParams/np.abs(bParams[0])

    def getQCParams(self):
        pass