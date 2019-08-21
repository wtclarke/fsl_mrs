#!/usr/bin/env python

# simu.py - Tools for simulating spectra
#         Based on Jamie Near's FID-A tools
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT

import pickle
import numpy as np
import scipy as sp
import os

H1_gamma = 42.576 

def metab_list():
    metab_file = os.path.join(os.path.dirname(__file__),'metabolites.pickle')
    with open(metab_file,"rb") as pickle_in:
        data = pickle.load(pickle_in)
        metabs = list(data.keys())
    return metabs

class MRS_sim(object):
    """
      Simulation class
      Does the job of simulating FIDs for known metabolites
    """
    

    def __init__(self,field_strength,num_points,dwelltime,linewidth,centreFreq,metab=None):
        """
           Parameters
           ----------
           field_strength : float (unit=tesla)
           num_points     : int
           dwelltime      : float (unit=seconds)
           linewidth      : float (unit=seconds)
                         FWHM of spectral peaks
           metab          : string
                        One of the following: 
        'Ala' - Alanine 
        'Asc' - Ascorbate 
        'Asp' - Aspartame
        'Cit' - Citrate
        'Cr'  - Creatine
        'GABA' - gamma-Aminobutyric Acid
        'GPC' - Glycerophosphocholine
        'GSH' - Glutathione
        'Glc' - Glucose
        'Gln' - Glutamine
        'Glu' - Glutamate
        'Gly' - Glycine
        'H2O' - Water
        'Ins' - Myo-Inositol
        'Lac' - L-Lactate
        'NAA' - N-Acetylaspartate
        'NAAG' - N-Acetylaspartylglutamate
        'PCh' - Phosphocholine
        'PCr' - Phosphocreatine
        'PE' - Phosphorylethanolamine
        'Phenyl' - 
        'Scyllo' - Scyllo-Inositol
        'Ser' - Serine
        'Tau' - Taurine
        'Tyros' - Tyrosine
        'bHB' - beta-hydroxybutyrate
        'bHG' - 2-hydroxyglutyrate
        """

        
        self.num_points     = num_points
        self.dwelltime      = dwelltime
        self.sw             = 1/dwelltime
        self.metab          = metab
        self.linewidth      = linewidth
        self.centreFreq     = centreFreq
        self.centreFreqPPM  = centreFreq*1E-6
        self.field_strength = field_strength

        if self.metab is not None:
            self.init_metab(self.metab)

    def init_metab(self,metab):
        # Init spin system and Hamiltonian
        sys = SpinSystem(metab)
        sys.shifts = [sys.shifts[k]-self.centreFreqPPM for k in range(sys.n)]
        self.H   = Hamiltonian(sys,self.field_strength)

    
    
    # Pulse sequence methods
    def PRESS(self,tau1,tau2):
        """
           [90x]--tau1/2--[180y]--(tau1+tau2)/2--[180y]--tau2/2--read
        """

        d = self.H.d # initial density matrix
        
        # Pulse sequence
        d = self.excite(d,self.H,axis='x',angle=90)
        d = self.evolve(d,self.H,tau=tau1/2)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=(tau1+tau2)/2)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=tau2/2)
        # Readout
        out = self.readout(d,rcvPhase=90)

        return out
    
    def LASER(self,te):
        """
           [90x]-te/6-[180y]-te/12-[180y]--te/12-[180y]-te/12-[180y]-te/12-[180y]-te/12-[180y]-te/6-read
        """

        d = self.H.d # initial density matrix

        tau=te/6
        
        # Pulse sequence
        d = self.excite(d,self.H,axis='x',angle=90)
        d = self.evolve(d,self.H,tau=tau/2)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=tau)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=tau)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=tau)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=tau)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=tau)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=tau/2)
        # Readout
        out = self.readout(d,rcvPhase=90)

        return out

    def SE(self,te):
        """
           [90x]--tau/2--[180y]--tau/2--read
        """
        
        d = self.H.d # initial density matrix
        
        # Pulse sequence
        d = self.excite(d,self.H,axis='x',angle=90)
        d = self.evolve(d,self.H,tau=te/2)
        d = self.rotate(d,self.H,axis='y',angle=180)
        d = self.evolve(d,self.H,tau=te/2)
        # Readout
        out = self.readout(d,rcvPhase=90)

        return out


    # GENERIC PULSE SEQUENCE METHODS
    
    def excite(self,d_in,H,axis,angle=90):
        d_out = []
        for m in range(H.sys.n):
            excite = np.zeros((2**H.sys.nspins[m],2**H.sys.nspins[m]),dtype=np.complex)
            for n in range(H.sys.nspins[m]):
                if self.H.shifts[m][n]>=30:
                    alpha=0;
                else:
                    alpha=angle*np.pi/180;
        
                if axis.lower() == 'x':
                    excite += alpha*H.Ix[m][:,:,n]
                elif axis.lower() == 'y':
                    excite += alpha*H.Iy[m][:,:,n]

            # assume conjudate symmetry
            D,U = np.linalg.eigh(excite)
        
            d1=np.diag(np.exp(-1j*D))
            d2=np.diag(np.exp(1j*D))
            
            d_out.append(U@d1@U.conj().T @ d_in[m] @ U@d2@U.conj().T)
        
        return d_out

    
    @staticmethod
    def evolve(d_in,H,tau):
        d_out = []
        for m in range(H.sys.n):
            
            # assume conjudate symmetry
            #D,U = np.linalg.eigh(H.HAB[m])        
            #d1=np.diag(np.exp(-1j*D*tau))
            #d2=np.diag(np.exp(1j*D*tau))
            #d_out.append(U@d1@U.conj().T @ d_in[m] @ U@d2@U.T)
            
            d_out.append(sp.linalg.expm(-1j*H.HAB[m]*tau)@d_in[m]@sp.linalg.expm(1j*H.HAB[m]*tau))
        
        return d_out
    
    def rotate(self,d_in,H,axis,angle):
        d_out = []
        
        for m in range(H.sys.n):
            R = np.zeros((2**H.sys.nspins[m],2**H.sys.nspins[m]),dtype=np.complex)
            
            for n in range(H.sys.nspins[m]):
                if self.H.shifts[m][n]>=30:
                    theta = 0
                else:
                    theta = angle*np.pi/180

                if axis.lower()=='x':                    
                    R += theta*H.Ix[m][:,:,n]
                elif axis.lower()=='y':
                    R += theta*H.Iy[m][:,:,n]
                elif axis.lower()=='z':
                    R += theta*H.Iz[m][:,:,n]
            
            d_out.append(sp.linalg.expm(-1j*R)@d_in[m]@sp.linalg.expm(1j*R))
        
        return d_out

    def readout(self,d_in,rcvPhase=0,shape='L'):

        # Decay
        t = np.asarray(range(self.num_points))*self.dwelltime
        if shape.upper() == 'L':
            t2    = 1/np.pi/self.linewidth
            decay = np.exp(-t/t2)
        elif shape.upper() == 'G':
            thalf=log(0.5)/(np.pi*0.5*linewidth)
            sigma=sqrt((thalf**2)/(-2*np.log(0.5)))
            decay = np.exp(-t**2/2/sigma/sigma)
        else:
            raise Exception('Unknown shape {}'.format(shape))

        # Get measurements
        phase = np.exp(1j*rcvPhase*np.pi/180)

        FID   = np.zeros(self.num_points,dtype=np.complex)
        FIDs  = []
        for m in range(self.H.sys.n):
            FIDs.append(np.zeros(self.num_points,dtype=np.complex))
            D,U = np.linalg.eig(self.H.HAB[m])
            val = 2**(2-float(self.H.sys.nspins[m]))

            Fxy   = self.H.Fx[m]+1j*self.H.Fy[m]            

            for k in range(self.num_points):
                d = np.diag(np.exp(-1j*k*self.dwelltime*D))                
                d = U@d@U.conj().T
                
                FIDs[m][k] = np.matrix.trace((d@d_in[m]@d.conj().T)@Fxy*phase)                
            
            FIDs[m] = val*FIDs[m]*decay

        for m in range(self.H.sys.n):
            FID += FIDs[m]

        return np.conj(FID)
    


class Hamiltonian(object):
    """
      Hamiltonian stuff
    """
    def __init__(self,sys,field_strength):
        """
          sys : SpinSystem object
          field_strength : float (units=tesla)                          
        """

        self.sys            = sys
        self.field_strength = field_strength

        # Hamiltonian tensors
        self.basis  = []
        self.basisA = []
        self.basisB = []
        self.Fx     = []
        self.Fy     = []
        self.Fz     = []
        self.HAB    = []
        # angular momenta
        self.Ix     = []
        self.Iy     = []
        self.Iz     = []
        # Density matrix
        self.d      = []
        # Helper
        self.diag   = []

        self.shifts = [self.sys.shifts[k] for k in range(self.sys.n)]

        
        # Convert to rad and rads
        omega0 = -2*np.pi*field_strength*H1_gamma
        
        self.sys.J_rad   = [self.sys.J[k]*2*np.pi for k in range(self.sys.n)]
        self.shifts_rads = [self.shifts[k]*omega0 for k in range(self.sys.n)]

        # Loop over subsystems to populate Hamiltonians
        for idx,n in enumerate(range(self.sys.n)):
            nspins = self.sys.nspins[n]
            states = self.create_basis_states(n=nspins)
            self.diag.append(np.eye(2**nspins))
            self.append_basis_tensors(idx=idx,n=nspins,states=states)
            self.calc_hamiltonians(idx=idx,n=nspins)
        
    def __str__(self):
        out = '------- Hamiltonian Object ---------\n'
        out += '----- Name:   {}\n'.format(self.sys.sysname)
        for n in range(self.sys.n):
            out += '----- SubName:   {}\n'.format(self.sys.subnames[n])
            out += '    HAB          = {}\n'.format(self.HAB[n].shape)
            out += '    basis        = {}\n'.format(self.basis[n].shape)
            out += '    basisA       = {}\n'.format(self.basisA[n].shape)
            out += '    basisB       = {}\n'.format(self.basisB[n].shape)
            out += '    Fx           = {}\n'.format(self.Fx[n].shape)
            out += '    Fy           = {}\n'.format(self.Fy[n].shape)
            out += '    Fz           = {}\n'.format(self.Fz[n].shape)
            out += '    Ix           = {}\n'.format(self.Ix[n].shape)
            out += '    Iy           = {}\n'.format(self.Iy[n].shape)
            out += '    Iz           = {}\n'.format(self.Iz[n].shape)
            out += '    d            = {}\n'.format(self.d[n].shape)
            out += '    shifts       = {}\n'.format(self.d[n].shape)
            out += '    shifts_rads  = {}\n'.format(self.d[n].shape)
        out += '-----------------------------------\n'
        return out

    @staticmethod
    def create_basis_states(n):
        """
           returns a matrix of all sets of possible up/down states
           matrix size is (2^n)x(n)
        """

        states = np.asarray([list('{0:0{n}b}'.format(k,n=n)) for k in range(2**n)],dtype=int)
        states[states==0]=-1
        
        return states

    def append_basis_tensors(self,idx,n,states):
        self.basis.append(np.zeros((2**n,2**n,2*n),dtype=np.complex))
        for k in range(2**n):
            for m in range(2**n):
                self.basis[idx][k,m,:n] = states[k,:]
                self.basis[idx][k,m,n:] = states[m,:]
        self.basisA.append(self.basis[idx][:,:,:n].copy())
        self.basisB.append(self.basis[idx][:,:,n:].copy())

        self.basis[idx]  *= 0.5
        self.basisA[idx] *= 0.5
        self.basisB[idx] *= 0.5

        self.Fx.append(np.zeros((2**n,2**n),dtype=np.complex))
        self.Fy.append(np.zeros((2**n,2**n),dtype=np.complex))
        self.Fz.append(np.zeros((2**n,2**n),dtype=np.complex))

        self.HAB.append(np.zeros((2**n,2**n),dtype=np.complex))

        self.Ix.append(np.zeros((2**n,2**n,n),dtype=np.complex))
        self.Iy.append(np.zeros((2**n,2**n,n),dtype=np.complex))
        self.Iz.append(np.zeros((2**n,2**n,n),dtype=np.complex))

        self.Fz[idx] = self.diag[idx]*np.sum(self.basisA[idx],axis=2)
        self.d.append(self.Fz[idx].copy()*self.sys.scaleFactor[idx])
        self.Iz[idx] = self.diag[idx][:,:,None]*self.basisA[idx]

    def calc_hamiltonians(self,idx,n):
        def match(A,B,p):
            return((A[:,:,p]==B[:,:,p+n]).astype(np.complex))

        
        B = self.basis[idx]   
        for q in range(n):            
            dFx =  .5*np.ones((2**n,2**n,n),dtype=np.complex)
            dFy = -.5*1j*np.ones((2**n,2**n,n),dtype=np.complex)
            for p in range(n):
                if p==q:        
                    dFx[:,:,p] = dFx[:,:,p]*(match(B,B+1,p)+match(B,B-1,p))
                    dFy[:,:,p] = dFy[:,:,p]*(match(B,B+1,p)-match(B,B-1,p))
                else:
                    dFx[:,:,q] = dFx[:,:,q]*(match(B,B,p))
                    dFy[:,:,q] = dFy[:,:,q]*(match(B,B,p))

                # Start filling z-component of H
                dotzcomp = self.sys.J_rad[idx][q,p]*B[:,:,q]*B[:,:,p]
                self.HAB[idx] += (self.diag[idx]*dotzcomp)

            self.Fx[idx]        += dFx[:,:,q]
            self.Fy[idx]        += dFy[:,:,q]
            self.Ix[idx][:,:,q] += dFx[:,:,q]
            self.Iy[idx][:,:,q] += dFy[:,:,q]
             

        # Resonance component
        for e in range(n):
            rescomp = self.shifts_rads[idx][e]*self.basis[idx][:,:,e]        
            self.HAB[idx] += self.diag[idx]*rescomp


        # Off-diagonal components
        for t in range(n):
            for u in range(n):
                deltaterma = np.ones((2**n,2**n),dtype=np.complex)
                deltatermb = np.ones((2**n,2**n),dtype=np.complex)

                for p in range(n):
                    if p==u:                        
                        deltaterma = deltaterma*match(B,B+1,p)
                    elif p==t:
                        deltaterma = deltaterma*match(B,B-1,p)
                    else:
                        deltaterma = deltaterma*match(B,B,p)
                        
                for q in range(n):                    
                    if q==u:                        
                        deltatermb = deltatermb*match(B,B-1,q)
                    elif q==t:
                        deltatermb = deltatermb*match(B,B+1,q)
                    else:
                        deltatermb = deltatermb*match(B,B,q)


                self.HAB[idx] += .5*self.sys.J_rad[idx][t,u]*(deltaterma+deltatermb)*(match(B,B+1,t)*match(B,B-1,u)+match(B,B-1,t)*match(B,B+1,u))
        
            
        return
        
class SpinSystem(object):
    def __init__(self,name=None):
        self.sysname        = None   # name of overall spin system
        self.subnames       = None   # list of subsystem names
        self.shifts         = None   # list of shifts
        self.J              = None   # list of J matrices
        self.scaleFactor    = None   # list of scale factors
        self.n              = None   # number of subsystems
        self.nspins         = None
        
        if name is not None:
            self.set_sys(name)


    def __str__(self):
        out = '------- SpinSystem Object ---------\n'
        out += '----- Name:   {}\n'.format(self.sysname)
        for n in range(self.n):
            out += '----- SubName:   {}\n'.format(self.subnames[n])
            out += '    J          = {}\n'.format(self.J[n].shape)
            out += '    shifts     = {}\n'.format(self.shifts[n].shape)
            out += '    scaleFator = {}\n'.format(self.scaleFactor[n])
            out += '    nspins     = {}\n'.format(self.nspins[n])
        
        out += '-----------------------------------\n'
        return out
        
    def set_sys(self,name):
        """
           Set system properties (J,shifts,scales, and names)
        """
        self.sysname = name
        
        # Pre-saved info on all metabolites (from FID-A)
        metab_file = os.path.join(os.path.dirname(__file__),'metabolites.pickle')
        with open(metab_file,"rb") as pickle_in:
            data = pickle.load(pickle_in)
            self.J             = data[name]['J']
            self.shifts        = data[name]['shifts']
            self.scaleFactor   = data[name]['scaleFactor']
            self.subnames      = data[name]['subnames']

        self.n       = len(self.J)
        self.nspins  = []
        for k in range(self.n):
            self.J[k]           = np.matrix(self.J[k],dtype=np.complex)
            self.shifts[k]      = np.asarray([self.shifts[k]],dtype=np.complex).flatten()                
            self.scaleFactor[k] = float(self.scaleFactor[k])
            self.nspins.append(self.shifts[k].size)

        # 

##########
# Below is a copy of the code used to import spin systems from Jamie's FID-A toolbox:
# import scipy.io as spio
# import glob
# import os
# import pandas as pd

# dirname = '/Users/saad/Desktop/Spectroscopy/FID-A/simulationTools/metabolites/v7'

# files = glob.glob(os.path.join(dirname,'*.mat'))

# data = {}
# for idx, file in enumerate(files):
#     mat  = spio.loadmat(file, squeeze_me=True)
#     name = os.path.split(file)[-1].replace('.mat','')
#     keys = ['subnames','shifts','J','scaleFactor']    
#     items = dict({'subnames':[],'shifts':[],'J':[],'scaleFactor':[]})

#     if mat['sys'+name].ndim>0:
#         for i in range(len(mat['sys'+name])):
#             items['J'].append(mat['sys'+name][i][0])
#             items['shifts'].append(mat['sys'+name][i][1])
#             items['subnames'].append(mat['sys'+name][i][2])
#             items['scaleFactor'].append(mat['sys'+name][i][3])
#     else:
#         items['J'].append(mat['sys'+name].tolist()[0])
#         items['shifts'].append(mat['sys'+name].tolist()[1])
#         items['subnames'].append(mat['sys'+name].tolist()[2])
#         items['scaleFactor'].append(mat['sys'+name].tolist()[3])
#     data[name] = items
    
# file = '/Users/saad/Git/fsl_mrs/fsl_mrs/utils/metabolites.pickle'
# import pickle

# pickle_out = open(file,"wb")
# pickle.dump(data, pickle_out)
# pickle_out.close()
