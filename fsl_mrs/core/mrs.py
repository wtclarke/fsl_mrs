# MRS.py - main MRS class definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT


import warnings

from fsl_mrs.utils import misc
from fsl_mrs.utils.constants import GYRO_MAG_RATIO, PPM_SHIFT, PPM_RANGE

import numpy as np


class MRS(object):
    """
      MRS Class - container for FID, Basis, and sequence info
    """
    def __init__(self, FID=None, header=None, basis=None, names=None,
                 basis_hdr=None, H2O=None, cf=None, bw=None, nucleus=None):

        # Read in class data input
        # If there is no FID data then return empty class object.
        # Data is copied.
        if FID is not None:
            self.set_FID(FID)
        else:
            return

        self.set_H2O(H2O)

        # Set FID class attributes
        if header is not None:
            self.set_acquisition_params(
                centralFrequency=header['centralFrequency'],
                bandwidth=header['bandwidth'])

            self.set_nucleus(header=header, nucleus=nucleus)
            self.calculate_axes()

        elif (cf is not None) and (bw is not None):
            self.set_acquisition_params(
                centralFrequency=cf,
                bandwidth=bw)
            self.set_nucleus(nucleus=nucleus)
            self.calculate_axes()
        else:
            raise ValueError('You must pass a header'
                             ' or bandwidth, nucleus, and central frequency.')

        # Set Basis info
        self.set_basis(basis, names, basis_hdr)

        # Other properties that need defaults
        self.metab_groups = None
        self.scaling = {'FID': 1.0, 'basis': 1.0}

    def __str__(self):
        cf_MHz = self.centralFrequency / 1e6
        cf_T = self.centralFrequency / self.gyromagnetic_ratio / 1e6

        out = '------- MRS Object ---------\n'
        out += f'     FID.shape             = {self.FID.shape}\n'
        out += f'     FID.centralFreq (MHz) = {cf_MHz:0.3f}\n'
        out += f'     FID.nucleus           = {self.nucleus}\n'
        out += f'     FID.centralFreq (T)   = {cf_T:0.3f}\n'
        out += f'     FID.bandwidth (Hz)    = {self.bandwidth:0.1f}\n'
        out += f'     FID.dwelltime (s)     = {self.dwellTime:0.5e}\n'

        if self.basis is not None:
            out += f'     basis.shape           = {self.basis.shape}\n'
            out += f'     Metabolites           = {self.names}\n'
            out += f'     numBasis              = {self.numBasis}\n'
        out += f'     timeAxis              = {self.timeAxis.shape}\n'
        out += f'     freqAxis              = {self.frequencyAxis.shape}\n'

        return out

    def __repr__(self) -> str:
        return str(self)

    # Acquisition parameters
    def set_acquisition_params(self, centralFrequency, bandwidth):
        """
          Set useful params for fitting

          Parameters
          ----------
          centralFrequency : float  (unit=Hz)
          bandwidth : float (unit=Hz)

        """
        # Store CF in Hz
        self.centralFrequency = misc.checkCFUnits(centralFrequency)
        self.bandwidth = bandwidth
        self.dwellTime = 1 / self.bandwidth

    def set_acquisition_params_basis(self, dwelltime):
        """
           sets basis-specific timing params
        """
        # Basis has different dwelltime
        self.basis_dwellTime = dwelltime
        self.basis_bandwidth = 1 / dwelltime

        axes = misc.calculateAxes(self.basis_bandwidth,
                                  self.centralFrequency,
                                  self.numBasisPoints,
                                  self.default_ppm_shift)

        self.basis_frequencyAxis = axes['freq']
        self.basis_timeAxis = axes['time']

    def set_nucleus(self, header=None, nucleus=None):
        if nucleus is not None:
            self.nucleus = nucleus
        elif header is not None and 'ResonantNucleus' in header:
            # Look for nucleus string in header
            self.nucleus = header['ResonantNucleus']
        elif header is not None \
                and 'json' in header \
                    and 'ResonantNucleus' in header['json']:
            # Look for nucleus string in header
            self.nucleus = header['json']['ResonantNucleus']
        else:
            # Else try to infer from central frequency range
            self.nucleus = self.infer_nucleus(self.centralFrequency)

        # Set associated parameters
        self.gyromagnetic_ratio = GYRO_MAG_RATIO[self.nucleus]
        self.default_ppm_shift = PPM_SHIFT[self.nucleus]
        self.default_ppm_range = PPM_RANGE[self.nucleus]

    @staticmethod
    def infer_nucleus(cf):
        cf_MHz = cf / 1e6
        for key in GYRO_MAG_RATIO:
            onefivet_range = GYRO_MAG_RATIO[key] * np.asarray([1.445, 1.505])
            siemensthreet_range = GYRO_MAG_RATIO[key] * np.asarray([2.890, 2.895])
            threet_range = GYRO_MAG_RATIO[key] * np.asarray([2.995, 3.005])
            sevent_range = GYRO_MAG_RATIO[key] * np.asarray([6.975, 7.005])
            ninefourt_range = GYRO_MAG_RATIO[key] * np.asarray([9.35, 9.45])
            elevensevent_range = GYRO_MAG_RATIO[key] * np.asarray([11.74, 11.8])
            if (cf_MHz > onefivet_range[0] and cf_MHz < onefivet_range[1]) or \
               (cf_MHz > siemensthreet_range[0] and cf_MHz < siemensthreet_range[1]) or \
               (cf_MHz > threet_range[0] and cf_MHz < threet_range[1]) or \
               (cf_MHz > sevent_range[0] and cf_MHz < sevent_range[1]) or \
               (cf_MHz > ninefourt_range[0] and cf_MHz < ninefourt_range[1]) or \
               (cf_MHz > elevensevent_range[0] and cf_MHz < elevensevent_range[1]):
                # print(f'Identified as {key} nucleus data.'
                #      f' Esitmated field: {cf_MHz/GYRO_MAG_RATIO[key]} T.')
                return key

        raise ValueError(f'Unidentified nucleus,'
                         f' central frequency is {cf_MHz} MHz.'
                         'Pass nucleus parameter explicitly.')

    def calculate_axes(self):
        ''' Calculate axes'''
        axes = misc.calculateAxes(self.bandwidth,
                                  self.centralFrequency,
                                  self.numPoints,
                                  self.default_ppm_shift)

        self.timeAxis = axes['time']
        self.frequencyAxis = axes['freq']
        self.ppmAxis = axes['ppm']
        self.ppmAxisShift = axes['ppmshift']

        # turn into column vectors
        self.timeAxis = self.timeAxis[:, None]
        self.frequencyAxis = self.frequencyAxis[:, None]
        self.ppmAxisShift = self.ppmAxisShift[:, None]

    def get_spec(self, ppmlim=None, shift=True):
        '''Returns spectrum over defined ppm limits
           Parameters:
           -----------
           ppmlim : tuple
           shift: bool
        '''
        spectrum = misc.FIDToSpec(self.FID)
        first, last = self.ppmlim_to_range(ppmlim, shift=shift)
        return spectrum[first:last]

    def getAxes(self, axis='ppmshift', ppmlim=None):
        ''' Return x axis over defined limits
            Axis must be one of ppmshift, ppm, freq,
            or time.
        '''
        if axis.lower() == 'ppmshift':
            first, last = self.ppmlim_to_range(ppmlim, shift=True)
            return np.squeeze(self.ppmAxisShift[first:last])
        elif axis.lower() == 'ppm':
            first, last = self.ppmlim_to_range(ppmlim, shift=False)
            return np.squeeze(self.ppmAxis[first:last])
        elif axis.lower() == 'freq':
            first, last = self.ppmlim_to_range(ppmlim, shift=False)
            return np.squeeze(self.frequencyAxis[first:last])
        elif axis.lower() == 'time':
            return np.squeeze(self.timeAxis)
        else:
            raise ValueError('axis must be one of ppmshift, '
                             'ppm, freq or time.')

    def ppmlim_to_range(self, ppmlim=None, shift=True):
        """
           turns ppmlim into data range

           Parameters:
           -----------

           ppmlim : tuple

           Outputs:
           --------

           int : first position
           int : last position
        """
        if ppmlim is not None:
            def ppm2range(x, shift):
                if shift:
                    return np.argmin(np.abs(self.ppmAxisShift - x))
                else:
                    return np.argmin(np.abs(self.ppmAxis - x))

            first = ppm2range(ppmlim[0], shift)
            last = ppm2range(ppmlim[1], shift)
            if first > last:
                first, last = last, first
        else:
            first, last = 0, self.numPoints

        return int(first), int(last)

    def resample_basis(self):
        """
           Usually the basis is simulated using different
           timings and/or number of points.
           This interpolates the basis to match the FID
        """
        self.basis = misc.ts_to_ts(self.basis,
                                   self.basis_dwellTime,
                                   self.dwellTime,
                                   self.numPoints)
        self.basis_dwellTime = self.dwellTime
        self.basis_bandwidth = 1 / self.dwellTime
        self.numBasisPoints = self.numPoints

    def processForFitting(self, ppmlim=(.2, 4.2), ind_scaling=None):
        """ Apply rescaling and run the conjugation checks"""
        self.check_FID(ppmlim=ppmlim, repair=True)
        self.check_Basis(ppmlim=ppmlim, repair=True)
        self.rescaleForFitting(ind_scaling=ind_scaling)

    def rescaleForFitting(self, scale=100, ind_scaling=None):
        """ Apply rescaling across data, basis and H20
            If ind_scaling is specified individual basis spectra
            can be rescaled independently (useful for MM).
        """

        scaledFID, scaling = misc.rescale_FID(self.FID, scale=scale)
        self.set_FID(scaledFID)
        # Apply the same scaling to the water FID.
        if self.H2O is not None:
            self.H2O *= scaling

        # Scale basis
        if self.basis is not None:
            if ind_scaling is None:
                self.basis, scaling_basis = misc.rescale_FID(self.basis,
                                                             scale=scale)
            else:
                index = [self.names.index(n) for n in ind_scaling]
                # First scale all those not selected together.
                mask = np.zeros_like(self.names, dtype=bool)
                mask[index] = True
                self.basis[:, ~mask], scaling_basis = misc.rescale_FID(
                    self.basis[:, ~mask],
                    scale=scale)
                scaling_basis = [scaling_basis]
                # Then loop over basis spec to independently scale
                for idx in index:
                    self.basis[:, idx], tmp = misc.rescale_FID(
                        self.basis[:, idx],
                        scale=scale)
                    scaling_basis.append(tmp)
        else:
            scaling_basis = None

        self.scaling = {'FID': scaling, 'basis': scaling_basis}

    def check_FID(self, ppmlim=(.2, 4.2), repair=False):
        """
           Check if FID needs to be conjugated
           by looking at total power within ppmlim range

        Parameters
        ----------
        ppmlim : list
        repair : if True applies conjugation to FID

        Returns
        -------
        0 if check successful and -1 if not (also issues warning)

        """
        first, last = self.ppmlim_to_range(ppmlim)
        Spec1 = np.real(misc.FIDToSpec(self.FID))[first:last]
        Spec2 = np.real(misc.FIDToSpec(np.conj(self.FID)))[first:last]

        if np.linalg.norm(misc.detrend(Spec1, deg=4)) < \
           np.linalg.norm(misc.detrend(Spec2, deg=4)):
            if repair is False:
                warnings.warn('YOU MAY NEED TO CONJUGATE YOUR FID!!!')
                return -1
            else:
                self.conj_FID()
                return 1

        return 0

    def conj_FID(self):
        """
        Conjugate FID
        """
        self.FID = np.conj(self.FID)

    def check_Basis(self, ppmlim=(.2, 4.2), repair=False):
        """
           Check if Basis needs to be conjugated
           by looking at total power within ppmlim range

        Parameters
        ----------
        ppmlim : list
        repair : if True applies conjugation to basis

        Returns
        -------
        0 if check successful and -1 if not (also issues warning)

        """
        first, last = self.ppmlim_to_range(ppmlim)

        conjOrNot = []
        for b in self.basis.T:
            Spec1 = np.real(misc.FIDToSpec(b))[first:last]
            Spec2 = np.real(misc.FIDToSpec(np.conj(b)))[first:last]
            if np.linalg.norm(misc.detrend(Spec1, deg=4)) < \
               np.linalg.norm(misc.detrend(Spec2, deg=4)):
                conjOrNot.append(1.0)
            else:
                conjOrNot.append(0.0)

        if (sum(conjOrNot) / len(conjOrNot)) > 0.5:
            if repair is False:
                warnings.warn('YOU MAY NEED TO CONJUGATE YOUR BASIS!!!')
                return -1
            else:
                self.conj_Basis()
                return 1

        return 0

    def conj_Basis(self):
        """
        Conjugate FID and recalculate spectrum
        """
        self.basis = np.conj(self.basis)

    def ignore(self, metabs):
        """
          Ignore a subset of metabolites by removing them from the basis

          Parameters
          ----------

          metabs: list
        """
        if self.basis is None:
            raise Exception('You must first specify a basis before ignoring a'
                            ' subset of it!')

        if metabs is None:
            return

        for m in metabs:
            if m not in self.names:
                raise ValueError(f'{m} not in current list of metabolites'
                                 f' ({self.names}).')

            names = np.asarray(self.names)
            index = names == m
            self.names = names[~index].tolist()
            self.basis = self.basis[:, ~index]

        self.numBasis = len(self.names)

    def keep(self, metabs):
        """
          Keep a subset of metabolites by removing all others from basis

          Parameters
          ----------

          metabs: list
        """
        if self.basis is None:
            raise Exception('You must first specify a basis before keeping a'
                            ' subset of it!')

        if metabs is None:
            return

        for m in metabs:
            if m not in self.names:
                raise ValueError(f'{m} not in current list of metabolites'
                                 f' ({self.names}).')

        metabs = [m for m in self.names if m not in metabs]
        self.ignore(metabs)

    def add_peak(self, ppm, amp, name, gamma=0, sigma=0):
        """
           Add Voigt peak to basis at specified ppm
        """

        peak = misc.create_peak(self, ppm, amp, gamma, sigma)[:, None]
        self.basis = np.append(self.basis, peak, axis=1)
        self.names.append(name)
        self.numBasis += 1

    def add_MM_peaks(self, ppmlist=None, amplist=None, gamma=0, sigma=0):
        """
           Add macromolecule list

        Parameters
        ----------

        ppmlist : default is [0.9,1.2,1.4,1.7,[2.08,2.25,1.95,3.0]]
        amplist : default is [3.0,2.0,2.0,2.0,[1.33,0.33,0.33,0.4]]

        gamma,sigma : float parameters of Voigt blurring
        """
        if ppmlist is None:
            ppmlist = [0.9, 1.2, 1.4, 1.7, [2.08, 2.25, 1.95, 3.0]]
            amplist = [3.0, 2.0, 2.0, 2.0, [1.33, 0.33, 0.33, 0.4]]

        for idx, _ in enumerate(ppmlist):
            if isinstance(ppmlist[idx], (float, int)):
                ppmlist[idx] = [float(ppmlist[idx]), ]
            if isinstance(amplist[idx], (float, int)):
                amplist[idx] = [float(amplist[idx]), ]

        names = [f'MM{i[0]*10:02.0f}' for i in ppmlist]

        for name, ppm, amp in zip(names, ppmlist, amplist):
            self.add_peak(ppm, amp, name, gamma, sigma)

        return len(ppmlist)

    def add_expt_MM(self, lw=5):
        """
           Add experimental MM basis derived from AA residues

        Parameters
        ----------

        lw : Linewidth of basis spectra
        """
        from fsl_mrs.mmbasis.mmbasis import getMMBasis
        warnings.warn('This is an experimental feature!')

        basisFIDs, names = getMMBasis(self, lw=lw, shift=True)
        for basis, n in zip(basisFIDs, names):
            self.basis = np.append(self.basis, basis[:, np.newaxis], axis=1)
            self.names.append('MM_' + n)
            self.numBasis += 1

    def set_FID(self, FID):
        """
          Sets the FID
        """
        if FID.ndim > 1:
            raise ValueError(f'MRS objects only handle one FID at a time.'
                             f' FID shape is {FID.shape}.')
        self.FID = FID.copy()
        self.numPoints = self.FID.size

    def set_H2O(self, FID):
        """
          Sets the water FID
        """

        if FID is None:
            self.H2O = None
            self.numPoints_H2O = None
            return

        if FID.ndim > 1:
            raise ValueError(f'MRS objects only handle one FID at a time.'
                             f' H2O FID shape is {FID.shape}.')
        self.H2O = FID.copy()
        self.numPoints_H2O = FID.size

    def set_basis(self, basis, names, basis_hdr):
        ''' Set basis in MRS class object '''
        if basis is not None:

            # Check for duplicate names
            for name in names:
                dupes = [idx for idx, n in enumerate(names) if n == name]
                if len(dupes) > 1:
                    for idx, ddx in enumerate(dupes[1:]):
                        names[ddx] = names[ddx] + f'_{idx+1}'
                        print(f'Found duplicate basis name "{name}", renaming to "{names[ddx]}".')

            self.basis = basis.copy()
            # Handle single basis spectra
            if self.basis.ndim == 1:
                self.basis = self.basis[:, np.newaxis]

            # Assume that there will always be more
            # timepoints than basis spectra.
            if self.basis.shape[0] < self.basis.shape[1]:
                self.basis = self.basis.T
            self.numBasis = self.basis.shape[1]
            self.numBasisPoints = self.basis.shape[0]

            if (names is not None) and (basis_hdr is not None):
                self.names = names.copy()
                self.set_acquisition_params_basis(1 / basis_hdr['bandwidth'])
            else:
                raise ValueError('Pass basis names and header with basis.')

            # Now interpolate the basis to the same time axis.
            self.resample_basis()

        else:
            self.basis = None
            self.names = None
            self.numBasis = None
            self.basis_dwellTime = None
            self.basis_bandwidth = None

    def plot(self, ppmlim=(0.2, 4.2)):
        from fsl_mrs.utils.plotting import plot_spectrum
        plot_spectrum(self, ppmlim=ppmlim)

    def plot_ref(self, ppmlim=(2.65, 6.65)):
        from fsl_mrs.utils.plotting import plot_spectrum
        plot_spectrum(self, FID=self.H2O, ppmlim=ppmlim)

    def plot_fid(self, tlim=None):
        from fsl_mrs.utils.plotting import plot_fid
        plot_fid(self, tlim)

    def plot_basis(self, add_spec=False, ppmlim=(0.2, 4.2)):
        from fsl_mrs.utils.plotting import plot_basis
        plot_basis(self, plot_spec=add_spec, ppmlim=ppmlim)
