# MRS.py - main MRS class definition
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         Will Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT


import warnings

from copy import deepcopy

from fsl_mrs.utils import misc
from fsl_mrs.utils.constants import GYRO_MAG_RATIO, PPM_SHIFT, PPM_RANGE
from fsl_mrs.core.basis import Basis

import numpy as np


class MRS(object):
    """
      MRS Class - The basic unit for fitting. Encapsulates a single spectrum, the basis spectra,
      and water reference information required to carry out fitting.
    """
    def __init__(self, FID=None, header=None, basis=None, names=None,
                 basis_hdr=None, H2O=None, cf=None, bw=None, nucleus=None):
        """Main init for the MRS class

        :param FID: [description], defaults to None
        :type FID: [type], optional
        :param header: [description], defaults to None
        :type header: [type], optional
        :param basis: [description], defaults to None
        :type basis: [type], optional
        :param names: [description], defaults to None
        :type names: [type], optional
        :param basis_hdr: [description], defaults to None
        :type basis_hdr: [type], optional
        :param H2O: [description], defaults to None
        :type H2O: [type], optional
        :param cf: [description], defaults to None
        :type cf: [type], optional
        :param bw: [description], defaults to None
        :type bw: [type], optional
        :param nucleus: [description], defaults to None
        :type nucleus: [type], optional
        :raises ValueError: [description]
        :raises TypeError: [description]
        """
        # properties that need defaults
        self._fid_scaling = 1.0
        self._keep = []
        self._ignore = []
        self._keep_ignore = []
        self._conj_basis = False
        self._conj_fid = False
        self._scaling_factor = None
        self._indept_scale = []

        # Read in class data input
        self.FID = FID
        self.H2O = H2O

        # Set FID class attributes
        if header is not None:
            self.set_acquisition_params(
                header['centralFrequency'],
                header['bandwidth'])

            self.set_nucleus(header=header, nucleus=nucleus)
            self._calculate_axes()

        elif (cf is not None) and (bw is not None):
            self.set_acquisition_params(
                cf,
                bw)
            self.set_nucleus(nucleus=nucleus)
            self._calculate_axes()
        else:
            raise ValueError('You must pass a header'
                             ' or bandwidth, nucleus, and central frequency.')

        # Set Basis info
        # After refactor still handle the old syntax of basis, names, headers
        # But also handle a Basis obejct
        if basis is not None:
            if isinstance(basis, np.ndarray):
                self.basis = Basis(basis, names, basis_hdr)
            elif isinstance(basis, Basis):
                self.basis = deepcopy(basis)
            else:
                raise TypeError('Basis must be a numpy array (+ names & headers) or a fsl_mrs.core.Basis object.')
        else:
            self._basis = None

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

    # Properties
    @property
    def FID(self):
        """Returns the FID"""
        if self._conj_fid:
            return self._FID.conj() * self._fid_scaling
        else:
            return self._FID * self._fid_scaling

    @FID.setter
    def FID(self, FID):
        """
          Sets the FID and resets the FID scaling
        """
        if FID.ndim > 1:
            raise ValueError(f'MRS objects only handle one FID at a time.'
                             f' FID shape is {FID.shape}.')
        self._FID = FID.copy()
        self._fid_scaling = 1.0

    @property
    def numPoints(self):
        return self.FID.size

    @property
    def H2O(self):
        """Returns the reference FID"""
        if self._H2O is None:
            return None

        if self._conj_fid:
            return self._H2O.conj() * self._fid_scaling
        else:
            return self._H2O * self._fid_scaling

    @H2O.setter
    def H2O(self, FID):
        """
          Sets the water reference FID
        """
        if FID is None:
            self._H2O = None
            self.numPoints_H2O = None
            return

        if FID.ndim > 1:
            raise ValueError(f'MRS objects only handle one FID at a time.'
                             f' H2O FID shape is {FID.shape}.')
        self._H2O = FID.copy()
        self.numPoints_H2O = FID.size

    @property
    def centralFrequency(self):
        return self._cf

    @centralFrequency.setter
    def centralFrequency(self, cf):
        # Store CF in Hz
        self._cf = misc.checkCFUnits(cf)

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bw):
        # Store bandwidth in Hz
        self._bandwidth = bw

    @property
    def dwellTime(self):
        return 1 / self._bandwidth

    @dwellTime.setter
    def dwellTime(self, dt):
        # Store dwellTime as bandwidth in Hz
        self._bandwidth = 1 / dt

    @property
    def nucleus(self):
        return self._nucleus

    @property
    def conj_FID(self):
        """Conjugation state of FID"""
        return self._conj_fid

    @conj_FID.setter
    def conj_FID(self, value):
        """Set conjugation state of FID

        :param value: True or False
        :type value: Bool
        """
        self._conj_fid = value

    @property
    def basis(self):
        """Returns the currently formatted basis spectra"""
        if self._basis is None:
            return None
        else:
            if self._conj_basis:
                return self._basis.get_formatted_basis(
                    self.bandwidth,
                    self.numPoints,
                    ignore=self._keep_ignore,
                    scale_factor=self._scaling_factor,
                    indept_scale=self._indept_scale).conj()
            else:
                return self._basis.get_formatted_basis(
                    self.bandwidth,
                    self.numPoints,
                    ignore=self._keep_ignore,
                    scale_factor=self._scaling_factor,
                    indept_scale=self._indept_scale)

    @basis.setter
    def basis(self, basis):
        ''' Set basis in MRS class object '''
        if isinstance(basis, Basis):
            self._basis = basis
        elif basis is None:
            self._basis = None
        else:
            raise TypeError('Basis must be None or a fsl_mrs.core.basis.Basis object.')

    @property
    def conj_Basis(self):
        """Conjugation state of basis spectra"""
        return self._conj_basis

    @conj_Basis.setter
    def conj_Basis(self, value):
        """Set conjugation state of basis spectra

        :param value: True or False
        :type value: Bool
        """
        self._conj_basis = value

    @property
    def names(self):
        """Return the names of the basis spectra currently configured."""
        if self._basis is None:
            return None
        else:
            return self._basis.get_formatted_names(self._keep_ignore)

    @property
    def numBasis(self):
        """Returns the number of currently configured basis spectra"""
        if self._basis is None:
            return None
        else:
            return len(self._basis.get_formatted_names(self._keep_ignore))

    @property
    def keep(self):
        return self._keep

    @keep.setter
    def keep(self, metabs):
        """Keep a subset of metabolites in the basis by ignoring all others.

        :param metabs: List of metabolite names to keep.
        :type metabs: List of str
        """
        if metabs is None or metabs == []:
            self._keep = []
            self._keep_ignore = []
            if self._ignore != []:
                self.ignore = self._ignore
            return

        for m in metabs:
            if m not in self.names:
                raise ValueError(f'{m} not in current list of metabolites'
                                 f' ({self.names}).')

        self._keep = metabs
        self._keep_ignore += [m for m in self.names if m not in metabs]

    @property
    def ignore(self):
        return self._ignore

    @ignore.setter
    def ignore(self, metabs):
        """Ignore a subset of metabolites in the basis

        :param metabs: List of metabolite names to remove
        :type metabs: List of str
        """
        if metabs is None or metabs == []:
            self._ignore = []
            self._keep_ignore = []
            if self._keep != []:
                self.keep = self._keep
            return

        for m in metabs:
            if m not in self.names:
                raise ValueError(f'{m} not in current list of metabolites'
                                 f' ({self.names}).')
        self._ignore += metabs
        self._keep_ignore += metabs

    @property
    def fid_scaling(self):
        """Scaling applied to the FID"""
        return self._fid_scaling

    @fid_scaling.setter
    def fid_scaling(self, scale):
        """Set scaling applied to the FID"""
        self._fid_scaling = scale

    @property
    def basis_scaling_target(self):
        """Scaling target for basis"""
        return self._scaling_factor

    @basis_scaling_target.setter
    def basis_scaling_target(self, scale):
        """Set ccaling target for basis"""
        self._scaling_factor = scale

    @property
    def basis_scaling(self):
        """Scaling applied to the Basis"""
        if self._basis is not None:
            return self._basis.get_rescale_values(
                self.bandwidth,
                self.numPoints,
                ignore=self._keep_ignore,
                scale_factor=self._scaling_factor,
                indept_scale=self._indept_scale)
        else:
            return [None]

    @property
    def scaling(self):
        """Return the scaling information as dict.
        Mostly for backwards compatibility, use .fid_scaling
        and .basis_scaling for new code.
        """
        return {'FID': self.fid_scaling,
                'basis': self.basis_scaling[0]}

    # Get methods
    def get_spec(self, ppmlim=None, shift=True):
        """Returns spectrum over defined ppm limits

        :param ppmlim: Chemical shift range over which to retun the spectrum, defaults to None
        :type ppmlim: 2-tuple of floats, optional
        :param shift: Applies referenciing shift if True, defaults to True
        :type shift: bool, optional
        :return: Complex spectrum over requested range
        :rtype: numpy.array
        """
        spectrum = misc.FIDToSpec(self.FID)
        first, last = self.ppmlim_to_range(ppmlim, shift=shift)
        return spectrum[first:last]

    def getAxes(self, axis='ppmshift', ppmlim=None):
        """Return x axis over defined limits
        Options: ppmshift, ppm, freq, or time

        :param axis: One of ppmshift, ppm, freq, or time, defaults to 'ppmshift'
        :type axis: str, optional
        :param ppmlim: Chemical shift range over which to retun the axes, defaults to None
            No effect on 'time'
        :type ppmlim: 2-tuple of floats, optional
        :return: Returns the requested axis as numpy array
        :rtype: numpy.array
        """
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

    # Initilisation/setting methods
    def set_acquisition_params(self, centralFrequency, bandwidth):
        """
          Set useful params for fitting

          Parameters
          ----------
          centralFrequency : float  (unit=Hz)
          bandwidth : float (unit=Hz)

        """
        self.centralFrequency = centralFrequency
        self.bandwidth = bandwidth

    def set_nucleus(self, header=None, nucleus=None):
        """Set the nucleus of the FID. Can either pass explicitly, via a header or
        attempt to infer from the central frequency.

        :param header: Data header information, defaults to None
        :type header: dict, optional
        :param nucleus: Nucleus string (e.g. "1H", "31P"), defaults to None
        :type nucleus: str, optional
        """
        if nucleus is not None:
            self._nucleus = nucleus
        elif header is not None and 'ResonantNucleus' in header:
            # Look for nucleus string in header
            self._nucleus = header['ResonantNucleus']
        elif header is not None \
                and 'json' in header \
                    and 'ResonantNucleus' in header['json']:
            # Look for nucleus string in header
            self._nucleus = header['json']['ResonantNucleus']
        else:
            # Else try to infer from central frequency range
            self._nucleus = self.infer_nucleus(self.centralFrequency)

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

    def _calculate_axes(self):
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

    # Other methods
    def parse_metab_groups(self, metab_grp_str):
        """Utility function for generating metabolite groups

        Input (metab_grp_str) may be:
            - A single string : corresponding metab(s) in own group.
                Multiple metabs may be combined into one group with '+'.
            - The strings 'separate_all' or 'combine_all'
            - A list of:
                * integers : output same as input
                * strings : each string is interpreted as metab name and has own group

        :param metab_grp_str:metabolite groups
        :type metab_grp_str: str or list
        :return: metabolite group indices
        :rtype: list
        """
        from fsl_mrs.utils.misc import parse_metab_groups
        return parse_metab_groups(self, metab_grp_str)

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
        if shift:
            return misc.limit_to_range(self.ppmAxisShift, ppmlim)
        else:
            return misc.limit_to_range(self.ppmAxis, ppmlim)

    def processForFitting(self, ppmlim=(.2, 4.2), ind_scaling=None):
        """ Apply rescaling and run the conjugation checks"""
        self.check_FID(ppmlim=ppmlim, repair=True)
        self.check_Basis(ppmlim=ppmlim, repair=True)
        self.rescaleForFitting(ind_scaling=ind_scaling)

    def rescaleForFitting(self, scale=100.0, ind_scaling=[]):
        """Apply rescaling across FID, basis, and H20

        :param scale: Arbitrary scaling target, defaults to 100
        :type scale: float, optional
        :param ind_scaling: List of basis spectra to independently scale, defaults to []
        :type ind_scaling: List of strings, optional
        """

        _, scaling = misc.rescale_FID(self._FID, scale=scale)
        # Set scaling that will be dynamically applied when the FID is fetched.
        self._fid_scaling = scaling

        # Set scaling options that will be dynamically applied when the formatted basis is fetched.
        self._scaling_factor = scale
        self._indept_scale = ind_scaling

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
                self.conj_FID = True
                return 1

        return 0

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
        basis = self.basis
        for b in basis.T:
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
                self.conj_Basis = True
                return 1

        return 0

    # Plotting functions
    def plot(self, ppmlim=(0.2, 4.2)):
        """Plot the spectrum in the mrs object

        :param ppmlim: Plot range, defaults to (0.2, 4.2)
        :type ppmlim: tuple, optional
        """
        from fsl_mrs.utils.plotting import plot_spectrum
        plot_spectrum(self, ppmlim=ppmlim)

    def plot_ref(self, ppmlim=(2.65, 6.65)):
        """Plot the reference spectrum in the mrs object

        :param ppmlim: Plot range, defaults to (2.65, 6.65)
        :type ppmlim: tuple, optional
        """
        from fsl_mrs.utils.plotting import plot_spectrum
        plot_spectrum(self, FID=self.H2O, ppmlim=ppmlim)

    def plot_fid(self, tlim=None):
        """Plot the time-domain data (FID)

        :param tlim: Plot range (in seconds), defaults to None
        :type tlim: tuple, optional
        """
        from fsl_mrs.utils.plotting import plot_fid
        plot_fid(self, tlim)

    def plot_basis(self, add_spec=False, ppmlim=(0.2, 4.2)):
        """Plot the formatted basis in the mrs object. Opptionally add the spectrum.

        :param add_spec: Add spectrum to the plot, defaults to False
        :type add_spec: bool, optional
        :param ppmlim: Plot range, defaults to (0.2, 4.2)
        :type ppmlim: tuple, optional
        """
        from fsl_mrs.utils.plotting import plot_mrs_basis
        plot_mrs_basis(self, plot_spec=add_spec, ppmlim=ppmlim)

    # Unused functions
    # def add_expt_MM(self, lw=5):
    #     """
    #        Add experimental MM basis derived from AA residues

    #     Parameters
    #     ----------

    #     lw : Linewidth of basis spectra
    #     """
    #     from fsl_mrs.mmbasis.mmbasis import getMMBasis
    #     warnings.warn('This is an experimental feature!')

    #     basisFIDs, names = getMMBasis(self, lw=lw, shift=True)
    #     for basis, n in zip(basisFIDs, names):
    #         self.basis = np.append(self.basis, basis[:, np.newaxis], axis=1)
    #         self.names.append('MM_' + n)
    #         self.numBasis += 1
