"""
Core Basis spectra handling class.

Author: Will Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright Will Clarke, University of Oxford, 2021.
"""
import numpy as np

from pathlib import Path

import fsl_mrs.utils.mrs_io as mrs_io
from fsl_mrs.utils.mrs_io import fsl_io
from fsl_mrs.utils import misc


class BasisError(Exception):
    pass


class IncompatibleBasisFormat(BasisError):
    pass


class BasisHasInsufficentCoverage(BasisError):
    pass


class Basis:
    """A Basis object is the FSL-MRS basis spectra handling class.
    """
    def __init__(self, fid_array, names, headers):
        """Generate a Basis object from an array of fids, names and header information.

        :param fid_array: 2D array of basis FIDs (time x metabs)
        :type fid_array: numpy.ndarray
        :param names: List of metabolite names corresponding to second dimension of fid_array
        :type names: List of str
        :param headers: List of basis headers for each column of fid_array
        :type headers: List of dict
        """
        # Check all the basis headers match
        def hdr_match(hdr1, hdr2):
            return np.isclose(hdr1['dwelltime'], hdr2['dwelltime'])\
                and np.isclose(hdr1['bandwidth'], hdr2['bandwidth'])\
                and np.isclose(hdr1['centralFrequency'], hdr2['centralFrequency'])\

        for hdr, name in zip(headers, names):
            if not hdr_match(hdr, headers[0]):
                raise BasisError(f'Basis headers must match, {name} does not match')

        # Check for duplicate names
        for name in names:
            dupes = [idx for idx, n in enumerate(names) if n == name]
            if len(dupes) > 0:
                for idx, ddx in enumerate(dupes[1:]):
                    names[ddx] = names[ddx] + f'_{idx + 1}'
                    print(f'Found duplicate basis name "{name}", renaming to "{names[ddx]}".')

        # Checks on shape of fids
        if fid_array.ndim == 1:
            fid_array = fid_array[:, np.newaxis]
        elif fid_array.ndim > 2:
            raise TypeError('Basis fids must be 2D')

        if fid_array.shape[0] < fid_array.shape[1]:
            fid_array = fid_array.T

        self._raw_fids = fid_array
        self._dt = headers[0]['dwelltime']
        self._cf = misc.checkCFUnits(headers[0]['centralFrequency'], units='MHz')
        self._names = names
        self._widths = [hdr['fwhm'] for hdr in headers]

        # Try to read nucleus from basis file.
        # If not found assume Nucleus is 1H
        # This only has baring on the plotting but causes difficulty in checking basis
        # suitability
        if 'nucleus' in headers[0] and headers[0]['nucleus'] is not None:
            self.nucleus = headers[0]['nucleus']
        else:
            self.nucleus = '1H'

        # Default interpolation is Fourier Transform based.
        self._use_fourier_interp = True

    @classmethod
    def from_file(cls, filepath):
        """Create a Basis object from a path

        :param filepath: Path to basis file or folder
        :type filepath: str or pathlib.Path
        :return: A Basis class object
        :rtype: .Basis
        """
        return mrs_io.read_basis(filepath)

    def __str__(self):
        out = '------- Basis Object ---------\n'
        out += f'     BASIS.NMetabs           = {self.n_metabs}\n'
        out += f'     BASIS.timepoints        = {self.original_points}\n'
        out += f'     BASIS.centralFreq (MHz) = {self.cf:0.3f}\n'
        out += f'     BASIS.bandwidth (Hz)    = {self.original_bw:0.1f}\n'
        out += f'     BASIS.dwelltime (s)     = {self.original_dwell:0.5e}\n'
        out += f'   \nMetabolites: {self.names}'
        return out

    def __repr__(self) -> str:
        return str(self)

    @property
    def cf(self):
        """Get the central frequency in MHz"""
        return self._cf

    @property
    def n_metabs(self):
        """Get the number of metabolites"""
        return self._raw_fids.shape[1]

    @property
    def names(self):
        """Get the names of all metabolites"""
        return self._names

    @property
    def original_points(self):
        """Get the original (input) number of points"""
        return self._raw_fids.shape[0]

    @property
    def original_dwell(self):
        """Get the original (input) dwell time in s"""
        return self._dt

    @property
    def original_bw(self):
        """Get the original (input) bandwidth in Hz"""
        return 1 / self._dt

    @property
    def original_basis_array(self):
        """Get the original input data"""
        return self._raw_fids

    @property
    def basis_fwhm(self):
        """Get the original input data fwhm"""
        return self._widths

    @property
    def original_time_axis(self):
        """Return the time axis of the raw basis set"""
        return misc.calculateAxes(self.original_bw, self.cf * 1E6, self.original_points, 0.0)['time']

    @property
    def original_ppm_axis(self):
        """Return the ppm axis of the raw basis set"""
        return misc.calculateAxes(self.original_bw, self.cf * 1E6, self.original_points, 0.0)['ppm']

    @property
    def original_ppm_shift_axis(self):
        """Return the ppm axis (with offset) of the raw basis set"""
        from fsl_mrs.utils.constants import PPM_SHIFT
        if self.nucleus in PPM_SHIFT:
            shift = PPM_SHIFT[self.nucleus]
            return misc.calculateAxes(self.original_bw, self.cf * 1E6, self.original_points, shift)['ppmshift']
        else:
            return self.original_ppm_axis

    @property
    def nucleus(self):
        """Return nucleus string"""
        return self._nucleus

    @nucleus.setter
    def nucleus(self, nucleus):
        """Set the nucleus string - only affects plotting"""
        if misc.check_nucleus_format(nucleus):
            self._nucleus = nucleus
        else:
            raise ValueError("Nucleus string must be in format of '1H', '31P', '23Na' etc.")

    @property
    def use_fourier_interp(self):
        """Return interpolation state"""
        return self._use_fourier_interp

    @use_fourier_interp.setter
    def use_fourier_interp(self, true_false):
        """Set to true to use FFT based interpolation (default)
        Or set to False to use time domain linear interpolation."""
        self._use_fourier_interp = true_false

    def save(self, out_path, overwrite=False, info_str=''):
        """Saves basis held in memory to a directory in FSL-MRS format.

        :param out_path: Directory to save files to, will be created if neccessary.
        :type out_path: str or pathlib.Path
        :param overwrite: Overwrite existing files, defaults to False.
        :type overwrite: bool, optional
        :param sim_info: Information to write to meta.SimVersion field, defaults to empy string
        :type sim_info: str, optional
        """
        out_path = Path(out_path)

        def out_hdr(width):
            return {'centralFrequency': self.cf * 1E6,
                    'bandwidth': self.original_bw,
                    'dwelltime': self.original_dwell,
                    'fwhm': width,
                    'nucleus': self.nucleus}

        for name, basis, width in zip(self.names, self.original_basis_array.T, self.basis_fwhm):
            hdr = out_hdr(width)
            if not (out_path / (name + '.json')).exists()\
                    or ((out_path / (name + '.json')).exists() and overwrite):
                fsl_io.write_fsl_basis_file(basis, name, hdr, out_path, info=info_str)
            else:
                continue

    def get_formatted_basis(self, bandwidth, points, ignore=[], scale_factor=None, indept_scale=[]):
        """Returns basis formatted to an appropriate number of points and bandwidth.
        Metabolites can be excluded based on the ignore options used.
        The basis spectra will be scaled to have a certain norm (if not None), with indept_scale indicating
        basis to be scaled separately.


        :param bandwidth: Bandwidth of target format
        :type bandwidth: float
        :param points: Number of points in target format
        :type points: int
        :param ignore: Ignores any metabolites in this list, defaults to empty List
        :type ignore: List of string, optional
        :param scale_factor: Norm of basis is scaled to this value, defaults to None
        :type scale_factor: float, optional
        :param indept_scale: [description], defaults to empty List
        :type indept_scale: List of strings, optional
        :return: Formatted basis (points * N metabolites)
        :rtype: numpy.ndarray
        """
        # 1. Resample
        formatted_basis = self._resampled_basis(1 / bandwidth, points)

        # 2. Select the correct basis using the ignore syntax
        ind_out = self._ignore_indicies(ignore)
        formatted_basis = formatted_basis[:, ind_out]

        # 3. Rescale
        if scale_factor:
            formatted_basis = self._rescale_basis(
                formatted_basis,
                self.get_formatted_names(ignore),
                scale_factor,
                indept_scale)[0]

        return formatted_basis

    def get_formatted_names(self, ignore=[]):
        """Return the names of metabolites included with any ignore options.

        :param ignore: Metabolites to ignore, defaults to None
        :type ignore: List of strings
        :return: Retained names
        :rtype: List of strings
        """
        ind_out = self._ignore_indicies(ignore)

        return np.asarray(self.names)[ind_out].tolist()

    def get_rescale_values(self, bandwidth, points, ignore=[], scale_factor=None, indept_scale=[]):
        """Return the rescaling values usingt he same syntax as get_formatted_basis"""
        # 1. Resample
        formatted_basis = self._resampled_basis(1 / bandwidth, points)

        # 2. Select the correct basis using the ignore syntax
        ind_out = self._ignore_indicies(ignore)
        formatted_basis = formatted_basis[:, ind_out]

        # 3. Rescale
        if scale_factor:
            return self._rescale_basis(
                formatted_basis,
                self.get_formatted_names(ignore),
                scale_factor,
                indept_scale)[1]
        else:
            return [1.0, ]

    def _ignore_indicies(self, ignore):
        """Returns indicies of metabolites that should be used given
        the loaded basis set and the ignore options passed.

        :param ignore: [description]
        :type ignore: [type]
        :return: [description]
        :rtype: [type]
        """
        for im in ignore:
            if im not in self.names:
                raise ValueError(f'{im} not in current list of metabolites'
                                 f' ({self.names}).')

        indicies_keep = []
        for idx, metab in enumerate(self.names):
            if metab not in ignore:
                indicies_keep.append(idx)

        return indicies_keep

    def _resampled_basis(self, target_dwell, target_points):
        """
           Usually the basis is simulated using different
           timings and/or number of points.
           This interpolates the basis to match the FID

           This only works if the basis has greater time-domain
           coverage than the FID.
        """
        try:
            if self.use_fourier_interp:
                basis = misc.ts_to_ts_ft(self._raw_fids,
                                         self.original_dwell,
                                         target_dwell,
                                         target_points)
            else:
                basis = misc.ts_to_ts(self._raw_fids,
                                      self.original_dwell,
                                      target_dwell,
                                      target_points)
        except misc.InsufficentTimeCoverageError:
            raise BasisHasInsufficentCoverage('The basis spectra covers too little time. '
                                              'Please reduce the dwelltime, number of points or pad this basis.')

        return basis

    @staticmethod
    def _rescale_basis(basis, names, scale, indept):
        """Calculate the recaled basis also return the scaling factor

        :param basis: Basis to rescale
        :type basis: numpy.ndarray
        :param names: List of metabolite names
        :type names: List of strings
        :param scale: Target scale
        :type scale: float
        :param indept: List of basis to rescale independently
        :type indept: List of strings
        :return: Scaled basis
        :rtype: numpy.ndarray
        :return: List of scaling factors corresponding to main scaling and independent factors
        :rtype: List of floats
        """
        if indept is None:
            indept = []
        index = [names.index(n) for n in indept]

        # First scale all those not selected together.
        mask = np.zeros_like(names, dtype=bool)
        mask[index] = True
        basis[:, ~mask], scaling = misc.rescale_FID(
            basis[:, ~mask],
            scale=scale)
        scaling = [scaling]

        # Then loop over basis spec to independently scale
        for idx in index:
            basis[:, idx], tmp = misc.rescale_FID(
                basis[:, idx],
                scale=scale)
            scaling.append(tmp)
        return basis, scaling

    def add_fid_to_basis(self, new_fid, name, width=None):
        """Adds a new FID to the basis set

        :param new_fid: 1-D FID to add
        :type new_fid: numpy.array
        :param name: Name of new fid, must not match existing value
        :type name: str
        :param width: Width (fwhm) in hz, defaults to None
        :type width: float, optional
        """
        new_fid = new_fid.squeeze()
        if new_fid.ndim > 1:
            raise IncompatibleBasisFormat('New FID must be 1D.')

        if new_fid.size != self.original_points:
            pts = self.original_points
            raise IncompatibleBasisFormat(f'New FID must have {pts} points.')

        if name in self.names:
            raise ValueError(f'New name must be different to existing names {self.names}')

        # Concatenate to end
        self._raw_fids = np.concatenate((self._raw_fids, new_fid[:, np.newaxis]), axis=1)
        self._names.append(name)
        self._widths.append(width)

    def remove_fid_from_basis(self, name):
        """'Permanently' remove a fid from the core basis.
        Typically use the keep/ignore syntax for this purpose.

        :param name: Name of metabolite/fid to remove
        :type name: str
        """
        index = self.names.index(name)
        self._raw_fids = np.delete(self._raw_fids, index, axis=1)
        self._names.pop(index)
        self._widths.pop(index)

    def add_peak(
            self,
            ppm,
            amp,
            name,
            gamma=0.0,
            sigma=0.0,
            phase=0.0,
            conj=False):
        """Add Voigt peak to basis at specified ppm

        :param ppm: The ppm position of the peak
        :type ppm: float | list[float]
        :param amp: Amplitude of the peak
        :type amp: float | list[float]
        :param name: Name of new basis
        :type name: str
        :param gamma: Lorentzian line broadening, defaults to 0
        :type gamma: float, optional
        :param sigma: Guassian line broadening, defaults to 0
        :type sigma: float, optional
        :param phase: Peak phae in radians, defaults to 0
        :type phase: float | list[float], optional
        :param conj: Conjugate fid, defaults to False
        :type conj: Bool, optional
        """
        # Calculate the time axis
        time_axis = self.original_time_axis
        time_axis -= time_axis[0]

        fid = misc.create_peak(time_axis, self.cf, ppm, amp, gamma, sigma, phase)[:, None]
        width = None  # TO DO
        if conj:
            fid = fid.conj()
        self.add_fid_to_basis(fid, name, width=width)

    def add_default_MM_peaks(self, gamma=0, sigma=0, conj=False):
        """Add the default MM peaks to the basis set
        These use the defined shifts and amplitudes
        ppmlist :  [0.9,1.2,1.4,1.7,[2.08,2.25,1.95,3.0]]
        amplist : [3.0,2.0,2.0,2.0,[1.33,0.33,0.33,0.4]]

        :param gamma: Lorentzian broadening, defaults to 0
        :type gamma: int, optional
        :param sigma: Gaussian broadening, defaults to 0
        :type sigma: int, optional
        """
        from fsl_mrs.utils.constants import DEFAULT_MM_AMP, DEFAULT_MM_PPM
        return self.add_MM_peaks(
            DEFAULT_MM_PPM,
            DEFAULT_MM_AMP,
            gamma=gamma,
            sigma=sigma,
            conj=conj)

    def add_default_MEGA_MM_peaks(self, gamma=0, sigma=0, conj=False):
        """Add the default MEGA-PRESS MM peaks to the basis set
        These use the defined shifts and amplitudes
        ppmlist : [[0.94, 3.0]]
        amplist : [[3.0, 2.0]]

        :param gamma: Lorentzian broadening, defaults to 0
        :type gamma: int, optional
        :param sigma: Gaussian broadening, defaults to 0
        :type sigma: int, optional
        """
        from fsl_mrs.utils.constants import DEFAULT_MM_MEGA_AMP, DEFAULT_MM_MEGA_PPM
        return self.add_MM_peaks(
            DEFAULT_MM_MEGA_PPM,
            DEFAULT_MM_MEGA_AMP,
            gamma=gamma,
            sigma=sigma,
            conj=conj)

    def add_MM_peaks(self, ppmlist, amplist, gamma=0, sigma=0, conj=False):
        """Add extra Gaussian peaks (normal MM spectra) to basis set

        :param ppmlist: List of shifts, nested lists group into single basis.
        :type ppmlist: List of floats
        :param amplist: List of amplitudes, nested lists group into single basis.
        :type amplist: List of floats
        :param gamma: Lorentzian broadening, defaults to 0
        :type gamma: int, optional
        :param sigma: Gaussian broadening, defaults to 0
        :type sigma: int, optional
        :return: Number of basis sets added
        :rtype: int
        """
        for idx, _ in enumerate(ppmlist):
            if isinstance(ppmlist[idx], (float, int)):
                ppmlist[idx] = [float(ppmlist[idx]), ]
            if isinstance(amplist[idx], (float, int)):
                amplist[idx] = [float(amplist[idx]), ]

        names = [f'MM{i[0] * 10:02.0f}' for i in ppmlist]

        for name, ppm, amp in zip(names, ppmlist, amplist):
            self.add_peak(ppm, amp, name, gamma, sigma, conj=conj)

        return names

    def add_water_peak(self, gamma=0.0, sigma=0.0, ppm=4.65, amp=1.0, name='H2O', conj=False):
        """Add a peak at 4.65 ppm to capture (residual) water.

        :param gamma: Lorentzian broadening, defaults to 0
        :type gamma: float, optional
        :param sigma: Gaussian broadening, defaults to 0
        :type sigma: float, optional
        :param ppm: Peak position, defaults to 4.65
        :type ppm: float, optional
        :param amp: Peak amplitude, defaults to 1.0
        :type amp: float, optional
        :param name: Basis name, defaults to 'H2O'
        :type name: str, optional
        :return: Number of basis spectra added (1).
        :rtype: int
        """

        self.add_peak(ppm, amp, name, gamma, sigma, conj=conj)
        return [name, ]

    def update_fid(self, new_fid, name):
        """Update a single unformatted basis

        :param new_fid: Updated basis FID
        :type new_fid: numpy.ndarray
        :param name: Name of metabolite to update
        :type name: str
        """
        index = self.names.index(name)
        self._raw_fids[:, index] = new_fid

    def plot(self, ppmlim=None, shift=True, conjugate=False):
        """Plot the basis contained in this Basis object

        :param ppmlim: Chemical shift plotting limits on x axis, defaults to None
        :type ppmlim: tuple, optional
        :param shift: Apply chemical shift referencing shift, defaults to True.
        :type shift: Bool, optional
        :param conjugate: Apply conjugation (flips frequency direction), defaults to False.
        :type conjugate: Bool, optional
        :return: Figure object
        """
        from fsl_mrs.utils.plotting import plot_basis
        return plot_basis(self, ppmlim=ppmlim, shift=shift, conjugate=conjugate)
