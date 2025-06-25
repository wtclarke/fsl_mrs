# misc.py - Various utils
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import os
from contextlib import contextmanager
from typing import Union
import re

import numpy as np
import itertools as it

from fsl.data.image import getExt

from .constants import PPM_SHIFT
H2O_PPM_TO_TMS = PPM_SHIFT['1H']


def ppm2hz(cf, ppm, shift=True, shift_amount=H2O_PPM_TO_TMS):
    """Convert ppm scale to frequency scale with optional shift."""
    if shift:
        return (ppm - shift_amount) * checkCFUnits(cf, units='MHz')
    else:
        return (ppm) * checkCFUnits(cf, units='MHz')


def hz2ppm(cf, hz, shift=True, shift_amount=H2O_PPM_TO_TMS):
    """Convert frequency scale to frequency scale with optional shift."""
    if shift:
        return 1E6 * hz / cf + shift_amount
    else:
        return 1E6 * hz / cf


def FIDToSpec(FID, axis=0):
    """ Convert FID to spectrum

        Performs fft along indicated axis
        Args:
            FID (np.array)      : array of FIDs
            axis (int,optional) : time domain axis

        Returns:
            x (np.array)        : array of spectra
    """
    # By convention the first point of the fid is special cased
    ss = [slice(None) for i in range(FID.ndim)]
    ss[axis] = slice(0, 1)
    ss = tuple(ss)
    FID[ss] *= 0.5
    out = np.fft.fftshift(np.fft.fft(FID,
                                     axis=axis,
                                     norm='ortho'),
                          axes=axis)
    FID[ss] *= 2
    return out


def SpecToFID(spec, axis=0):
    """ Convert spectrum to FID

        Performs fft along indicated axis
        Args:
            spec (np.array)     : array of spectra
            axis (int,optional) : freq domain axis

        Returns:
            x (np.array)        : array of FIDs
    """
    fid = np.fft.ifft(np.fft.ifftshift(spec,
                                       axes=axis),
                      axis=axis, norm='ortho')
    ss = [slice(None) for i in range(fid.ndim)]
    ss[axis] = slice(0, 1)
    ss = tuple(ss)
    fid[ss] *= 2
    return fid


def calculateAxes(bandwidth, centralFrequency, points, shift):
    """Generate time, frequency and ppm axes.
    Input:
        bandwidth: Bandwidth in Hz.
        centralFrequency: Spectroscopy frequency in Hz.
        points: Number of time domain points.
        shift: Shift on ppm axis.
    Returns:
        Dict with 'time', 'freq', 'ppm', 'ppmshift' fields.
    """
    centralFrequency = checkCFUnits(centralFrequency)
    dwellTime = 1 / bandwidth
    timeAxis = np.linspace(dwellTime,
                           dwellTime * points,
                           points)
    frequencyAxis = np.linspace(-bandwidth / 2,
                                bandwidth / 2,
                                points)
    ppmAxis = hz2ppm(centralFrequency,
                     frequencyAxis,
                     shift=False)
    ppmAxisShift = hz2ppm(centralFrequency,
                          frequencyAxis,
                          shift=True,
                          shift_amount=shift)

    return {'time': timeAxis,
            'freq': frequencyAxis,
            'ppm': ppmAxis,
            'ppmshift': ppmAxisShift}


def checkCFUnits(cf, units='Hz'):
    """ Check the units of central frequency and adjust if required."""
    # Assume cf in Hz > 1E5, if it isn't assume that user has passed in MHz
    if cf < 1E5:
        if units.lower() == 'hz':
            cf *= 1E6
        elif units.lower() == 'mhz':
            pass
        else:
            raise ValueError('Only Hz or MHz defined')
    else:
        if units.lower() == 'hz':
            pass
        elif units.lower() == 'mhz':
            cf /= 1E6
        else:
            raise ValueError('Only Hz or MHz defined')
    return cf


def limit_to_range(axis, limit):
    """turns limit (ppm, frequency, or time) into data range

    :param axis: Index to apply limits to
    :type axis: numpy.ndarray
    :param limit: Limits - tuple of (low, high)
    :type limit: tuple of floats
    :return: First and last indicies
    :rtype: tuple of ints
    """
    if limit is not None:
        def ppm2range(x):
            return np.argmin(np.abs(axis - x))

        first = ppm2range(limit[0])
        last = ppm2range(limit[1])
        if first > last:
            first, last = last, first
    else:
        first, last = 0, axis.size

    return int(first), int(last)


def filter(mrs, FID, ppmlim, filter_type='bandpass'):
    """
       Filter in/out frequencies defined in ppm

       Parameters
       ----------
       mrs    : MRS Object
       FID    : array-like
              temporal signal to filter
       ppmlim : float or tuple
       filter_type: {'lowpass','highpass','bandpass', 'bandstop'}
              default type is 'bandstop')

       Outputs
       -------
       numpy array
    """
    from scipy.signal import butter, lfilter

    # Sampling frequency (Hz)
    fs = 1 / mrs.dwellTime
    nyq = 0.5 * fs

    f1 = np.abs(ppm2hz(mrs.centralFrequency, ppmlim[0]) / nyq)
    f2 = np.abs(ppm2hz(mrs.centralFrequency, ppmlim[1]) / nyq)

    if f1 > f2:
        f1, f2 = f2, f1
    wn = [f1, f2]

    order = 6

    b, a = butter(order, wn, btype=filter_type)
    y = lfilter(b, a, FID)
    return y


class InsufficentTimeCoverageError(Exception):
    pass


def ts_to_ts(old_ts, old_dt, new_dt, new_n):
    """Temporal resampling where the new time series has a smaller number of points

    Input:
        old_ts: Input time-domain data
        old_dt: Input dwelltime
        new_dt: Output dwelltime
        new_n: Output number of points
    """
    from scipy.interpolate import interp1d

    old_n = old_ts.shape[0]
    old_t = np.linspace(old_dt, old_dt * old_n, old_n) - old_dt
    new_t = np.linspace(new_dt, new_dt * new_n, new_n) - new_dt
    # Round to nanoseconds
    old_t = np.round(old_t, 9)
    new_t = np.round(new_t, 9)

    if new_t[-1] > old_t[-1]:
        raise InsufficentTimeCoverageError('Input data covers less time than is requested by interpolation.'
                                           ' Change interpolation points or dwell time.')

    f = interp1d(old_t, old_ts, axis=0)
    new_ts = f(new_t)

    return new_ts


def ts_to_ts_ft(old_ts, old_dt, new_dt, new_n):
    """Temporal resampling using Fourier transform based resampling

    Using the method implemented in LCModel:
    1. Data is padded or truncated in the spectral domain to match the bandwidth of the target.
    The ifft then returns the time domain data with the right overall length.
    2. The data is then padded or truncated in the time domain to the length of the target.
    If the data is then FFT it return the interpolated data.

    :param old_ts: Input time-domain data
    :type old_ts: numpy.ndarray
    :param old_dt: Input dwelltime
    :type old_dt: float
    :param new_dt: Target dwell time
    :type new_dt: float
    :param new_n: Target number of points
    :type new_n: int
    :rtype: numpy.ndarray
    """

    old_n = old_ts.shape[0]
    old_t = np.linspace(old_dt, old_dt * old_n, old_n) - old_dt
    new_t = np.linspace(new_dt, new_dt * new_n, new_n) - new_dt
    # Round to nanoseconds
    old_t = np.round(old_t, 9)
    new_t = np.round(new_t, 9)

    if new_t[-1] > old_t[-1]:
        raise InsufficentTimeCoverageError('Input data covers less time than is requested by interpolation.'
                                           ' Change interpolation points or dwell time.')

    def f2s(x):
        return np.fft.fftshift(np.fft.fft(x, axis=0), axes=0)

    def s2f(x):
        return np.fft.ifft(np.fft.ifftshift(x, axes=0), axis=0)

    # Input data to frequency domain
    old_fs = f2s(old_ts)

    # Step 1: pad or truncate in the frequency domain
    new_bw = 1 / new_dt
    old_bw = 1 / old_dt
    npoints_f = (new_bw - old_bw) / (old_bw / old_ts.shape[0])
    npoints_f_half = int(np.round(npoints_f / 2))

    # scale_factor = np.abs(float(npoints_f_half) * 2.0) / new_n
    if npoints_f_half < 0:
        # New bandwidth is smaller than old. Truncate
        npoints_f_half *= -1
        step1 = s2f(old_fs[npoints_f_half:-npoints_f_half])
    elif npoints_f_half > 0:
        # New bandwidth is larger than old. Pad
        step1 = s2f(np.pad(old_fs, ((npoints_f_half, npoints_f_half), (0, 0)), 'constant', constant_values=(0j, 0j)))
    else:
        step1 = s2f(old_fs)

    # Scaling for different length fft/ifft
    step1 = step1 * step1.shape[0] / old_fs.shape[0]

    # Step 2: pad or truncate in the temporal domain
    if step1.shape[0] < new_n:
        step2 = np.pad(step1, ((0, new_n - step1.shape[0]), (0, 0)), 'constant', constant_values=(0j, 0j))
    else:
        step2 = step1[:new_n]

    return step2


# Numerical differentiation (light)
# Gradient Function
def gradient(x, f):
    """
      Calculate f'(x): the numerical gradient of a function

      Parameters:
      -----------
      x : array-like
      f : scalar function

      Returns:
      --------
      array-like
    """
    N = len(x)
    gradient = []
    for i in range(N):
        eps = abs(x[i]) * np.finfo(np.float32).eps
        if eps <= np.finfo(np.float32).eps:
            eps = np.finfo(np.float32).eps
        xl = np.array(x)
        xu = np.array(x)
        xl[i] -= eps
        xu[i] += eps
        fl = f(xl)
        fu = f(xu)
        gradient.append((fu - fl) / (2 * eps))
    return np.array(gradient)


# Hessian Matrix
def hessian(x, f):
    """
       Calculate numerical Hessian of f at x

       Parameters:
       -----------
       x : array-like
       f : function

       Returns:
       --------
       matrix
    """
    N = len(x)
    hessian = []
    gd_0 = gradient(x, f)
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps
    if eps == 0:
        eps = 1e-5
    for i in range(N):
        hessian.append([])
        xx0 = 1. * x[i]
        x[i] = xx0 + eps
        gd_1 = gradient(x, f)
        for j in range(N):
            hessian[i].append((gd_1[j, :] - gd_0[j, :]) / eps)
        x[i] = xx0
    return np.asarray(hessian)


def hessian_diag(x, f):
    """
       Calculate numerical second order derivative of f at x
       (the diagonal of the Hessian)

       Parameters:
       -----------
       x : array-like
       f : function

       Returns:
       --------
       array-like
    """
    N = x.size
    hess = np.zeros((N, 1))
    gd_0 = gradient(x, f)
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps

    if eps == 0:
        eps = 1e-5
    for i in range(N):
        xx0 = 1. * x[i]
        x[i] = xx0 + eps
        gd_1 = gradient(x, f)
        hess[i] = ((gd_1[i] - gd_0[i]) / eps)
        x[i] = xx0

    return hess


def check_gradients():
    """Little bit of code for checking the gradients."""
    m = np.linspace(0, 10, 100)

    def cf(p):
        return np.sum(p[0] * np.exp(-p[1] * m))
    x0 = np.random.randn(2) * .1
    grad_num = gradient(x0, cf)

    def E(x):
        return np.sum(np.exp(-x[1] * m))

    grad_anal = np.array([E(x0), -x0[0] * np.sum(m * np.exp(-x0[1] * m))])
    hess_anal = np.zeros((2, 2))
    hess_anal[0, 1] = -np.sum(m * np.exp(-x0[1] * m))
    hess_anal[1, 0] = -np.sum(m * np.exp(-x0[1] * m))
    hess_anal[1, 1] = x0[0] * np.sum(m**2 * np.exp(-x0[1] * m))
    hess_num = hessian(x0, cf)
    hess_diag = hessian_diag(x0, cf)
    print('x0 = {}, f(x0)  = {}'.format(x0, cf(x0)))
    print('Grad Analytic   : {}'.format(grad_anal))
    print('Grad Numerical  : {}'.format(grad_num))
    print('Hess Analytic   : {}'.format(hess_anal))
    print('Hess Numreical  : {}'.format(hess_num))
    print('Hess Diag       : {}'.format(hess_diag))


def calculate_crlb(x, f, data):
    """
       Calculate Cramer-Rao Lower Bound
       This assumes a model of the form data = f(x) + noise
       where noise ~ N(0,sig^2)
       In which case the CRLB is sum( |f'(x)|^2 )/sig^2
       It uses numerical differentiation to get f'(x)

      Parameters:
       x : array-like
       f : function
       data : array-like

      Returns:
        array-like
    """
    # estimate noise variance empirically
    sig2 = np.var(data - f(x))
    grad = gradient(x, f)
    crlb = 1 / (np.sum(np.abs(grad)**2, axis=1) / sig2)

    return crlb


def calculate_lap_cov(x, f, data, grad=None, sig2=None, additional_term=None):
    """
      Calculate approximate covariance using
      Fisher information matrix

      Assumes forward model is data=f(x)+N(0,sig^2)

      Parameters:
       x : array-like
       f : function
       data : array-like
       grad : optional jacobian matrix
       sig2 : optional noise variance
       additional_term: optional additional (penalty) term

      Returns:
        2D array
    """
    x = np.asarray(x)
    N = x.size
    if sig2 is None:
        sig2 = np.var(data - f(x))
    if grad is None:
        grad = gradient(x, f)

    J = np.concatenate((np.real(grad), np.imag(grad)), axis=1)
    P0 = np.diag(np.ones(N) * 1E-5)
    P = np.dot(J, J.transpose()) / sig2
    if additional_term is not None:
        if additional_term.shape != P.shape:
            raise ValueError(f'additional_term must be the same size as P {P.shape}.')
        C = np.linalg.inv(P + P0 + additional_term)
    else:
        C = np.linalg.inv(P + P0)

    return C


# Various utilities
def multiply(x, y):
    """
     Elementwise multiply numpy arrays x and y

     Returns same shape as x
    """
    shape = x.shape
    r = x.flatten() * y.flatten()
    return np.reshape(r, shape)


def shift_FID(mrs, FID, eps):
    """
       Shift FID in spectral domain

    Parameters:
       mrs : MRS object
       FID : array-like
       eps : shift factor (Hz)

    Returns:
       array-like
    """
    t = mrs.timeAxis
    FID_shifted = multiply(FID, np.exp(-1j * 2 * np.pi * t * eps))

    return FID_shifted


def blur_FID(mrs, FID, gamma):
    """
       Blur FID in spectral domain

    Parameters:
       mrs   : MRS object
       FID   : array-like
       gamma : blur factor in Hz

    Returns:
       array-like
    """
    t = mrs.timeAxis
    FID_blurred = multiply(FID, np.exp(-t * gamma))
    return FID_blurred


def blur_FID_Voigt(time_axis, FID, gamma, sigma):
    """
       Blur FID in spectral domain

    Parameters:
       time_axis : time_axis
       FID       : array-like
       gamma     : Lorentzian line broadening
       sigma     : Gaussian line broadening

    Returns:
       array-like
    """
    FID_blurred = multiply(FID, np.exp(-time_axis * (gamma + time_axis * sigma**2 / 2)))
    return FID_blurred


def rescale_FID(x, scale=100):
    """
    Useful for ensuring values are within nice range

    Forces norm of 1D arrays to be = scale
    Forces norm of column-mean of 2D arrays to be = scale (i.e. preserves relative norms of the columns)

    Parameters
    ----------
    x : 1D or 2D array
    scale : float
    """

    y = x.copy()

    if isinstance(y, list):
        factor = np.linalg.norm(sum(y) / len(y))
        return [yy / factor * scale for yy in y], 1 / factor * scale

    if y.ndim == 1:
        factor = np.linalg.norm(y)
    else:
        factor = np.linalg.norm(np.mean(y, axis=1), axis=0)
    y = y / factor * scale
    return y, 1 / factor * scale


def create_peak(
        time_axis: np.ndarray,
        cf: float,
        ppm: Union[float, list[float]],
        amp: Union[float, list[float]],
        gamma: float = 0,
        sigma: float = 0,
        phase: Union[float, list[float]] = 0) -> np.ndarray:
    """creates FID for peak(s) at specific ppm(s)

    Returns a single fid with one or more specified peaks

    :param time_axis: time axis (in seconds)
    :type time_axis: np.ndarray
    :param cf: Spectrometer (central) frequency in MHz
    :type cf: float
    :param ppm: Position of peak(s) in ppm (relative to cf)
    :type ppm: Union[float, list[float]]
    :param amp: Peak amplitude(s)
    :type amp: Union[float, list[float]]
    :param gamma: Lorentzian peak broadening, defaults to 0
    :type gamma: float, optional
    :param sigma: Gaussian peak broadening, defaults to 0
    :type sigma: float, optional
    :param phase: Phase of peak(s) in radians, defaults to 0
    :type phase: float, optional
    :return: FID
    :rtype: np.ndarray
    """

    if isinstance(ppm, (float, int)):
        ppm = [float(ppm), ]
    if isinstance(amp, (float, int)):
        amp = [float(amp), ]

    if len(ppm) != len(amp):
        raise ValueError(f'ppm and amp should have the same length, currently {len(ppm)} and {len(amp)}')

    if isinstance(phase, (float, int)):
        phase = phase * np.ones_like(ppm)
    elif len(ppm) != len(phase):
        raise ValueError(f'ppm and phase should have the same length, currently {len(ppm)} and {len(phase)}')

    out = np.zeros(time_axis.shape[0], dtype=np.complex128)

    for p, a, phs in zip(ppm, amp, phase):
        freq = ppm2hz(cf, p)

        x = a * np.exp(1j * 2 * np.pi * freq * time_axis).flatten()

        if gamma > 0 or sigma > 0:
            x = blur_FID_Voigt(time_axis, x, gamma, sigma)

        # phase
        x = x * np.exp(1j * phs)
        out += x

    return out


def extract_spectrum(mrs, FID, ppmlim=None, shift=True):
    """
       Extracts spectral interval

    Parameters:
       mrs : MRS object
       FID : array-like
       ppmlim : tuple

    Returns:
       array-like
    """
    if ppmlim is None:
        ppmlim = mrs.default_ppm_range
    spec = FIDToSpec(FID)
    first, last = mrs.ppmlim_to_range(ppmlim=ppmlim, shift=shift)
    spec = spec[first:last]

    return spec


def normalise(x, axis=0):
    """
       Devides x by norm of x
    """
    return x / np.linalg.norm(x, axis=axis)


def ztransform(x, axis=0):
    """
       Demeans x and make norm(x)=1
    """
    return (x - np.mean(x, axis=axis)) / np.std(x, axis) / np.sqrt(x.size)


def correlate(x, y):
    """
       Computes correlation between complex signals x and y
       Uses formula : sum( conj(z(x))*z(y)) where z() is the ztransform
    """
    return np.real(np.sum(np.conjugate(ztransform(x)) * ztransform(y)))


def phase_correct(mrs, FID, ppmlim=(1, 3)):
    """
       Apply phase correction to FID
    """
    first, last = mrs.ppmlim_to_range(ppmlim)
    phases = np.linspace(0, 2 * np.pi, 1000)
    x = []
    for phase in phases:
        f = np.real(np.fft.fft(FID * np.exp(1j * phase), axis=0))
        x.append(np.sum(f[first:last] < 0))
    phase = phases[np.argmin(x)]
    return FID * np.exp(1j * phase)


def detrend(data, deg=1, keep_mean=True):
    """
    remove polynomial trend from data
    works along first dimension
    """
    n = data.shape[0]
    x = np.arange(n)
    M = np.zeros((n, deg + 1))
    for i in range(deg + 1):
        M[:, i] = x**i

    beta = np.linalg.pinv(M) @ data

    pred = M @ beta
    m = 0
    if keep_mean:
        m = np.mean(data, axis=0)
    return data - pred + m


def regress_out(x, conf, keep_mean=True):
    """
    Linear deconfounding
    """
    if isinstance(conf, list):
        confa = np.squeeze(np.asarray(conf)).T
    else:
        confa = conf
    if keep_mean:
        m = np.mean(x, axis=0)
    else:
        m = 0
    return x - confa @ (np.linalg.pinv(confa) @ x) + m


def parse_metab_groups(mrs, metab_groups):
    """
    Creates list of indices per metabolite group

    Parameters:
    -----------
    metab_groups :
       - A single index    : output is a list of 0's
       - A single string   : corresponding metab in own group
       - The strings 'separate_all' or 'combine_all'
       - A list:
        - list of integers : output same as input
        - list of strings  : each string is interpreted as metab name and has own group
       Entries in the lists above can also be lists, in which case the corresponding metabs are grouped

    mrs : MRS or MRSI Object

    Returns
    -------
    list of integers
    """
    if isinstance(metab_groups, list) and len(metab_groups) == 1:
        metab_groups = metab_groups[0]

    out = [0] * mrs.numBasis

    if isinstance(metab_groups, int):
        return out

    if isinstance(metab_groups, str):
        if metab_groups.lower() == 'separate_all':
            return list(range(mrs.numBasis))

        if metab_groups.lower() == 'combine_all':
            return [0] * mrs.numBasis

        entry = metab_groups.split('+')

        if isinstance(entry, str):
            out[mrs.names.index(entry)] = 1
        elif isinstance(entry, list):
            for n in entry:
                assert isinstance(n, str)
                out[mrs.names.index(n)] = 1

        return out

    if isinstance(metab_groups, list):
        if isinstance(metab_groups[0], int):
            assert len(metab_groups) == mrs.numBasis
            return metab_groups

        grpcounter = 0
        for entry in metab_groups:
            if isinstance(entry, str):
                entry = entry.split('+')
            grpcounter += 1
            if isinstance(entry, str):
                out[mrs.names.index(entry)] = grpcounter
            elif isinstance(entry, list):
                for n in entry:
                    assert isinstance(n, str)
                    out[mrs.names.index(n)] = grpcounter
            else:
                raise Exception('entry must be string or list of strings')

    m = min(out)
    if m > 0:
        out = [x - m for x in out]

    return out


def detect_conjugation(
        data: np.ndarray,
        ppmaxis: np.ndarray,
        ppmlim: tuple) -> bool:
    """Detect whether data should be conjugated based on
    the amount of information content in ppm range.

    :param data: FID or stack of FIDS (last dimension is time).
    :type data: np.ndarray
    :param ppmaxis: ppmaxis to match FFT of FID
    :type ppmaxis: np.ndarray
    :param ppmlim: Limits that define region of expected signal
    :type ppmlim: tuple
    :return: True if FIDs should be conjugated to maximise signal in limits.
    :rtype: bool
    """
    if data.shape[-1] != ppmaxis.size:
        raise ValueError("data's last dimension must matcht he size of ppmaxis.")

    first, last = limit_to_range(ppmaxis, ppmlim)

    def conj_or_not(x):
        Spec1 = np.real(FIDToSpec(x))[first:last]
        Spec2 = np.real(FIDToSpec(np.conj(x)))[first:last]
        if np.linalg.norm(detrend(Spec1, deg=4)) < \
                np.linalg.norm(detrend(Spec2, deg=4)):
            return 1
        else:
            return 0

    if data.ndim == 2:
        return sum([conj_or_not(fid) for fid in data]) >= 0.5
    elif data.ndim == 1:
        return bool(conj_or_not(data))
    else:
        raise ValueError(f'data must have one or two dimensions, not {data.ndim}.')


# ----- MRSI stuff ---- #
def volume_to_list(data, mask):
    """
       Turn voxels within mask into list of data

    Parameters
    ----------

    data : 4D array

    mask : 3D array

    Returns
    -------

    list

    """
    nx, ny, nz = data.shape[:3]
    voxels = []
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x, y, z]:
            voxels.append((x, y, z))
    voxdata = [data[x, y, z, :] for (x, y, z) in voxels]
    return voxdata


def list_to_volume(data_list, mask, dtype=float):
    """
       Turn list of voxelwise data into 4D volume

    Parameters
    ----------
    voxdata : list
    mask    : 3D volume
    dtype   : force output data type

    Returns
    -------
    4D or 3D volume
    """

    nx, ny, nz = mask.shape
    nt = data_list[0].size
    if nt > 1:
        data = np.zeros((nx, ny, nz, nt), dtype=dtype)
    else:
        data = np.zeros((nx, ny, nz,), dtype=dtype)
    i = 0
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x, y, z]:
            if nt > 1:
                data[x, y, z, :] = data_list[i]
            else:
                data[x, y, z] = data_list[i]
            i += 1

    return data


def unravel(idx, mask):
    nx, ny, nz = mask.shape
    counter = 0
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x, y, z]:
            if counter == idx:
                return np.array([x, y, z])
            counter += 1


def ravel(arr, mask):
    nx, ny, nz = mask.shape
    counter = 0
    for x, y, z in it.product(range(nx), range(ny), range(nz)):
        if mask[x, y, z]:
            if arr == [x, y, z]:
                return counter
            counter += 1


def check_nucleus_format(nuc_str: str) -> bool:
    """Checks that the nucleus string matches the expected format

    1H, 31P, 13C etc

    :param nuc_str: Nucleus string
    :type nuc_str: str
    :return: True = correct format, false otherwise
    :rtype: bool
    """
    rexp = re.compile(r'^\d{1,3}[A-Z][a-z]?$')
    return rexp.match(nuc_str) is not None


# FMRS Stuff

def smooth_FIDs(FIDlist, window):
    """
    Smooth a list of FIDs
    (makes sense if acquired one after the other as the smoothing is done along the "time" dimension)

    Note: at the edge of the list of FIDs the smoothing wraps around the
    list so make sure that the beginning and the end are 'compatible'.

    Parameters:
    -----------
    FIDlist : list of FIDs
    window  : int (preferably odd number)

    Returns:
    --------
    list of FIDs
    """
    sFIDlist = []
    for idx, FID in enumerate(FIDlist):
        fid = 0
        n = 0
        for i in range(-int(window / 2), int(window / 2) + 1, 1):
            fid = fid + FIDlist[(idx + i) % len(FIDlist)]
            n = n + 1
        fid = fid / n
        sFIDlist.append(fid)
    return sFIDlist


# Path calculations / manipulations

@contextmanager
def _cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def create_rel_symlink(dst, src, name, match_ext=True):
    """Create a symlink at src to dst.

    Might return OSError on Windows depending on version and 'developer mode'

    :param dst: File to link to
    :type dst: pathlib.Path
    :param src: Location of symlink creation
    :type src: pathlib.Path
    :param name: symlink name
    :type name: str
    :param match_ext: Ensure the symlink generated has the same extension as the dst file, defaults to True
    :type match_ext: bool, optional
    """
    if match_ext:
        ext = getExt(dst)
        name += ext

    relpath = os.path.relpath(dst, src)

    if os.supports_dir_fd:
        fd = os.open(src, os.O_RDONLY)
        try:
            os.symlink(relpath, name, dir_fd=fd)
        finally:
            os.close(fd)
    else:
        with _cd(src):
            os.symlink(relpath, name)
