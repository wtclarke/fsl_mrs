'''nifti_mrs_proc.py - translation layer for using NIFTI_MRS with the operations in
fsl_mrs.utils.preproc.

Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
        Will Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 2021 University of Oxford
SHBASECOPYRIGHT'''
from fsl_mrs.utils import preproc
import numpy as np
from fsl_mrs.core import NIFTI_MRS


class DimensionsDoNotMatch(Exception):
    pass


class OnlySVS(Exception):
    pass


def first_index(idx):
    return all([ii == slice(None, None, None) or ii == 0 for ii in idx])


def coilcombine(data, reference=None, no_prewhiten=False, report=None, report_all=False):
    '''Coil combine data optionally using reference data.
    :param NIFTI_MRS data: Data to coil combine
    :param NIFTI_MRS reference: reference dataset to calculate weights
    :param no_prewhiten: True to disable prewhitening
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Combined data in NIFTI_MRS format.
    '''
    if reference is not None:
        if data.shape[data.dim_position('DIM_COIL')] != reference.shape[data.dim_position('DIM_COIL')]:
            raise DimensionsDoNotMatch('Reference and data coil dimension does not match.')

    combinedc_obj = data.copy(remove_dim='DIM_COIL')
    for main, idx in data.iterate_over_dims(dim='DIM_COIL',
                                            iterate_over_space=True,
                                            reduce_dim_index=True):
        if reference is None:
            combinedc_obj[idx] = preproc.combine_FIDs(
                list(main.T),
                'svd',
                do_prewhiten=~no_prewhiten)
        else:
            _, refWeights = preproc.combine_FIDs(
                list(reference[idx[:4]].T),
                'svd_weights',
                do_prewhiten=~no_prewhiten)
            combinedc_obj[idx] = preproc.combine_FIDs(
                list(main.T),
                'weighted',
                weights=refWeights)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            combine_FIDs_report(main,
                                combinedc_obj[idx],
                                data.bandwidth,
                                data.spectrometer_frequency[0],
                                data.nucleus[0],
                                ncha=data.shape[data.dim_position('DIM_COIL')],
                                ppmlim=(0.0, 6.0),
                                method='svd',
                                dim='DIM_COIL',
                                html=report)
    return combinedc_obj


def average(data, dim, report=None, report_all=False):
    '''Average (take the mean) of FIDs across a dimension
    specified by a tag.

    :param NIFTI_MRS data: Data to average
    :param str dim: NIFTI-MRS dimension tag
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Combined data in NIFTI_MRS format.
    '''

    combined_obj = data.copy(remove_dim=dim)
    for dd, idx in data.iterate_over_dims(dim=dim,
                                          iterate_over_space=True,
                                          reduce_dim_index=True):
        combined_obj[idx] = preproc.combine_FIDs(dd, 'mean')

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            combine_FIDs_report(dd,
                                combined_obj[idx],
                                data.bandwidth,
                                data.spectrometer_frequency[0],
                                data.nucleus[0],
                                ncha=data.shape[data.dim_position(dim)],
                                ppmlim=(0.0, 6.0),
                                method=f'Mean along dim = {dim}',
                                html=report)
    return combined_obj


def align(data, dim, target=None, ppmlim=None, niter=2, apodize=10, report=None, report_all=False):
    '''Align frequencies of spectra across a dimension
    specified by a tag.

    :param NIFTI_MRS data: Data to align
    :param str dim: NIFTI-MRS dimension tag
    :param target: Optional target FID
    :param ppmlim: ppm search limits.
    :param int niter: Number of total iterations
    :param float apodize: Apply apodisation in Hz.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Combined data in NIFTI_MRS format.
    '''

    aligned_obj = data.copy()
    for dd, idx in data.iterate_over_dims(dim=dim,
                                          iterate_over_space=True,
                                          reduce_dim_index=True):

        out = preproc.phase_freq_align(
            dd.T,
            data.bandwidth,
            data.spectrometer_frequency[0],
            nucleus=data.nucleus[0],
            ppmlim=ppmlim,
            niter=niter,
            apodize=apodize,
            verbose=False,
            target=target)

        aligned_obj[idx], phi, eps = out[0].T, out[1], out[2]

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.align import phase_freq_align_report
            phase_freq_align_report(dd.T,
                                    aligned_obj[idx].T,
                                    phi,
                                    eps,
                                    data.bandwidth,
                                    data.spectrometer_frequency[0],
                                    nucleus=data.nucleus[0],
                                    ppmlim=ppmlim,
                                    html=report)
    return aligned_obj


def aligndiff(data,
              dim_align,
              dim_diff,
              diff_type,
              target=None,
              ppmlim=None,
              report=None,
              report_all=False):
    '''Align frequencies of difference spectra across a dimension
    specified by a tag.

    :param NIFTI_MRS data: Data to align
    :param str dim_align: NIFTI-MRS dimension tag to align along
    :param str dim_diff: NIFTI-MRS dimension across which diffrence is taken.
    :param str diff_type: Either 'add' or 'sub'
    :param target: Optional target FID
    :param ppmlim: ppm search limits.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Combined data in NIFTI_MRS format.
    '''
    if data.shape[data.dim_position(dim_diff)] != 2:
        raise DimensionsDoNotMatch('Diff dimension must be of length 2')

    aligned_obj = data.copy()
    diff_index = data.dim_position(dim_diff)
    data_0 = []
    data_1 = []
    index_0 = []
    for dd, idx in data.iterate_over_dims(dim=dim_align,
                                          iterate_over_space=True,
                                          reduce_dim_index=False):
        if idx[diff_index] == 0:
            data_0.append(dd)
            index_0.append(idx)
        else:
            data_1.append(dd)

    for d0, d1, idx in zip(data_0, data_1, index_0):
        out = preproc.phase_freq_align_diff(
            d0.T,
            d1.T,
            data.bandwidth,
            data.spectrometer_frequency[0],
            nucleus=data.nucleus[0],
            diffType=diff_type,
            ppmlim=ppmlim,
            target=target)

        aligned_obj[idx], _, phi, eps = np.asarray(out[0]).T, out[1], out[2], out[3]

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.align import phase_freq_align_diff_report
            phase_freq_align_diff_report(d0.T,
                                         d1.T,
                                         aligned_obj[idx].T,
                                         d1.T,
                                         phi,
                                         eps,
                                         data.bandwidth,
                                         data.spectrometer_frequency[0],
                                         nucleus=data.nucleus[0],
                                         diffType=diff_type,
                                         ppmlim=ppmlim,
                                         html=report)
    return aligned_obj


def ecc(data, reference, report=None, report_all=False):
    '''Apply eddy current correction using a reference dataset
    :param NIFTI_MRS data: Data to eddy current correct
    :param NIFTI_MRS reference: reference dataset to calculate phase
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Corrected data in NIFTI_MRS format.
    '''
    if data.shape != reference.shape\
            and reference.ndim > 4:
        raise DimensionsDoNotMatch('Reference and data shape must match'
                                   ' or reference must be single FID.')

    corrected_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):

        if data.shape == reference.shape:
            ref = reference[idx]
        else:
            ref = reference[idx[0], idx[1], idx[2], :]

        corrected_obj[idx] = preproc.eddy_correct(dd, ref)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.eddycorrect import eddy_correct_report
            eddy_correct_report(dd,
                                corrected_obj[idx],
                                ref,
                                data.bandwidth,
                                data.spectrometer_frequency[0],
                                nucleus=data.nucleus[0],
                                html=report)

    return corrected_obj


def remove_peaks(data, limits, limit_units='ppm+shift', report=None, report_all=False):
    '''Apply HLSVD to remove peaks from specta
    :param NIFTI_MRS data: Data to remove peaks from
    :param limits: ppm limits between which peaks will be removed
    :param str limit_units: Can be 'Hz', 'ppm' or 'ppm+shift'.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Corrected data in NIFTI_MRS format.
    '''
    corrected_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):

        corrected_obj[idx] = preproc.hlsvd(dd,
                                           data.dwelltime,
                                           data.spectrometer_frequency[0],
                                           limits,
                                           limitUnits=limit_units)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.remove import hlsvd_report
            hlsvd_report(dd,
                         corrected_obj[idx],
                         limits,
                         data.bandwidth,
                         data.spectrometer_frequency[0],
                         nucleus=data.nucleus[0],
                         limitUnits=limit_units,
                         html=report)

    return corrected_obj


def tshift(data, tshiftStart=0.0, tshiftEnd=0.0, samples=None, report=None, report_all=False):
    '''Apply time shift or resampling to each FID
    :param NIFTI_MRS data: Data to shift
    :param float tshiftStart: Shift start time (s), negative padds with zeros
    :param float tshiftEnd: Shift end time (s), negative truncates
    :param float samples: Resample to this many points
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Shifted data in NIFTI_MRS format.
    '''
    if samples is None:
        samples = data.shape[3]
        shifted_obj = data.copy()
    else:
        new_shape = list(data.shape)
        new_shape[3] = samples
        shifted_obj = NIFTI_MRS(
            np.zeros(new_shape, dtype=data.dtype),
            header=data.header)

    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        shifted_obj[idx], newDT = preproc.timeshift(dd,
                                                    data.dwelltime,
                                                    tshiftStart,
                                                    tshiftEnd,
                                                    samples)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report

            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            new_hdr = {'bandwidth': 1 / newDT,
                       'centralFrequency': data.spectrometer_frequency[0],
                       'ResonantNucleus': data.nucleus[0]}
            shift_report(dd,
                         shifted_obj[idx],
                         original_hdr,
                         new_hdr,
                         html=report,
                         function='timeshift')

    shifted_obj.dwelltime = newDT

    return shifted_obj


def truncate_or_pad(data, npoints, position, report=None, report_all=False):
    '''Truncate or pad by integer number of points
    :param NIFTI_MRS data: Data to truncate or pad
    :param int npoints: Pad (positive) or truncate (negative) by npoints
    :param str position: 'first' or 'last', add or remove points at the
    start or end of the FID
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Padded or truncated data in NIFTI_MRS format.
    '''

    new_shape = list(data.shape)
    new_shape[3] += npoints
    trunc_obj = NIFTI_MRS(
        np.zeros(new_shape, dtype=data.dtype),
        header=data.header)

    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        if npoints > 0:
            trunc_obj[idx] = preproc.pad(dd,
                                         np.abs(npoints),
                                         position)
            rep_func = 'pad'
        elif npoints < 0:
            trunc_obj[idx] = preproc.truncate(dd,
                                              np.abs(npoints),
                                              position)
            rep_func = 'truncate'
        else:
            rep_func = 'none'

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}

            shift_report(dd,
                         trunc_obj[idx],
                         original_hdr,
                         original_hdr,
                         html=report,
                         function=rep_func)
    return trunc_obj


def apodize(data, amount, filter='exp', report=None, report_all=False):
    '''Apodize FIDs using a exponential or Lorentzian to Gaussian filter.
    Lorentzian to Gaussian filter takes requires two window parameters (t_L and t_G)

    :param NIFTI_MRS data: Data to truncate or pad
    :param tuple amount: If filter='exp' single valued. If filter='l2g' then two valued.
    :param str filter: 'exp' or 'l2g'. Choose exponential or Lorentzian to Gaussian filter
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Filtered data in NIFTI_MRS format.
    '''
    apod_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        apod_obj[idx] = preproc.apodize(dd,
                                        data.dwelltime,
                                        amount,
                                        filter=filter)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.filtering import apodize_report
            apodize_report(dd,
                           apod_obj[idx],
                           data.bandwidth,
                           data.spectrometer_frequency[0],
                           nucleus=data.nucleus[0],
                           html=report)

    return apod_obj


def fshift(data, amount, report=None, report_all=False):
    '''Apply frequency shift

    :param NIFTI_MRS data: Data to truncate or pad
    :param float amount: Shift amount in Hz
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Shifted data in NIFTI_MRS format.
    '''

    shift_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        shift_obj[idx] = preproc.freqshift(dd,
                                           data.dwelltime,
                                           amount)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            shift_report(dd,
                         shift_obj[idx],
                         original_hdr,
                         original_hdr,
                         html=report,
                         function='freqshift')

    return shift_obj


def shift_to_reference(data, ppm_ref, peak_search, report=None, report_all=False):
    '''Shift peak to known reference

    :param NIFTI_MRS data: Data to truncate or pad
    :param float ppm_ref: Reference shift that peak will be moved to
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Shifted data in NIFTI_MRS format.
    '''

    shift_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        shift_obj[idx], _ = preproc.shiftToRef(dd,
                                               ppm_ref,
                                               data.bandwidth,
                                               data.spectrometer_frequency[0],
                                               nucleus=data.nucleus[0],
                                               ppmlim=peak_search)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            shift_report(dd,
                         shift_obj[idx],
                         original_hdr,
                         original_hdr,
                         html=report,
                         function='shiftToRef')

    return shift_obj


def remove_unlike(data, ppmlim=None, sdlimit=1.96, niter=2, report=None):
    '''Remove unlike dynamics

    :param NIFTI_MRS data: Data to truncate or pad
    :param report: Provide output location as path to generate report

    :return: Shifted data in NIFTI_MRS format.
    '''
    if data.shape[:3] != (1, 1, 1):
        raise OnlySVS("remove_unlike only specified for SVS data")

    if data.ndim > 5:
        raise ValueError('remove_unlike only makes sense for a single dynamic dimension. Combined coils etc. first')
    elif data.ndim < 5:
        raise ValueError('remove_unlike only makes sense for data with a dynamic dimension')

    goodFIDs, badFIDs, gIndicies, bIndicies, metric = \
        preproc.identifyUnlikeFIDs(data[0, 0, 0, :, :].T,
                                   data.bandwidth,
                                   data.spectrometer_frequency[0],
                                   nucleus=data.nucleus[0],
                                   ppmlim=ppmlim,
                                   sdlimit=sdlimit,
                                   iterations=niter,
                                   shift=True)

    if report:
        from fsl_mrs.utils.preproc.unlike import identifyUnlikeFIDs_report
        identifyUnlikeFIDs_report(goodFIDs,
                                  badFIDs,
                                  gIndicies,
                                  bIndicies,
                                  metric,
                                  data.bandwidth,
                                  data.spectrometer_frequency[0],
                                  nucleus=data.nucleus[0],
                                  ppmlim=ppmlim,
                                  sdlimit=sdlimit,
                                  html=report)

    goodFIDs = np.asarray(goodFIDs).T
    goodFIDs = goodFIDs.reshape([1, 1, 1] + list(goodFIDs.shape))

    good_out = NIFTI_MRS(
        goodFIDs,
        header=data.header)

    if len(badFIDs) > 0:
        badFIDs = np.asarray(badFIDs).T
        badFIDs = badFIDs.reshape([1, 1, 1] + list(badFIDs.shape))
        bad_out = NIFTI_MRS(
            badFIDs,
            header=data.header)
    else:
        bad_out = None

    return good_out, bad_out


def phase_correct(data, ppmlim, hlsvd=True, report=None, report_all=False):
    '''Zero-order phase correct based on peak maximum

    :param NIFTI_MRS data: Data to truncate or pad
    :param float ppmlim: Search for peak between limits
    :param bool hlsvd: Use HLSVD to remove peaks outside the ppmlim
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Phased data in NIFTI_MRS format.
    '''

    phs_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        phs_obj[idx], _, pos = preproc.phaseCorrect(
            dd,
            data.bandwidth,
            data.spectrometer_frequency[0],
            nucleus=data.nucleus[0],
            ppmlim=ppmlim,
            use_hlsvd=hlsvd)

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.phasing import phaseCorrect_report
            phaseCorrect_report(dd,
                                phs_obj[idx],
                                pos,
                                data.bandwidth,
                                data.spectrometer_frequency[0],
                                nucleus=data.nucleus[0],
                                ppmlim=ppmlim,
                                html=report)

    return phs_obj


def apply_fixed_phase(data, p0, p1=0.0, report=None, report_all=False):
    '''Apply fixed phase correction

    :param NIFTI_MRS data: Data to truncate or pad
    :param float p0: Zero order phase correction in degrees
    :param float p0: First order phase correction in seconds
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Phased data in NIFTI_MRS format.
    '''
    phs_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        phs_obj[idx] = preproc.applyPhase(dd,
                                          p0 * (np.pi / 180.0))

        if p1 != 0.0:
            phs_obj[idx], _ = preproc.timeshift(
                phs_obj[idx],
                data.dwelltime,
                p1,
                p1,
                samples=data.shape[3])

        if report and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.general import generic_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            generic_report(dd,
                           phs_obj[idx],
                           original_hdr,
                           original_hdr,
                           ppmlim=(0.2, 4.2),
                           html=report,
                           function='fixed phase')

    return phs_obj


def subtract(data0, data1=None, dim=None, report=None, report_all=False):
    '''Either subtract data1 from data0 or subtract index 1 from
     index 0 along specified dimension

    :param NIFTI_MRS data: Data to truncate or pad
    :param data1: If specified data1 will be subtracted from data0
    :param dim: If specified index 1 will be subtracted from 0 across this dimension.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Subtracted data in NIFTI_MRS format.
    '''

    if dim is not None:
        # Check dim is of correct size
        if data0.shape[data0.dim_position(dim)] != 2:
            raise DimensionsDoNotMatch('Subtraction dimension must be of length 2.'
                                       f' Currently {data0.shape[data0.dim_position(dim)]}')

        sub_ob = data0.copy(remove_dim=dim)
        for dd, idx in data0.iterate_over_dims(dim=dim,
                                               iterate_over_space=True,
                                               reduce_dim_index=True):
            sub_ob[idx] = preproc.subtract(dd.T[0], dd.T[1])

            if report and (report_all or first_index(idx)):
                from fsl_mrs.utils.preproc.general import add_subtract_report
                add_subtract_report(dd.T[0],
                                    dd.T[1],
                                    sub_ob[idx],
                                    data0.bandwidth,
                                    data0.spectrometer_frequency[0],
                                    nucleus=data0.nucleus[0],
                                    ppmlim=(0.2, 4.2),
                                    html=report,
                                    function='subtract')

    elif data1 is not None:

        sub_ob = data0.copy()
        sub_ob[:] = (data0[:] - data1[:]) / 2

    else:
        raise ValueError('One of data1 or dim arguments must not be None.')

    return sub_ob


def add(data0, data1=None, dim=None, report=None, report_all=False):
    '''Either add data1 to data0 or add index 1 to
     index 0 along specified dimension

    :param NIFTI_MRS data: Data to truncate or pad
    :param data1: If specified data1 will be added to data0
    :param dim: If specified index 1 will be added to 0 across this dimension.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Subtracted data in NIFTI_MRS format.
    '''

    if dim is not None:
        # Check dim is of correct size
        if data0.shape[data0.dim_position(dim)] != 2:
            raise DimensionsDoNotMatch('Addition dimension must be of length 2.'
                                       f' Currently {data0.shape[data0.dim_position(dim)]}')

        add_ob = data0.copy(remove_dim=dim)
        for dd, idx in data0.iterate_over_dims(dim=dim,
                                               iterate_over_space=True,
                                               reduce_dim_index=True):
            add_ob[idx] = preproc.add(dd.T[0], dd.T[1])

            if report and (report_all or first_index(idx)):
                from fsl_mrs.utils.preproc.general import add_subtract_report
                add_subtract_report(dd.T[0],
                                    dd.T[1],
                                    add_ob[idx],
                                    data0.bandwidth,
                                    data0.spectrometer_frequency[0],
                                    nucleus=data0.nucleus[0],
                                    ppmlim=(0.2, 4.2),
                                    html=report,
                                    function='add')

    elif data1 is not None:

        add_ob = data0.copy()
        add_ob[:] = (data0[:] + data1[:]) / 2

    else:
        raise ValueError('One of data1 or dim arguments must not be None.')

    return add_ob


def conjugate(data, report=None, report_all=False):
    '''Conjugate the data

    :param NIFTI_MRS data: Data to truncate or pad
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Conjugated data in NIFTI_MRS format.
    '''

    conj_data = data.copy()
    conj_data[:] = conj_data[:].conj()

    if report:
        for dd, idx in data.iterate_over_dims(iterate_over_space=True):
            if report_all or first_index(idx):
                from fsl_mrs.utils.preproc.general import generic_report
                original_hdr = {'bandwidth': data.bandwidth,
                                'centralFrequency': data.spectrometer_frequency[0],
                                'ResonantNucleus': data.nucleus[0]}
                generic_report(dd,
                               conj_data[idx],
                               original_hdr,
                               original_hdr,
                               ppmlim=(0.2, 4.2),
                               html=report,
                               function='conjugate')

    return conj_data
