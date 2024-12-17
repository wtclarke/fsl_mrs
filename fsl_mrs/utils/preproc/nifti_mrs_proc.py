'''nifti_mrs_proc.py - translation layer for using NIFTI_MRS with the operations in
fsl_mrs.utils.preproc.

Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
        Will Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 2021 University of Oxford
SHBASECOPYRIGHT'''
from datetime import datetime

import numpy as np

from fsl_mrs.utils import preproc
from fsl_mrs.core import NIFTI_MRS
from fsl_mrs.core import nifti_mrs as ntools
from fsl_mrs import __version__
from fsl_mrs.utils.misc import shift_FID


class DimensionsDoNotMatch(Exception):
    pass


class OnlySVS(Exception):
    pass


def update_processing_prov(nmrs_obj: NIFTI_MRS, method, details):
    """Insert appropriate processing provenance information into the
    NIfTI-MRS header extension.

    :param nmrs_obj: NIFTI-MRS object which has been modified
    :type nmrs_obj: fsl_mrs.core.NIFTI_MRS
    :param method: [description]
    :type method: str
    :param details: [description]
    :type details: str
    """
    # 1. Check for ProcessingApplied key and create if not present
    if 'ProcessingApplied' in nmrs_obj.hdr_ext:
        current_processing = nmrs_obj.hdr_ext['ProcessingApplied']
    else:
        current_processing = []

    # 2. Form object to append.
    prov_dict = {
        'Time': datetime.now().isoformat(sep='T', timespec='milliseconds'),
        'Program': 'FSL-MRS',
        'Version': __version__,
        'Method': method,
        'Details': details}

    # 3. Append
    current_processing.append(prov_dict)
    nmrs_obj.add_hdr_field('ProcessingApplied', current_processing)


def first_index(idx):
    return all([ii == slice(None, None, None) or ii == 0 for ii in idx])


def coilcombine(
        data,
        reference=None,
        noise=None, covariance=None, no_prewhiten=False,
        figure=False,
        report=None,
        report_all=False):
    '''Coil combine data optionally using reference data.
    :param NIFTI_MRS data: Data to coil combine
    :param NIFTI_MRS reference: reference dataset to calculate weights
    :param noise: Supply noise (NCoils x M) to estimate coil covariance (overridden by no_prewhiten)
    :param covariance: Supply coil-covariance for prewhitening (overridden by noise or no_prewhiten)
    :param no_prewhiten: True to disable prewhitening
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Combined data in NIFTI_MRS format.
    '''
    if reference is not None:
        if data.shape[data.dim_position('DIM_COIL')] != reference.shape[data.dim_position('DIM_COIL')]:
            raise DimensionsDoNotMatch('Reference and data coil dimension does not match.')

    combinedc_obj = data.copy(remove_dim='DIM_COIL')
    coil_dim = data.dim_position('DIM_COIL')
    ncoils = data.shape[coil_dim]

    if noise is not None:
        if noise.shape[0] != ncoils:
            raise ValueError(
                f'Noise must have a first dimension (currently {noise.shape[0]}) = NCoils {ncoils}')

        coil_cov = np.cov(noise)

    elif covariance is not None:
        if covariance.shape[0] != ncoils\
                or covariance.shape[1] != ncoils:
            raise ValueError(
                f'Covariance must be square and equal to the number of coils {ncoils}, '
                f'it is currently {covariance.shape}')
        coil_cov = covariance

    elif np.prod(data.shape[0:3]) > 1:
        # Use multiple voxels to estimate the covariance
        from fsl_mrs.utils.preproc.combine import estimate_noise_cov
        data_array = np.moveaxis(
            data[:],
            (coil_dim, 3),
            (-1, -2))
        data_array = data_array.reshape((-1, ) + data_array.shape[-2:])
        coil_cov = estimate_noise_cov(data_array)
    elif data.ndim > 5:
        # Use multiple dynamics to estimate the covariance
        from fsl_mrs.utils.preproc.combine import estimate_noise_cov
        stacked_data = []
        for dd, idx in data.iterate_over_dims(dim='DIM_COIL',
                                              iterate_over_space=True,
                                              reduce_dim_index=True):
            stacked_data.append(dd)
        stacked_data = np.asarray(stacked_data)
        coil_cov = estimate_noise_cov(stacked_data)
    else:
        coil_cov = None

    if reference is not None:
        # Use .image.shape to exclude any trailing singleton dimensions
        weights_shape = reference.image.shape[:3] + reference.image.shape[4:]
        ref_weights = np.zeros(weights_shape, dtype=complex)
        # Run wSVD on the reference data storing up weights
        for ref, idx in reference.iterate_over_dims(dim='DIM_COIL',
                                                    iterate_over_space=True,
                                                    reduce_dim_index=False):
            # TODO make this without_time an kwarg in iterate_over_dims
            idx_no_t = idx[:3] + idx[4:]
            _, ref_weights[idx_no_t], _ = preproc.combine_FIDs(
                ref,
                'svd_weights',
                do_prewhiten=not no_prewhiten,
                cov=coil_cov)

        # Axes swapping fun for broadcasting along multiple dimensions.
        weighted_data = np.moveaxis(data[:], 3, -1).T * ref_weights.T
        weighted_data = np.moveaxis(weighted_data.T, -1, 3)

        combinedc_obj[:] = np.sum(weighted_data, axis=coil_dim)

        if (figure or report):
            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            for main, idx in data.iterate_over_dims(dim='DIM_COIL',
                                                    iterate_over_space=True,
                                                    reduce_dim_index=True):

                if (report_all or first_index(idx)):
                    fig = combine_FIDs_report(
                        main,
                        combinedc_obj[idx],
                        data.bandwidth,
                        data.spectrometer_frequency[0],
                        data.nucleus[0],
                        ncha=data.shape[data.dim_position('DIM_COIL')],
                        ppmlim=(0.0, 6.0),
                        method='svd',
                        dim='DIM_COIL',
                        html=report)
                    if figure:
                        fig.show()
                if first_index(idx):
                    break

    else:
        # If there is no reference data (or [TODO] supplied weights) then have to run
        # per-voxel wSVD. This is slow for high-res MRSI data
        for main, idx in data.iterate_over_dims(dim='DIM_COIL',
                                                iterate_over_space=True,
                                                reduce_dim_index=True):

            combinedc_obj[idx] = preproc.combine_FIDs(
                list(main.T),
                'svd',
                do_prewhiten=not no_prewhiten,
                cov=coil_cov)

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            fig = combine_FIDs_report(
                main,
                combinedc_obj[idx],
                data.bandwidth,
                data.spectrometer_frequency[0],
                data.nucleus[0],
                ncha=data.shape[data.dim_position('DIM_COIL')],
                ppmlim=(0.0, 6.0),
                method='svd',
                dim='DIM_COIL',
                html=report)
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.coilcombine, '
    if reference is None:
        processing_info += 'reference=None, '
    elif reference.image.dataSource is None:
        processing_info += 'reference=Used but unknown source, '
    else:
        processing_info += f'reference={reference.filename}, '
    processing_info += f'no_prewhiten={no_prewhiten}.'

    update_processing_prov(combinedc_obj, 'RF coil combination', processing_info)

    return combinedc_obj


def average(data, dim, figure=False, report=None, report_all=False):
    '''Average (take the mean) of FIDs across a dimension
    specified by a tag.

    :param NIFTI_MRS data: Data to average
    :param str dim: NIFTI-MRS dimension tag
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Combined data in NIFTI_MRS format.
    '''
    # Check that requested dimension exists and is non-singleton
    # First check carried out in dim_position method
    if data.shape[data.dim_position(dim)] == 1:
        print(f'{dim} dimension is singleton, no averaging performed, returning unmodified input.')
        return data

    combined_obj = data.copy(remove_dim=dim)
    for dd, idx in data.iterate_over_dims(dim=dim,
                                          iterate_over_space=True,
                                          reduce_dim_index=True):
        combined_obj[idx] = preproc.combine_FIDs(dd, 'mean')

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            fig = combine_FIDs_report(dd,
                                      combined_obj[idx],
                                      data.bandwidth,
                                      data.spectrometer_frequency[0],
                                      data.nucleus[0],
                                      ncha=data.shape[data.dim_position(dim)],
                                      ppmlim=(0.0, 6.0),
                                      method=f'Mean along dim = {dim}',
                                      html=report)
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.average, '
    processing_info += f'dim={dim}.'

    update_processing_prov(combined_obj, 'Signal averaging', processing_info)

    return combined_obj


def align(
        data,
        dim,
        window=None,
        target=None,
        ppmlim=None,
        niter=2,
        figure=False,
        report=None,
        report_all=False):
    '''Align frequency and phase of spectra. Can be run across a dimension (specified by a tag), or all spectra
    stored in higher dimensions.

    Optionally define a window size to repeatedly align using hanning-weighted windows of spectra.
    E.g. 4 will perform alignment on spectra formed from a moving window of size 4.

    :param NIFTI_MRS data: Data to align
    :param str dim: NIFTI-MRS dimension tag, or 'all'
    :param int window: Window size.
    :param target: Optional target FID
    :param ppmlim: ppm search limits.
    :param int niter: Number of total iterations
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Combined data in NIFTI_MRS format.
    '''

    aligned_obj = data.copy()

    if dim.lower() == 'all':
        generator = data.iterate_over_spatial()
    else:
        # Check that requested dimension exists and is non-singleton
        # First check carried out in dim_position method
        if data.shape[data.dim_position(dim)] == 1:
            print(f'{dim} dimension is singleton, no alignment performed, returning unmodified input.')
            return data
        generator = data.iterate_over_dims(dim=dim,
                                           iterate_over_space=True,
                                           reduce_dim_index=False)

    mrs = data.mrs()[0]
    for dd, idx in generator:

        if dim == 'all':
            # flatten
            original_shape = dd.shape
            dd = dd.reshape(original_shape[0], -1)

        if window is None:
            # Use original single transient alignment
            out = preproc.phase_freq_align(
                dd.T,
                data.bandwidth,
                data.spectrometer_frequency[0],
                nucleus=data.nucleus[0],
                ppmlim=ppmlim,
                niter=niter,
                verbose=False,
                target=target)

            if dim == 'all':
                aligned_obj[idx], phi, eps = out[0].T.reshape(original_shape), out[1], out[2]
            else:
                aligned_obj[idx], phi, eps = out[0].T, out[1], out[2]

        else:
            # Use iterative windowed alignment
            curr_phs = np.zeros(dd.shape[1])
            curr_eps = np.zeros(dd.shape[1])
            curr_raw = dd.copy()

            mean_eps = 1
            nwiter = 0
            win_size = window
            if target is None:
                set_target = True
            else:
                set_target = False
            while mean_eps > 0.02:
                if win_size % 2:
                    # Odd window size: up the size of the window by two
                    # discard the outer two zeros
                    weighting_func = np.hanning(win_size + 2)
                    weighting_func = weighting_func[1:-1]
                    stride_size = win_size
                else:
                    # Even window size: up the size of the window by three
                    # discard the outer two zeros
                    weighting_func = np.hanning(win_size + 3)
                    weighting_func = weighting_func[1:-1]
                    stride_size = win_size + 1
                half_win = int(win_size / 2)

                # Handle window size 1 case
                if win_size == 1:
                    padded_data = curr_raw
                else:
                    padded_data = np.concatenate(
                        (curr_raw[:, -half_win:], curr_raw[:, :], curr_raw[:, :half_win]),
                        axis=1)

                win_avg_data = np.lib.stride_tricks.sliding_window_view(
                    padded_data,
                    stride_size,
                    axis=1) * weighting_func
                win_avg_data = win_avg_data.mean(axis=-1)

                if set_target:
                    target = curr_raw.mean(axis=1)

                _, phi, eps = preproc.phase_freq_align(
                    win_avg_data.T,
                    data.bandwidth,
                    data.spectrometer_frequency[0],
                    ppmlim=ppmlim,
                    niter=niter,
                    target=target)

                for jdx, fid in enumerate(curr_raw.T):
                    curr_raw.T[jdx] = np.exp(-1j * phi[jdx]) * shift_FID(mrs, fid, eps[jdx])

                curr_phs += phi
                curr_eps += eps
                mean_eps = np.abs(eps).mean()
                nwiter += 1
                print(f'{nwiter}: {np.abs(phi).mean()} deg, {mean_eps} Hz.')
                if nwiter == 30:
                    print('Reached windowed average iteration limit. Stopping.')
                    break

            if dim == 'all':
                aligned_obj[idx], phi, eps = curr_raw.reshape(original_shape), curr_phs, curr_eps
            else:
                aligned_obj[idx], phi, eps = curr_raw, curr_phs, curr_eps

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.align import phase_freq_align_report
            if dim == 'all':
                output_for_report = aligned_obj[idx].reshape(original_shape[0], -1)
            else:
                output_for_report = aligned_obj[idx]
            fig = phase_freq_align_report(dd.T,
                                          output_for_report.T,
                                          phi,
                                          eps,
                                          data.bandwidth,
                                          data.spectrometer_frequency[0],
                                          nucleus=data.nucleus[0],
                                          ppmlim=ppmlim,
                                          html=report)
            if figure:
                for ff in fig:
                    ff.show()

    # Update processing prov
    processing_info = f'{__name__}.align, '
    processing_info += f'dim={dim}, '
    processing_info += f'window={window}, '
    if target is not None:
        processing_info += 'target used, '
    else:
        processing_info += 'target=None, '
    processing_info += f'ppmlim={ppmlim}, '
    processing_info += f'niter={niter}.'

    update_processing_prov(aligned_obj, 'Frequency and phase correction', processing_info)

    return aligned_obj


def aligndiff(data,
              dim_align,
              dim_diff,
              diff_type,
              target=None,
              ppmlim=None,
              figure=False,
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
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Aligned data in NIFTI_MRS format.
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

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.align import phase_freq_align_diff_report
            fig = phase_freq_align_diff_report(d0.T,
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
            if figure:
                for ff in fig:
                    ff.show()

    # Update processing prov
    processing_info = f'{__name__}.aligndiff, '
    processing_info += f'dim_align={dim_align}, '
    processing_info += f'dim_diff={dim_diff}, '
    processing_info += f'diff_type={diff_type}, '
    if target is not None:
        processing_info += 'target used, '
    else:
        processing_info += 'target=None, '
    processing_info += f'ppmlim={ppmlim}.'

    update_processing_prov(aligned_obj, 'Alignment of subtraction sub-spectra', processing_info)

    return aligned_obj


def ecc(data, reference, figure=False, report=None, report_all=False):
    '''Apply eddy current correction using a reference dataset
    :param NIFTI_MRS data: Data to eddy current correct
    :param NIFTI_MRS reference: reference dataset to calculate phase
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Corrected data in NIFTI_MRS format.
    '''
    if data.shape != reference.shape\
            and reference.ndim > 4\
            and np.prod(reference.shape[4:]) > 1:
        raise DimensionsDoNotMatch('Reference and data shape must match'
                                   ' or reference must be single FID.')

    corrected_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):

        if data.shape == reference.shape:
            # Reference is the same shape as data, voxel-wise and spectrum-wise iteration
            ref = reference[idx]
        else:
            # Only one reference FID, only iterate over spatial voxels.
            ref = reference[idx[0], idx[1], idx[2], :]

        corrected_obj[idx] = preproc.eddy_correct(dd, ref)

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.eddycorrect import eddy_correct_report
            fig = eddy_correct_report(dd,
                                      corrected_obj[idx],
                                      ref,
                                      data.bandwidth,
                                      data.spectrometer_frequency[0],
                                      nucleus=data.nucleus[0],
                                      html=report)
            if figure:
                for ff in fig:
                    ff.show()

    # Update processing prov
    processing_info = f'{__name__}.ecc, '
    processing_info += f'reference={reference.filename}.'

    update_processing_prov(corrected_obj, 'Eddy current correction', processing_info)

    return corrected_obj


def remove_peaks(data, limits, limit_units='ppm+shift', figure=False, report=None, report_all=False):
    '''Apply HLSVD to remove peaks from specta
    :param NIFTI_MRS data: Data to remove peaks from
    :param limits: ppm limits between which peaks will be removed
    :param str limit_units: Can be 'Hz', 'ppm' or 'ppm+shift'.
    :param figure: True to show figure.
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

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.remove import hlsvd_report
            fig = hlsvd_report(dd,
                               corrected_obj[idx],
                               limits,
                               data.bandwidth,
                               data.spectrometer_frequency[0],
                               nucleus=data.nucleus[0],
                               limitUnits=limit_units,
                               html=report)
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.remove_peaks, '
    processing_info += f'limits={limits}, '
    processing_info += f'limit_units={limit_units}.'

    update_processing_prov(corrected_obj, 'Nuisance peak removal', processing_info)

    return corrected_obj


def hlsvd_model_peaks(data, limits,
                      limit_units='ppm+shift', components=5, figure=False, report=None, report_all=False):
    '''Apply HLSVD to model spectum
    :param NIFTI_MRS data: Data to model
    :param limits: ppm limits between which spectrum will be modeled
    :param str limit_units: Can be 'Hz', 'ppm' or 'ppm+shift'.
    :param int components: Number of lorentzian components to model
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Corrected data in NIFTI_MRS format.
    '''
    corrected_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):

        corrected_obj[idx] = preproc.model_fid_hlsvd(
            dd,
            data.dwelltime,
            data.spectrometer_frequency[0],
            limits,
            limitUnits=limit_units,
            numSingularValues=components)

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.remove import hlsvd_report
            fig = hlsvd_report(dd,
                               corrected_obj[idx],
                               limits,
                               data.bandwidth,
                               data.spectrometer_frequency[0],
                               nucleus=data.nucleus[0],
                               limitUnits=limit_units,
                               html=report)
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.hlsvd_model_peaks, '
    processing_info += f'limits={limits}, '
    processing_info += f'limit_units={limit_units}, '
    processing_info += f'components={components}.'

    update_processing_prov(corrected_obj, 'HLSVD modeling', processing_info)

    return corrected_obj


def tshift(data, tshiftStart=0.0, tshiftEnd=0.0, samples=None, figure=False, report=None, report_all=False):
    '''Apply time shift or resampling to each FID
    :param NIFTI_MRS data: Data to shift
    :param float tshiftStart: Shift start time (s), negative padds with zeros
    :param float tshiftEnd: Shift end time (s), negative truncates
    :param float samples: Resample to this many points
    :param figure: True to show figure.
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

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report

            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            new_hdr = {'bandwidth': 1 / newDT,
                       'centralFrequency': data.spectrometer_frequency[0],
                       'ResonantNucleus': data.nucleus[0]}
            fig = shift_report(dd,
                               shifted_obj[idx],
                               original_hdr,
                               new_hdr,
                               html=report,
                               function='timeshift')
            if figure:
                fig.show()

    shifted_obj.dwelltime = newDT

    # Update processing prov
    processing_info = f'{__name__}.tshift, '
    processing_info += f'tshiftStart={tshiftStart}, '
    processing_info += f'tshiftEnd={tshiftEnd}, '
    processing_info += f'samples={samples}.'

    update_processing_prov(shifted_obj, 'Temporal resample', processing_info)

    return shifted_obj


def truncate_or_pad(data, npoints, position, figure=False, report=None, report_all=False):
    '''Truncate or pad by integer number of points
    :param NIFTI_MRS data: Data to truncate or pad
    :param int npoints: Pad (positive) or truncate (negative) by npoints
    :param str position: 'first' or 'last', add or remove points at the
    start or end of the FID
    :param figure: True to show figure.
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

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}

            fig = shift_report(dd,
                               trunc_obj[idx],
                               original_hdr,
                               original_hdr,
                               html=report,
                               function=rep_func)
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.truncate_or_pad, '
    processing_info += f'npoints={npoints}, '
    processing_info += f'position={position}.'

    update_processing_prov(trunc_obj, 'Zero-filling', processing_info)

    return trunc_obj


def apodize(data, amount, filter='exp', figure=False, report=None, report_all=False):
    '''Apodize FIDs using a exponential or Lorentzian to Gaussian filter.
    Lorentzian to Gaussian filter takes requires two window parameters (t_L and t_G)

    :param NIFTI_MRS data: Data to truncate or pad
    :param tuple amount: If filter='exp' single valued. If filter='l2g' then two valued.
    :param str filter: 'exp' or 'l2g'. Choose exponential or Lorentzian to Gaussian filter
    :param figure: True to show figure.
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

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.filtering import apodize_report
            fig = apodize_report(dd,
                                 apod_obj[idx],
                                 data.bandwidth,
                                 data.spectrometer_frequency[0],
                                 nucleus=data.nucleus[0],
                                 html=report)
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.apodize, '
    processing_info += f'amount={amount}, '
    processing_info += f'filter={filter}.'

    update_processing_prov(apod_obj, 'Apodization', processing_info)

    return apod_obj


def fshift(data, amount, figure=False, report=None, report_all=False):
    '''Apply frequency shift.

    Two modes of operation:
    1) Specify a single shift which is applied to all FIDs/spectra - amount has float type
    2) Specify a shift per FID/spectra - amount is numpy array matching data shape

    :param NIFTI_MRS data: Data to frequency shift
    :param amount: Shift amount in Hz, can be array matching data size
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Shifted data in NIFTI_MRS format.
    '''
    if not isinstance(amount, float) and amount.size > 1:
        required_shape = data.shape[:3] + data.shape[4:]
        if amount.shape != required_shape:
            raise ValueError(
                'Shift map must be the same size as the NIfTI-MRS spatial + higher dimensions. '
                f'Current size = {amount.shape}, required shape = {required_shape}.')
        shift_map = True
    else:
        shift_map = False
        toshift = amount

    shift_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        if shift_map:
            toshift = amount[idx[:3] + idx[4:]]
        shift_obj[idx] = preproc.freqshift(dd,
                                           data.dwelltime,
                                           toshift)

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            fig = shift_report(dd,
                               shift_obj[idx],
                               original_hdr,
                               original_hdr,
                               html=report,
                               function='freqshift')
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.fshift, '
    if shift_map:
        processing_info += 'amount=per-voxel shifts specified.'
    else:
        processing_info += f'amount={amount}.'

    update_processing_prov(shift_obj, 'Frequency and phase correction', processing_info)

    return shift_obj


def shift_to_reference(data, ppm_ref, peak_search, use_avg=False, figure=False, report=None, report_all=False):
    '''Shift peak to known reference

    :param NIFTI_MRS data: Data to truncate or pad
    :param float ppm_ref: Reference shift that peak will be moved to
    :param tuple peak_search: Search for peak between these ppm limits e.g. (2.8, 3.2) for tCr
    :param bool use_avg: If multiple spectra in higher dimensions,
        use the average of all the higher dimension spectra to calculate shift correction.
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Shifted data in NIFTI_MRS format.
    '''

    shift_obj = data.copy()
    if use_avg:
        # Combine all higher dimensions
        shift = np.zeros(data.shape[:3])
        for dd, idx in data.iterate_over_spatial():
            comb_data = preproc.combine_FIDs(
                dd.reshape(dd.shape[0], -1),
                'svd',
                do_prewhiten=False)
            # Run shift estimation
            _, shift[idx[:3]] = preproc.shiftToRef(
                comb_data,
                ppm_ref,
                data.bandwidth,
                data.spectrometer_frequency[0],
                nucleus=data.nucleus[0],
                ppmlim=peak_search)

    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        if use_avg:
            shift_obj[idx] = preproc.freqshift(
                dd,
                data.dwelltime,
                - shift[idx[:3]] * data.spectrometer_frequency[0])
        else:
            shift_obj[idx], _ = preproc.shiftToRef(
                dd,
                ppm_ref,
                data.bandwidth,
                data.spectrometer_frequency[0],
                nucleus=data.nucleus[0],
                ppmlim=peak_search)

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.shifting import shift_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            fig = shift_report(dd,
                               shift_obj[idx],
                               original_hdr,
                               original_hdr,
                               html=report,
                               function='shiftToRef')
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.shift_to_reference, '
    processing_info += f'ppm_ref={ppm_ref}, '
    processing_info += f'peak_search={peak_search}, '
    processing_info += f'use_avg={use_avg}.'

    update_processing_prov(shift_obj, 'Frequency and phase correction', processing_info)

    return shift_obj


def remove_unlike(data, ppmlim=None, sdlimit=1.96, niter=2, figure=False, report=None):
    '''Remove unlike dynamics operating on DIM_DYN

    :param NIFTI_MRS data: Data to truncate or pad
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report

    :return: Data passing likeness criteria.
    :return: Data failing likness criteria
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

    if figure or report:
        from fsl_mrs.utils.preproc.unlike import identifyUnlikeFIDs_report
        fig = identifyUnlikeFIDs_report(goodFIDs,
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
        if figure:
            fig.show()

    # goodFIDs = np.asarray(goodFIDs).T
    # goodFIDs = goodFIDs.reshape([1, 1, 1] + list(goodFIDs.shape))

    if len(badFIDs) > 0:
        bad_out, good_out  = ntools.split(
            data,
            data.dim_tags[0],
            gIndicies)
    else:
        good_out = data.copy()

    good_out.add_hdr_field(
        f'{data.dim_tags[0]} Indices',
        gIndicies,
        doc=f"Data's original index values in the {data.dim_tags[0]} dimension")

    if len(badFIDs) > 0:
        bad_out.add_hdr_field(
            f'{data.dim_tags[0]} Indices',
            bIndicies,
            doc=f"Data's original index values in the {data.dim_tags[0]} dimension")
    else:
        bad_out = None

    # Update processing prov
    processing_info = f'{__name__}.remove_unlike, '
    if ppmlim is None:
        processing_info += 'ppmlim=None, '
    else:
        processing_info += f'ppmlim={ppmlim}, '
    processing_info += f'sdlimit={sdlimit}, '
    processing_info += f'niter={niter}.'

    update_processing_prov(good_out, 'Outlier removal', processing_info)

    return good_out, bad_out


def phase_correct(data, ppmlim, hlsvd=False, use_avg=False, figure=False, report=None, report_all=False):
    '''Zero-order phase correct based on peak maximum

    :param NIFTI_MRS data: Data to truncate or pad
    :param float ppmlim: Search for peak between limits
    :param bool hlsvd: Use HLSVD to remove peaks outside the ppmlim
    :param bool use_avg: If multiple spectra in higher dimensions,
        use the average of all the higher dimension spectra to calculate phase correction.
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Phased data in NIFTI_MRS format.
    '''

    phs_obj = data.copy()
    if use_avg:
        # Combine all higher dimensions
        p0 = np.zeros(data.shape[:3])
        pos_all = np.zeros(data.shape[:3], int)
        for dd, idx in data.iterate_over_spatial():
            comb_data = preproc.combine_FIDs(
                dd.reshape(dd.shape[0], -1),
                'svd',
                do_prewhiten=False)
            # Run phase correction estimation
            _, p0[idx[:3]], pos_all[idx[:3]] = preproc.phaseCorrect(
                comb_data,
                phs_obj.bandwidth,
                phs_obj.spectrometer_frequency[0],
                nucleus=phs_obj.nucleus[0],
                ppmlim=ppmlim,
                use_hlsvd=hlsvd)

    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        if use_avg:
            phs_obj[idx] = preproc.applyPhase(
                dd,
                p0[idx[:3]])
            pos = pos_all[idx[:3]]
        else:
            phs_obj[idx], _, pos = preproc.phaseCorrect(
                dd,
                data.bandwidth,
                data.spectrometer_frequency[0],
                nucleus=data.nucleus[0],
                ppmlim=ppmlim,
                use_hlsvd=hlsvd)

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.phasing import phaseCorrect_report
            fig = phaseCorrect_report(dd,
                                      phs_obj[idx],
                                      pos,
                                      data.bandwidth,
                                      data.spectrometer_frequency[0],
                                      nucleus=data.nucleus[0],
                                      ppmlim=ppmlim,
                                      html=report)
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.phase_correct, '
    processing_info += f'ppmlim={ppmlim}, '
    processing_info += f'hlsvd={hlsvd}, '
    processing_info += f'use_avg={use_avg}.'

    update_processing_prov(phs_obj, 'Phasing', processing_info)

    return phs_obj


def apply_fixed_phase(data, p0, p1=0.0, p1_type='shift', figure=False, report=None, report_all=False):
    '''Apply fixed phase correction

    :param NIFTI_MRS data: Data to truncate or pad
    :param float p0: Zero order phase correction in degrees
    :param float p1: First order phase correction in seconds
    :param str p1_type: 'shift' for interpolated time-shift, 'linphase' for frequency-domain phasing.
    :param figure: True to show figure.
    :param report: Provide output location as path to generate report
    :param report_all: True to output all indicies

    :return: Phased data in NIFTI_MRS format.
    '''
    phs_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):
        phs_obj[idx] = preproc.applyPhase(dd,
                                          p0 * (np.pi / 180.0))

        if p1 != 0.0:
            if p1_type.lower() == 'shift':
                phs_obj[idx], _ = preproc.timeshift(
                    phs_obj[idx],
                    data.dwelltime,
                    p1,
                    p1,
                    samples=data.shape[3])
            elif p1_type.lower() == 'linphase':
                from fsl_mrs.utils.misc import calculateAxes
                faxis = calculateAxes(
                    data.spectralwidth,
                    data.spectrometer_frequency[0],
                    data.shape[3],
                    0.0)['freq']
                phs_obj[idx] = preproc.applyLinPhase(
                    phs_obj[idx],
                    faxis,
                    p1)
            else:
                raise ValueError("p1_type kwarg must be 'shift' or 'linphase'.")

        if (figure or report) and (report_all or first_index(idx)):
            from fsl_mrs.utils.preproc.general import generic_report
            original_hdr = {'bandwidth': data.bandwidth,
                            'centralFrequency': data.spectrometer_frequency[0],
                            'ResonantNucleus': data.nucleus[0]}
            fig = generic_report(dd,
                                 phs_obj[idx],
                                 original_hdr,
                                 original_hdr,
                                 ppmlim=(0.2, 4.2),
                                 html=report,
                                 function='fixed phase')
            if figure:
                fig.show()

    # Update processing prov
    processing_info = f'{__name__}.apply_fixed_phase, '
    processing_info += f'p0={p0}, '
    processing_info += f'p1={p1}, '
    processing_info += f'p1_type={p1_type}.'

    update_processing_prov(phs_obj, 'Phasing', processing_info)

    return phs_obj


def subtract(data0, data1=None, dim=None, figure=False, report=None, report_all=False):
    '''Either subtract data1 from data0 or subtract index 1 from
     index 0 along specified dimension

    :param NIFTI_MRS data: Data to truncate or pad
    :param data1: If specified data1 will be subtracted from data0
    :param dim: If specified index 1 will be subtracted from 0 across this dimension.
    :param figure: True to show figure.
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

            if (figure or report) and (report_all or first_index(idx)):
                from fsl_mrs.utils.preproc.general import add_subtract_report
                fig = add_subtract_report(dd.T[0],
                                          dd.T[1],
                                          sub_ob[idx],
                                          data0.bandwidth,
                                          data0.spectrometer_frequency[0],
                                          nucleus=data0.nucleus[0],
                                          ppmlim=(0.2, 4.2),
                                          html=report,
                                          function='subtract')
                if figure:
                    fig.show()

    elif data1 is not None:

        sub_ob = data0.copy()
        sub_ob[:] = (data0[:] - data1[:]) / 2

    else:
        raise ValueError('One of data1 or dim arguments must not be None.')

    # Update processing prov
    processing_info = f'{__name__}.subtract, '
    if data1 is None:
        processing_info += 'data1=None, '
    else:
        processing_info += f'data1={data1.filename}, '
    if dim is None:
        processing_info += 'dim=None.'
    else:
        processing_info += f'dim={dim}.'

    update_processing_prov(sub_ob, 'Subtraction of sub-spectra', processing_info)

    return sub_ob


def add(data0, data1=None, dim=None, figure=False, report=None, report_all=False):
    '''Either add data1 to data0 or add index 1 to
     index 0 along specified dimension

    :param NIFTI_MRS data: Data to truncate or pad
    :param data1: If specified data1 will be added to data0
    :param dim: If specified index 1 will be added to 0 across this dimension.
    :param figure: True to show figure.
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

            if (figure or report) and (report_all or first_index(idx)):
                from fsl_mrs.utils.preproc.general import add_subtract_report
                fig = add_subtract_report(dd.T[0],
                                          dd.T[1],
                                          add_ob[idx],
                                          data0.bandwidth,
                                          data0.spectrometer_frequency[0],
                                          nucleus=data0.nucleus[0],
                                          ppmlim=(0.2, 4.2),
                                          html=report,
                                          function='add')
                if figure:
                    fig.show()

    elif data1 is not None:

        add_ob = data0.copy()
        add_ob[:] = (data0[:] + data1[:]) / 2

    else:
        raise ValueError('One of data1 or dim arguments must not be None.')

    # Update processing prov
    processing_info = f'{__name__}.add, '
    if data1 is None:
        processing_info += 'data1=None, '
    else:
        processing_info += f'data1={data1.filename}, '
    if dim is None:
        processing_info += 'dim=None.'
    else:
        processing_info += f'dim={dim}.'

    update_processing_prov(add_ob, 'Addition of sub-spectra', processing_info)

    return add_ob


def conjugate(data, figure=False, report=None, report_all=False):
    '''Conjugate the data

    :param NIFTI_MRS data: Data to truncate or pad
    :param figure: True to show figure.
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
                fig = generic_report(dd,
                                     conj_data[idx],
                                     original_hdr,
                                     original_hdr,
                                     ppmlim=(0.2, 4.2),
                                     html=report,
                                     function='conjugate')
                if figure:
                    fig.show()

    # Update processing prov
    processing_info = f'{__name__}.conjugate.'
    update_processing_prov(conj_data, 'Conjugation', processing_info)

    return conj_data
