'''nifti_mrs_proc.py - translation layer for using NIFTI_MRS with the operations in
fsl_mrs.utils.preproc.

Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
        Will Clarke <william.clarke@ndcn.ox.ac.uk>

Copyright (C) 2021 University of Oxford
SHBASECOPYRIGHT'''
from fsl_mrs.utils import preproc


class DimensionsDoNotMatch(Exception):
    pass


def coilcombine(data, reference=None, no_prewhiten=False, report=None, report_all=False):
    '''Coil combine data optionally using reference data.
    :param NIFTI_MRS data: Data to coil combine
    :param NIFTI_MRS reference: reference dataset to calculate weights
    :param no_prewhiten: True to disable prewhitening
    :param report: Provide path to generate
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

        if report and (report_all or idx == 0):
            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            combine_FIDs_report(list(main.T),
                                list(combinedc_obj[idx]),
                                data.bandwidth,
                                data.spectrometer_frequency,
                                data.nucleus,
                                ncha=data.shape[data.dim_position('DIM_COIL')],
                                ppmlim=(0.0, 6.0),
                                method='svd',
                                html=report)
    return combinedc_obj
