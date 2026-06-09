#!/usr/bin/env python
'''
FSL_MRS results validator for nii, csv & json files;
to be used to compare newer to older versions of the pipeline

Authors: Vasilis Karlaftis      <vasilis.karlaftis@ndcn.ox.ac.uk>

Copyright (C) 2025 University of Oxford
'''

from fsl.data.image import Image
import numpy as np
import json
from pathlib import Path
from pandas import read_csv, isna
import numbers


class FolderTester:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.is_equal = True

    def _is_different(self, filename, message):
        self.is_equal = False
        if self.verbose:
            print(filename, ':', message)

    def compare_images(self, ref, est):
        ref_img = Image(ref)
        est_img = Image(est)

        if ref_img.data.shape != est_img.data.shape:
            self._is_different(ref.name, 'Image size is NOT equal!')
        elif not np.allclose(ref_img.data, est_img.data):
            self._is_different(ref.name, 'Image data are NOT equal!')
        elif ref_img.header != est_img.header:
            self._is_different(ref.name, 'Headers are NOT equal!')

    def compare_csv(self, ref, est):
        ref_csv = read_csv(ref)
        est_csv = read_csv(est)

        if list(ref_csv.keys()) != list(est_csv.keys()):
            self._is_different(ref.name, 'CSV header is NOT equal!')
            return

        for key in ref_csv.keys():
            if len(ref_csv[key]) != len(est_csv[key]):
                self._is_different(ref.name, 'CSV data are NOT equal!')
                return

            if np.issubdtype(ref_csv[key].dtype, np.number):
                if not np.allclose(ref_csv[key], est_csv[key], atol=0.001, equal_nan=True):
                    self._is_different(ref.name, 'CSV data are NOT equal!')
                    return
            else:
                nan_r = isna(ref_csv[key])
                nan_e = isna(est_csv[key])
                if not np.array_equal(nan_r, nan_e) \
                   or not np.array_equal(ref_csv[key][~nan_r], est_csv[key][~nan_e]):
                    self._is_different(ref.name, 'CSV data are NOT equal!')
                    return

    def compare_json(self, ref, est):
        with open(ref) as f:
            ref_json = json.load(f)
        with open(est) as f:
            est_json = json.load(f)

        if list(ref_json.keys()) != list(est_json.keys()):
            self._is_different(ref.name, 'JSON keys are NOT equal!')
            return

        for key in ref_json.keys():
            if not self._compare_json_values(ref_json[key], est_json[key]):
                self._is_different(ref.name, 'JSON files are NOT equal!')
                return

    @staticmethod
    def _compare_json_values(ref_value, est_value):
        if isinstance(ref_value, numbers.Number):
            return np.allclose(ref_value, est_value, atol=0.001)

        if isinstance(ref_value, str):
            return ref_value == est_value

        if len(ref_value) != len(est_value):
            return False

        ref_array = np.asarray(ref_value)
        est_array = np.asarray(est_value)
        if np.issubdtype(ref_array.dtype, np.number):
            return np.allclose(ref_array, est_array, atol=0.001)

        return np.array_equal(ref_array, est_array)

    def _run_subfile_code(self, subfile, corresponding_est_file):
        # skip file if it is the 'basis' symlink
        if corresponding_est_file.exists() is False and subfile.name != 'basis':
            self._is_different(subfile.name, 'File does not exist in estimated path!')
            return
        if subfile.suffix in ['.nii', '.gz']:
            self.compare_images(subfile, corresponding_est_file)
        elif subfile.suffix == '.csv':
            self.compare_csv(subfile, corresponding_est_file)
        elif subfile.suffix == '.json':
            self.compare_json(subfile, corresponding_est_file)

    def compare_folders(self, ref_path, est_path, subdir=False):
        ref_path = Path(ref_path)
        est_path = Path(est_path)
        for file in ref_path.glob('*'):
            if file.name.startswith('.'):
                continue
            if subdir and file.is_dir():
                for subfile in file.glob('*'):
                    if subfile.name.startswith('.'):
                        continue
                    if subfile.is_dir():
                        for subsubfile in subfile.glob('*'):
                            if subsubfile.name.startswith('.'):
                                continue
                            corresponding_est_file = est_path / file.name / subfile.name / subsubfile.name
                            self._run_subfile_code(subsubfile, corresponding_est_file)
                    else:
                        corresponding_est_file = est_path / file.name / subfile.name
                        self._run_subfile_code(subfile, corresponding_est_file)
            else:
                corresponding_est_file = est_path / file.name
                self._run_subfile_code(file, corresponding_est_file)

        return self.is_equal


def compare_folders(ref_path, est_path, subdir=False, verbose=False):
    return FolderTester(verbose=verbose).compare_folders(ref_path, est_path, subdir=subdir)
