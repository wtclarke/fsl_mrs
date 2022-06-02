#!/usr/bin/env python

# fsl_mrs_veridy - script to verify fsl_mrs sucessfull installation
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#         Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2022 University of Oxford
# SHBASECOPYRIGHT

# Quick imports

import urllib.request
import shutil
import tarfile
from pathlib import Path
import subprocess

import pandas as pd
import numpy as np

from fsl_mrs import __version__


def main():

    print(f'FSL-MRS version {__version__}.')

    # Verify SVS fitting functionality
    print('Verifying SVS fitting:')
    print('--> Downloading data.')

    data_base_url = 'https://git.fmrib.ox.ac.uk/fsl/fsl_mrs/-/'
    svs_data = {
        'metab.nii.gz': 'raw/master/example_usage/example_data/metab.nii.gz?inline=false',
        'wref.nii.gz': 'raw/master/example_usage/example_data/wref.nii.gz?inline=false',
        'steam_11ms.tar.gz': 'archive/master/fsl_mrs-master.tar.gz?path=example_usage/example_data/steam_11ms',
        'T1.anat.tar.gz': 'archive/master/fsl_mrs-master.tar.gz?path=example_usage/example_data/T1.anat'}

    for file in svs_data:
        curr_url = data_base_url + svs_data[file]
        with urllib.request.urlopen(curr_url) as response, open(file, 'wb') as f:
            shutil.copyfileobj(response, f)

        if 'tar.gz' in file:
            # extract file
            tfile = tarfile.open(file)
            for member in tfile.getmembers():
                # Remove file structure
                if member.isfile():
                    member.name = Path(member.name).name
                    tfile.extract(member, path='./' + file.replace('.tar.gz', ''))
            tfile.close()
            Path(file).unlink()

    print('--> Running svs_segment')

    subprocess.check_call([
        'svs_segment',
        '-a', 'T1.anat',
        'metab.nii.gz'
    ], stdout=subprocess.DEVNULL)

    print('--> Running fsl_mrs')
    t1_path = Path('T1.anat') / 'T1_biascorr.nii.gz'
    subprocess.check_call([
        'fsl_mrs',
        '--data', 'metab.nii.gz',
        '--basis', 'steam_11ms',
        '--output', 'fsl_mrs_test_results',
        '--metab_groups', 'Mac',
        '--combine', 'Cr', 'PCr',
        '--combine', 'NAA', 'NAAG',
        '--combine', 'GPC', 'PCh',
        '--t1', str(t1_path),
        '--tissue_frac', 'segmentation.json',
        '--h2o', 'wref.nii.gz',
        '--report',
    ])

    assert (Path('fsl_mrs_test_results') / 'summary.csv').is_file()
    assert (Path('fsl_mrs_test_results') / 'report.html').is_file()
    assert (Path('fsl_mrs_test_results') / 'voxel_location.png').is_file()

    svs_results = pd.read_csv(Path('fsl_mrs_test_results') / 'summary.csv', index_col=0)
    assert np.isclose(svs_results.loc['NAA+NAAG', 'mM'], 13.18, atol=1E-2)

    print('SVS fitting checks passed')
    print('All done')


if __name__ == '__main__':
    main()
