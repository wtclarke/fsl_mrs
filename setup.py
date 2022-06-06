#!/usr/bin/env python

from setuptools import setup
import versioneer

with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f.readlines()]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fsl_mrs',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='FSL Tools for Spectroscopy',
      author=['Saad Jbabdi', 'William Clarke'],
      author_email=['saad@fmrib.ox.ac.uk', 'william.clarke@ndcn.ox.ac.uk'],
      url='www.fmrib.ox.ac.uk/fsl',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['fsl_mrs',
                'fsl_mrs.scripts',
                'fsl_mrs.core',
                'fsl_mrs.utils',
                'fsl_mrs.utils.mrs_io',
                'fsl_mrs.utils.stats',
                'fsl_mrs.utils.preproc',
                'fsl_mrs.utils.synthetic',
                'fsl_mrs.utils.nifti_mrs_tools',
                'fsl_mrs.dynamic',
                'fsl_mrs.denmatsim',
                'fsl_mrs.mmbasis',
                'fsl_mrs.auxiliary'
                ],
      package_data={'fsl_mrs.denmatsim': ['spinSystems.json'],
                    'fsl_mrs.mmbasis': ['mmbasis.json'],
                    'fsl_mrs.utils.preproc': ['templates/*.html'],
                    'fsl_mrs': ['pkg_data/mrs_fitting_challenge/*/*']},
      install_requires=install_requires,
      entry_points={
          'console_scripts': [
              'fsl_mrs = fsl_mrs.scripts.fsl_mrs:main',
              'fsl_mrsi = fsl_mrs.scripts.fsl_mrsi:main',
              'fsl_dynmrs = fsl_mrs.scripts.fsl_dynmrs:main',
              'fsl_mrs_preproc = fsl_mrs.scripts.fsl_mrs_preproc:main',
              'fsl_mrs_preproc_edit = fsl_mrs.scripts.fsl_mrs_preproc_edit:main',
              'fsl_mrs_proc = fsl_mrs.scripts.fsl_mrs_proc:main',
              'fsl_mrs_sim = fsl_mrs.scripts.fsl_mrs_sim:main',
              'fmrs_stats = fsl_mrs.scripts.fmrs_stats:main',
              'mrs_tools = fsl_mrs.scripts.mrs_tools:main',
              'basis_tools = fsl_mrs.scripts.basis_tools:main',
              'merge_mrs_reports = fsl_mrs.scripts.merge_mrs_reports:main',
              'svs_segment = fsl_mrs.scripts.svs_segment:main',
              'mrsi_segment = fsl_mrs.scripts.mrsi_segment:main',
              'results_to_spectrum = fsl_mrs.scripts.results_to_spectrum:main',
              'fsl_mrs_verify = fsl_mrs.scripts.fsl_mrs_verify:main'
          ]
      }
      )
