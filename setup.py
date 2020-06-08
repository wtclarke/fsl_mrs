#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fsl_mrs',
      version='1.0.0',
      description='FSL Tools for Spectroscopy',
      author=['Saad Jbabdi','William Clarke'],
      author_email=['saad@fmrib.ox.ac.uk','william.clarke@ndcn.ox.ac.uk'],
      url='www.fmrib.ox.ac.uk/fsl',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['fsl_mrs',
                'fsl_mrs.core',
                'fsl_mrs.utils',
                'fsl_mrs.utils.mrs_io',
                'fsl_mrs.utils.stats',
                'fsl_mrs.utils.preproc',
                'fsl_mrs.utils.synthetic',
                'fsl_mrs.denmatsim',
                'fsl_mrs.mmbasis',
                'fsl_mrs.aux'
                ],
      package_data={'fsl_mrs.denmatsim':['spinSystems.json'],
                    'fsl_mrs.mmbasis':['mmbasis.json'],
                    'fsl_mrs.utils.preproc':['templates/*.html'],
                    'fsl_mrs':['pkg_data/mrs_fitting_challenge/*/*']},
      install_requires=install_requires,
      scripts=['fsl_mrs/scripts/fsl_mrs',
                'fsl_mrs/scripts/fsl_mrsi',
                'fsl_mrs/scripts/fsl_mrs_preproc',
                'fsl_mrs/scripts/fsl_mrs_proc',
                'fsl_mrs/scripts/fsl_mrs_sim',
                'fsl_mrs/scripts/mrs_vis',
                'fsl_mrs/scripts/merge_mrs_reports',
                'fsl_mrs/scripts/svs_segment',
                'fsl_mrs/scripts/mrsi_segment']               
     )
