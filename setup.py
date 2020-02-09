#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]


setup(name='FSL_MRS',
      version='1.0.0',
      description='FSL Tools for Spectroscopy',
      author=['Saad Jbabdi'],
      author_email=['saad@fmrib.ox.ac.uk'],
      url='www.fmrib.ox.ac.uk/fsl',
      packages=['fsl_mrs','fsl_mrs.utils','fsl_mrs.denmatsim'],
      package_data={'fsl_mrs.utils':['mrs_report_template.html','metabolites.pickle'],'fsl_mrs.denmatsim':['spinSystems.json']},
      install_requires=install_requires,
      scripts=['fsl_mrs/scripts/fsl_mrs','fsl_mrs/scripts/fsl_mrsi','fsl_mrs/scripts/fsl_mrs_preproc','fsl_mrs/scripts/fsl_mrs_sim','fsl_mrs/scripts/mrs_vis']               
     )
