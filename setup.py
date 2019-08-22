#!/usr/bin/env python

from distutils.core import setup

setup(name='FSL_MRS',
      version='0.1',
      description='FSL Tools for Spectroscopy',
      author=['Saad Jbabdi'],
      author_email=['saad@fmrib.ox.ac.uk'],
      url='www.fmrib.ox.ac.uk/fsl',
      packages=['fsl_mrs','fsl_mrs.utils'],
      package_data={'fsl_mrs.utils':['mrs_report_template.html','metabolites.pickle']},
      install_requires=['numpy','scipy','matplotlib', 'plotly', 'jinja2', 'pandas'],
      scripts=['fsl_mrs/scripts/fsl_mrs','fsl_mrs/scripts/fsl_mrs_sim']
     )
