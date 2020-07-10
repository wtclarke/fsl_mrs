import datetime
date = datetime.date.today()

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'FSL-MRS'
copyright = f'{date.year}, Will Clarke & Saad Jbabdi, University of Oxford, Oxford, UK'
author = 'William Clarke'

# The full version, including alpha/beta/rc tags
version = '1.0.2'
release = version

# From PM's fsleyes doc
# Things which I want to be able to
# quote in the documentation go here.
rst_epilog = """

.. |fsl_version|     replace:: 6.0.4

.. |fslmrs_gitlab| replace:: FSL-MRS GitLab
.. _fslmrs_gitlab: https://git.fmrib.ox.ac.uk/fsl/fsl_mrs

.. |fslmrs_github_tracker| replace:: FSL-MRS GitHub Issue Tracker
.. _fslmrs_github_tracker: https://github.com/wexeee/fsl_mrs/issues

.. |fslmrs_pkg_data| replace:: FSL-MRS example data
.. _fslmrs_pkg_data: https://users.fmrib.ox.ac.uk/~wclarke/fsl_mrs/example_usage.zip

.. |dev_email| replace:: developers
.. _dev_email: mailto:william.clarke@ndcn.ox.ac.uk,saad@fmrib.ox.ac.uk
"""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
        ],
     }
