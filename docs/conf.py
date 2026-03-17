#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = [
    'source/bibtex/cite.bib',
    'source/bibtex/ref.bib',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False

todo_include_todos = True

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

Argonne = u'Argonne National Laboratory'
project = u'denoise'
copyright = u'2025, ' + Argonne

version = open(os.path.join('..', 'VERSION')).read().strip()
release = version

exclude_patterns = ['_build']

show_authors = False

pygments_style = 'sphinx'

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

html_theme_options = {
    'style_nav_header_background': '#4f8fb8ff',
    'collapse_navigation': False,
    'logo_only': True,
}

html_context = {
    'display_github': True,
    'github_user': 'AISDC',
    'github_repo': 'Noise2Inverse360',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

html_logo = 'source/img/workflow.svg'

htmlhelp_basename = project + 'doc'

latex_elements = {}

latex_documents = [
    ('index',
     project + '.tex',
     project + u' Documentation',
     Argonne, 'manual'),
]

man_pages = [
    ('index', project,
     project + u' Documentation',
     [Argonne],
     1)
]

texinfo_documents = [
    ('index',
     project,
     project + u' Documentation',
     Argonne,
     project,
     'Noise2Inverse CT denoising library.',
     'Miscellaneous'),
]

autodoc_mock_imports = [
    'os',
    'os.path',
    'json',
    'pathlib',
    'sys',
    'numpy',
    'torch',
    'tifffile',
    'tqdm',
    'yaml',
    'albumentations',
    'matplotlib',
    'skimage',
    'scipy',
]
