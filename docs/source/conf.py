# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Baytomo'
copyright = '2024, Kuan-Yu Ke'
author = 'Kuan-Yu Ke'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))
#extensions = []
#extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon','sphinx_autodoc_typehints',]
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx_autodoc_typehints',
              'sphinx.ext.mathjax'
              ]

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
    ''',
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise
    "github_user": 'kuanyuke',
    "github_repo": 'Baytomo',
    "github_version": "main",
    "doc_path": "docs",
    'fixed_sidebar': True,
    'github_button': True,
}

html_static_path = ['_static']
numfig = True
numtab = True
