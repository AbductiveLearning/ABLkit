# -*- coding: utf-8 -*-

import sys
import os
import re

if not 'READTHEDOCS' in os.environ:
    sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath('./ABL/'))

# from sphinx.locale import _
from sphinx_rtd_theme import __version__


project = u'ABL'
slug = re.sub(r'\W+', '-', project.lower())
author = u'Yu-Xuan Huang, Wen-Chao Hu, En-Hao Gao'
version = u'0.1.0'
release = u'0.1.0'
copyright = u'2023, LAMDA, Nanjing University, China'
language = 'en'

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'recommonmark',
    'sphinx_markdown_tables',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
exclude_patterns = []
# locale_dirs = ['locale/']
gettext_compact = False

master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']
pygments_style = 'default'

# intersphinx_mapping = {
#     'rtd': ('https://docs.readthedocs.io/en/latest/', None),
#     'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
# }

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True
}
# html_theme_path = ["../.."]
# html_logo = "demo/static/logo-wordmark-light.svg"
# html_show_sourcelink = True

htmlhelp_basename = slug

# latex_documents = [
#   ('index', '{0}.tex'.format(slug), project, author, 'manual'),
# ]

man_pages = [
    ('index', slug, project, [author], 1)
]

texinfo_documents = [
  ('index', slug, project, author, slug, project, 'Miscellaneous'),
]


# Extensions to theme docs
def setup(app):
    from sphinx.domains.python import PyField
    from sphinx.util.docfields import Field

    app.add_object_type(
        'confval',
        'confval',
        objname='configuration value',
        indextemplate='pair: %s; configuration value',
        doc_field_types=[
            PyField(
                'type',
                label=('Type'),
                has_arg=False,
                names=('type',),
                bodyrolename='class'
            ),
            Field(
                'default',
                label=('Default'),
                has_arg=False,
                names=('default',),
            ),
        ]
    )
