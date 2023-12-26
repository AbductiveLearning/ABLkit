import os
import re
import sys

from docutils import nodes
from docutils.parsers.rst import roles


def colored_text_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.inline(rawtext, text, classes=[role])
    return [node], []

roles.register_local_role('green-bold', colored_text_role)
roles.register_local_role('blue-bold', colored_text_role)
roles.register_local_role('yellow-bold', colored_text_role)
roles.register_local_role('green', colored_text_role)
roles.register_local_role('blue', colored_text_role)
roles.register_local_role('yellow', colored_text_role)



if "READTHEDOCS" not in os.environ:
    sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath("./ABL/"))


project = "ABL"
slug = re.sub(r"\W+", "-", project.lower())
project = "ABL-Package"
copyright = "2023, Author"
author = "Author"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    'sphinx.ext.autosummary',
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
exclude_patterns = []
# locale_dirs = ['locale/']
gettext_compact = False

master_doc = "index"
suppress_warnings = ["image.nonlocal_uri"]
pygments_style = "default"

# intersphinx_mapping = {
#     'rtd': ('https://docs.readthedocs.io/en/latest/', None),
#     'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
# }

html_theme = "sphinx_rtd_theme"
html_theme_options = {"display_version": True}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# html_theme_path = ["../.."]
# html_logo = "demo/static/logo-wordmark-light.svg"
# html_show_sourcelink = True

htmlhelp_basename = slug

# latex_documents = [
#   ('index', '{0}.tex'.format(slug), project, author, 'manual'),
# ]

man_pages = [("index", slug, project, [author], 1)]

texinfo_documents = [
    ("index", slug, project, author, slug, project, "Miscellaneous"),
]


# Extensions to theme docs
def setup(app):
    from sphinx.domains.python import PyField
    from sphinx.util.docfields import Field

    app.add_object_type(
        "confval",
        "confval",
        objname="configuration value",
        indextemplate="pair: %s; configuration value",
        doc_field_types=[
            PyField("type", label=("Type"), has_arg=False, names=("type",), bodyrolename="class"),
            Field(
                "default",
                label=("Default"),
                has_arg=False,
                names=("default",),
            ),
        ],
    )
