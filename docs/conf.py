import os
import re
import sys

from docutils import nodes
from docutils.parsers.rst import roles

from sphinx.application import Sphinx

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))

from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = ["numpy", "pyswip", "torch", "torchvision", "zoopt", "termcolor"]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


import ablkit  # noqa: E402

# -- Project information -----------------------------------------------------

project = "ABL Kit"
copyright = "LAMDA, 2024"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
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
gettext_compact = False

master_doc = "index"
suppress_warnings = ["image.nonlocal_uri"]
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {"display_version": True}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

slug = re.sub(r"\W+", "", project.lower())
htmlhelp_basename = slug
man_pages = [("index", slug, project, 1)]
texinfo_documents = [
    ("index", slug, project, slug, project, "Miscellaneous"),
]

copybutton_selector = "div:not(.code-out) > div.highlight > pre"


def remove_noqa(app: Sphinx, what: str, name: str, obj, options, lines):
    new_lines = []
    for line in lines:
        new_line = re.sub(r"\s*#\s*noqa.*$", "", line)
        new_lines.append(new_line)
    lines[:] = new_lines


def colored_text_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.inline(rawtext, text, classes=[role])
    return [node], []


roles.register_local_role("green-bold", colored_text_role)
roles.register_local_role("blue-bold", colored_text_role)
roles.register_local_role("yellow-bold", colored_text_role)
roles.register_local_role("green", colored_text_role)
roles.register_local_role("blue", colored_text_role)
roles.register_local_role("yellow", colored_text_role)


# Extensions to theme docs
def setup(app):
    from sphinx.domains.python import PyField
    from sphinx.util.docfields import Field

    app.connect("autodoc-process-docstring", remove_noqa)
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
