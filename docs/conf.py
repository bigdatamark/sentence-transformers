# [MODULE DOC]
# Sphinx configuration for the Sentence Transformers documentation site.
# Sets project metadata (copyright year taken from the current date at build time), the
# extension list (napoleon, autodoc, myst, markdown tables, copy button, intersphinx,
# linkcode, inline tabs, mermaid, collapse), include patterns, intersphinx targets for
# Hugging Face libraries and PyTorch, and the Read-the-Docs HTML theme with custom
# static CSS/JS, logo and favicon.
# It also defines hooks: a linkcode resolver that maps documented objects to GitHub source
# line ranges, a monkey-patched HTML download-link visitor that points downloads at GitHub,
# and autodoc docstring preprocessors registered in setup.
# Side effects at import time: replaces the HTML5 translator's download reference visitor globally.
# KNOWN ISSUE: the include pattern "sentence_transformers/**/.py" matches only files literally named ".py"; "*.py" was probably intended.
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

import datetime
import importlib
import inspect
import os
import posixpath
import re

import sphinx.ext.autodoc
from sphinx.application import Sphinx
from sphinx.writers.html5 import HTML5Translator

# -- Project information -----------------------------------------------------

project = "Sentence Transformers"
copyright = str(datetime.datetime.now().year)
author = "Nils Reimers, Tom Aarsen"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
    "sphinx_toolbox.collapse",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to include when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
include_patterns = [
    "docs/**",
    "sentence_transformers/**/.py",
    "examples/**",
    "index.rst",
]

autodoc_inherit_docstrings = True

intersphinx_mapping = {
    "datasets": ("https://huggingface.co/docs/datasets/main/en/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "huggingface_hub": ("https://huggingface.co/docs/huggingface_hub/main/en/", None),
    "optimum": ("https://huggingface.co/docs/optimum/main/en/", None),
    "optimum-onnx": ("https://huggingface.co/docs/optimum-onnx/main/en/", None),
    "peft": ("https://huggingface.co/docs/peft/main/en/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torchcodec": ("https://meta-pytorch.org/torchcodec/stable/", None),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": True,
    "canonical_url": "https://www.sbert.net",
    "collapse_navigation": False,
    "navigation_depth": 3,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "img/hf-logo.svg"]

# Add any paths that contain "extra" files, such as .htaccess or
# robots.txt.
html_extra_path = [".htaccess"]

html_css_files = [
    "css/custom.css",
]

html_js_files = [
    "js/custom.js",
]

html_show_sourcelink = False
html_context = {
    "display_github": True,
    "github_user": "huggingface",
    "github_repo": "sentence-transformers",
    "github_version": "main/",
}

html_logo = "img/logo.png"
html_favicon = "img/favicon.ico"

autoclass_content = "both"

# Required to get rid of some myst.xref_missing warnings
myst_heading_anchors = 3


# https://github.com/readthedocs/sphinx-autoapi/issues/202#issuecomment-907582382
def linkcode_resolve(domain, info):
    # [FUNCTION DOC]
    # Sphinx linkcode callback. Inputs: the domain name and an info mapping with "module" and
    # "fullname". Returns None for JavaScript objects or the "connect4" tutorial module; asserts the
    # domain is Python. Imports the module, resolves the object (for a dotted fullname, the attribute
    # of a top-level object; returns None if that attribute is missing, i.e. a plain class attribute),
    # unwraps decorators, and finds its source file and line range. Returns None if source lookup raises
    # a type error or the file path, made relative to the parent of the working directory, is not under
    # sentence_transformers. Otherwise returns a GitHub URL on the main branch with a start-end line anchor.
    # Side effects: imports arbitrary modules.
    # KNOWN ISSUE: a fullname with more than one dot (e.g. a nested class method) makes the two-way split raise a ValueError instead of being handled.
    # Non-linkable objects from the starter kit in the tutorial.
    if domain == "js" or info["module"] == "connect4":
        return

    assert domain == "py", "expected only Python objects"

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])
    obj = inspect.unwrap(obj)

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    if not file.startswith("sentence_transformers"):
        # e.g. object is a typing.NewType
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"https://github.com/huggingface/sentence-transformers/blob/main/{file}#L{start}-L{end}"


def visit_download_reference(self, node):
    # [FUNCTION DOC]
    # Replacement for the HTML5 translator's download-reference visitor (self is the translator, node
    # the docutils node). If the builder has no download support, pushes an empty closing tag. If the node
    # has an external URI, emits an anchor to it; if it has a target and source document, emits an anchor
    # to the file under the GitHub tree URL (document directory joined with target). Otherwise pushes an
    # empty closer. Anchors get class "reference download external" and a download attribute; the matching
    # closing tag is pushed onto the translator context. Mutates the translator body and context.
    root = "https://github.com/huggingface/sentence-transformers/tree/main"
    atts = {"class": "reference download", "download": ""}

    if not self.builder.download_support:
        self.context.append("")
    elif "refuri" in node:
        atts["class"] += " external"
        atts["href"] = node["refuri"]
        self.body.append(self.starttag(node, "a", "", **atts))
        self.context.append("</a>")
    elif "reftarget" in node and "refdoc" in node:
        atts["class"] += " external"
        atts["href"] = posixpath.join(root, os.path.dirname(node["refdoc"]), node["reftarget"])
        self.body.append(self.starttag(node, "a", "", **atts))
        self.context.append("</a>")
    else:
        self.context.append("")


HTML5Translator.visit_download_reference = visit_download_reference


def preprocess_docstring(app, obj_type: str, name: str, obj, options: sphinx.ext.autodoc.Options, lines: list[str]):
    # [FUNCTION DOC]
    # autodoc-process-docstring handler that rewrites the docstring lines list in place.
    # Inputs: the app, the object type, the full dotted name, the object, autodoc options and the lines.
    # Steps: (1) for functions, decorators, methods, properties and attributes documented under
    # sentence_transformers whose real module is in transformers, prepends a note directive saying the
    # docstring is inherited from the transformers object of the same qualified name, plus a blank line.
    # (2) Converts Markdown triple-backtick fences to rST code-block directives (language defaults to
    # python, "py" mapped to python), keeping the fence indentation, adding a blank indented line, and
    # re-indenting each inner line to four spaces past the directive (original inner indentation is lost);
    # the closing fence is dropped. (3) Converts <Tip>...</Tip> sections to a tip admonition with the body
    # indented three spaces. (4) Fixes a misplaced backtick typo in TrainerCallback references, then
    # rewrites bracketed transformers references to Python object roles. Returns nothing.
    # KNOWN ISSUE: inner code lines are stripped of their own leading whitespace, so nested indentation inside fenced code blocks is flattened.
    # If we are documenting something under `sentence_transformers` but the
    # underlying object actually lives in `transformers`, we treat the
    # docstring as inherited from transformers and prepend a note that
    # links back to the original API in the transformers docs.
    origin_module = getattr(obj, "__module__", "")
    if (
        obj_type in {"function", "decorator", "method", "property", "attribute"}
        and name.startswith("sentence_transformers.")
        and origin_module.startswith("transformers")
    ):
        qualname = getattr(obj, "__qualname__", name.split(".")[-1])
        target = f"transformers.{qualname}"
        # Use a generic Python object reference so intersphinx can
        # resolve it to the transformers docs.
        lines.insert(0, "")
        lines.insert(0, f".. note:: This docstring is inherited from :py:obj:`{target}`.")

    # Convert Markdown-style fenced code blocks (```py ... ```)
    # into rST ``.. code-block::`` directives so Sphinx renders them correctly.
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("```"):
            # Opening fenced block
            # Detect language, e.g. ```py or ```python
            fence = stripped
            lang = fence[3:].strip()  # part after ```
            if not lang:
                lang = "python"
            elif lang == "py":
                lang = "python"

            # Keep the original indentation of the fence line
            indent = line[: len(line) - len(stripped)]
            new_lines.append(f"{indent}.. code-block:: {lang}")

            # rST expects a blank indented line after the directive
            new_lines.append(f"{indent}    ")

            # Consume lines until the closing fence ```
            i += 1
            while i < len(lines):
                inner = lines[i]
                inner_stripped = inner.lstrip()

                # Closing fence
                if inner_stripped.startswith("```"):
                    break

                # Indent code at least 4 spaces more than directive
                code_content = inner_stripped
                new_lines.append(f"{indent}    {code_content}")
                i += 1

            # Skip the closing fence line itself (if present)
            while i < len(lines):
                if lines[i].lstrip().startswith("```"):
                    i += 1
                    break
                i += 1

            continue

        # Non-fenced line: keep as-is
        new_lines.append(line)
        i += 1

    # Replace original lines content in-place so autodoc sees updates
    lines[:] = new_lines

    # Replace <Tip> ... </Tip> with a proper rST ``.. tip::`` admonition.
    tip_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("<Tip>"):
            # Keep the original indentation of the <Tip> line
            indent = line[: len(line) - len(stripped)]
            tip_lines.append(f"{indent}.. tip::")

            # Consume lines until closing </Tip>
            i += 1
            while i < len(lines):
                inner = lines[i]
                inner_stripped = inner.lstrip()

                if inner_stripped.startswith("</Tip>"):
                    break

                # Indent tip body relative to the directive
                tip_lines.append(f"{indent}   {inner_stripped}")
                i += 1

            # Skip the closing </Tip> line itself (if present)
            while i < len(lines):
                if lines[i].lstrip().startswith("</Tip>"):
                    i += 1
                    break
                i += 1

            continue

        tip_lines.append(line)
        i += 1

    lines[:] = tip_lines

    # Update the [`~transformers.TrainerCallback`] transformers style references to
    # :py:obj:`transformers.TrainerCallback` so intersphinx can resolve them.
    pattern = re.compile(r"\[`~?transformers\.([A-Za-z0-9_.]+)`\]")
    for idx, line in enumerate(lines):
        # First correct a common mistake in transformers
        line = line.replace("[`~transformers.TrainerCallback]`", "[`~transformers.TrainerCallback`]")
        lines[idx] = pattern.sub(r":py:obj:`transformers.\1`", line)


def inherit_class_docstrings(app, what, name, obj, options, lines):
    # [FUNCTION DOC]
    # autodoc-process-docstring handler. When documenting a class whose docstring lines are empty, fetches
    # the docstring via inspect (which walks the MRO) and, if found, appends it to lines after autodoc's
    # docstring preparation using the configured tab width (default 8). Mutates lines in place.
    """Inherit parent class docstrings for classes that don't define their own.

    ``autodoc_inherit_docstrings`` only applies to methods, not classes.
    This handler fills that gap by walking the MRO when a class has no docstring.
    """
    if what == "class" and not lines:
        doc = inspect.getdoc(obj)
        if doc:
            tab_width = app.config.autodoc_tab_width if hasattr(app.config, "autodoc_tab_width") else 8
            lines.extend(sphinx.ext.autodoc.prepare_docstring(doc, tab_width))


def setup(app: Sphinx):
    # [FUNCTION DOC]
    # Sphinx extension entry point. Registers inherit_class_docstrings on autodoc-process-docstring with
    # priority 100 (so it runs before Napoleon) and preprocess_docstring with the default priority.
    # Register with low priority (before Napoleon at 500) so inherited docstrings
    # get processed by Napoleon (Args -> :param:) and preprocess_docstring too.
    app.connect("autodoc-process-docstring", inherit_class_docstrings, priority=100)
    app.connect("autodoc-process-docstring", preprocess_docstring)
