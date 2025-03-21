#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/docs/source/conf.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'vllm-ascend'
copyright = '2025, vllm-ascend team'
author = 'the vllm-ascend team'

# The full version, including alpha/beta/rc tags
release = ''

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Copy from https://github.com/vllm-project/vllm/blob/main/docs/source/conf.py
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxarg.ext",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_substitution_extensions",
]

myst_enable_extensions = ["colon_fence", "substitution"]

# Change this when cut down release
myst_substitutions = {
    # the branch of vllm, used in vllm clone
    # - main branch: 'main'
    # - vX.Y.Z branch: 'vX.Y.Z'
    'vllm_version': 'main',
    # the branch of vllm-ascend, used in vllm-ascend clone and image tag
    # - main branch: 'main'
    # - vX.Y.Z branch: latest vllm-ascend release tag
    'vllm_ascend_version': 'main',
    # the newest release version of vllm-ascend and matched vLLM, used in pip install.
    # This value should be updated when cut down release.
    'pip_vllm_ascend_version': "0.7.3rc1",
    'pip_vllm_version': "0.7.3",
    # CANN image tag
    'cann_image_tag': "8.0.0-910b-ubuntu22.04-py3.10",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '.venv',
    'README.md',
    'user_guide/release.template.md',
    # TODO(yikun): Remove this after zh supported
    '**/*.zh.md'
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = project
html_theme = 'sphinx_book_theme'
html_logo = 'logos/vllm-ascend-logo-text-light.png'
html_theme_options = {
    'path_to_docs': 'docs/source',
    'repository_url': 'https://github.com/vllm-project/vllm-ascend',
    'use_repository_button': True,
    'use_edit_page_button': True,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


def setup(app):
    pass
