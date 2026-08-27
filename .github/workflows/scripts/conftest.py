#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Pytest configuration for ``scripts/`` unit tests.

The ``select_tests`` module stores ``runner_mapping`` in a module-level
global that is normally populated by ``main()`` from a two-document YAML
config. Tests that exercise the routing internals directly need this global
to be set. The :func:`_load_runner_mapping` autouse fixture loads the real
``test_config.yaml`` once before each test, so internal-function tests work
in isolation.

End-to-end tests that call ``main()`` with their own config still need to
include ``runner_mapping`` in that config (see ``_write_two_doc_config``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import regex as re
import yaml

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

sys.modules.setdefault("regex", re)
select_tests = importlib.import_module("select_tests")


@pytest.fixture(autouse=True)
def _load_runner_mapping():
    """Pre-populate ``select_tests._RUNNER_MAPPING`` from the real config.

    Tests that call internal routing functions (``_route_ut_dir``,
    ``_scan_ut_test_dir``, etc.) depend on this global. Loading it from the
    real ``test_config.yaml`` mirrors the production behavior of ``main()``.
    """
    config_path = _SCRIPT_DIR / "test_config.yaml"
    if config_path.exists():
        docs = list(yaml.safe_load_all(config_path.read_text()))
        meta = docs[1] if len(docs) >= 2 and docs[1] else {}
        select_tests._load_runner_mapping(meta)
    yield
