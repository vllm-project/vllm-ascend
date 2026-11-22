#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Tests for ``check_forbidden_imports.py``.

Colocated with the script, like ``.github/workflows/scripts/test_codeowner_gate.py``:
the cpu-ut CI partition only collects ``tests/ut``, so run this file directly:

    pytest tools/test_check_forbidden_imports.py
"""

import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).with_name("check_forbidden_imports.py")


def _run_tool(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_unreadable_source_file_is_skipped_without_crashing(tmp_path: Path) -> None:
    """A non-UTF-8 source file must be skipped, not abort the whole run.

    Python accepts any source encoding declared in a ``coding:`` comment, so a
    latin-1 file is a legitimate input for a hook that walks Python files.
    ``check_file`` returns an ``int`` (violation count) for every input, and
    ``main`` combines those results with ``|=``. Returning a list for the
    undecodable branch raised ``TypeError`` instead of skipping the file.
    """
    latin1_file = tmp_path / "latin1_module.py"
    latin1_file.write_bytes(b"# -*- coding: latin-1 -*-\n# Author: Ren\xe9 M\xfcller\nimport os\n")

    result = _run_tool(latin1_file)

    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


def test_unreadable_file_does_not_mask_a_real_violation(tmp_path: Path) -> None:
    """Skipping an undecodable file must not hide violations in other files."""
    latin1_file = tmp_path / "latin1_module.py"
    latin1_file.write_bytes(b"# -*- coding: latin-1 -*-\n# Ren\xe9\nimport os\n")
    offending = tmp_path / "offending.py"
    offending.write_text("import triton\n", encoding="utf-8")

    result = _run_tool(latin1_file, offending)

    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 1
    assert f"{offending}" in result.stdout
    assert "forbidden import: triton" in result.stdout
