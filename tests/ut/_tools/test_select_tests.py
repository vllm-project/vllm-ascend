# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import json
import subprocess
import sys
from pathlib import Path

import pytest

SELECTOR = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/select_tests.py"


@pytest.mark.parametrize(
    "suffix,expected_suffix",
    [
        ("", ""),
        ("::test_present", "::test_present"),
        ("::test_present[case::one]", "::test_present[case::one]"),
        ("::TestGroup::test_async", "::TestGroup::test_async"),
        ("::test_removed", ""),
        ("::MissingGroup::test_async", ""),
        ("::TestGroup::test_removed", ""),
    ],
)
def test_recommendations_fall_back_to_file_for_missing_nodes(tmp_path, monkeypatch, suffix, expected_suffix):
    test_file = "tests/e2e/pull_request/two_card/test_example.py"
    path = tmp_path / test_file
    path.parent.mkdir(parents=True)
    path.write_text(
        'raise RuntimeError("test selection must not import this module")\n'
        "def test_present(): pass\n"
        "class TestGroup:\n"
        "    async def test_async(self): pass\n",
        encoding="utf-8",
    )
    targets = tmp_path / "recommended.txt"
    targets.write_text(test_file + suffix + "\n", encoding="utf-8")
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    result = subprocess.run(
        [sys.executable, str(SELECTOR), "--test-list-file", str(targets)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    groups = json.loads(outputs["test_groups"])
    assert [target for group in groups for target in group["tests"].splitlines()] == [test_file + expected_suffix]
    assert ("falling back to file" in result.stderr) == (suffix != expected_suffix)
