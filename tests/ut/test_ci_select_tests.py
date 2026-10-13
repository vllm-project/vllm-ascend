# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import runpy
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def selector(monkeypatch):
    script_dir = Path(__file__).resolve().parents[2] / ".github/workflows/scripts"
    monkeypatch.syspath_prepend(str(script_dir))
    namespace = runpy.run_path(str(script_dir / "select_tests.py"))
    namespace["_load_runner_mapping"](yaml.safe_load((script_dir / "test_config.yaml").read_text(encoding="utf-8")))
    return namespace


@pytest.fixture
def test_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("tests/e2e/pull_request/two_card/test_model.py")
    path.parent.mkdir(parents=True)
    return path


@pytest.mark.parametrize("test_name", ["test_fia_acl_graph", "test_encoder_acl_graph_mrv2"])
def test_renamed_recommendation_collects_on_both_branches(selector, test_file, test_name, tmp_path):
    test_file.write_text(f"def {test_name}():\n    pass\n", encoding="utf-8")
    target = f"{test_file}::test_encoder_acl_graph_mrv2"
    groups: dict[str, list[str]] = defaultdict(list)
    selector["_route_explicit_test_target"](target, groups)
    expected = target if test_name == "test_encoder_acl_graph_mrv2" else str(test_file)
    assert groups == {"a3-2": [expected]}

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--confcutdir", str(tmp_path), expected],
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"::{test_name}" in result.stdout


@pytest.mark.parametrize(
    "node",
    ["test_function", "test_async_function", "TestModel", "TestModel::test_method", "test_parametrized[first]"],
)
def test_valid_node_targets_stay_precise(selector, test_file, node):
    test_file.write_text(
        "import pytest\n"
        "def test_function(): pass\n"
        "async def test_async_function(): pass\n"
        "class TestModel:\n"
        "    def test_method(self): pass\n"
        "@pytest.mark.parametrize('value', ['first'])\n"
        "def test_parametrized(value): pass\n",
        encoding="utf-8",
    )
    target = f"{test_file}::{node}"
    groups: dict[str, list[str]] = defaultdict(list)
    selector["_route_explicit_test_target"](target, groups)
    assert groups == {"a3-2": [target]}


def test_missing_node_warns_and_keeps_file_coverage(selector, test_file, capsys):
    test_file.write_text("def test_existing(): pass\n", encoding="utf-8")
    groups: dict[str, list[str]] = defaultdict(list)
    selector["_route_explicit_test_target"](f"{test_file}::TestRemoved::test_missing", groups)
    assert groups == {"a3-2": [str(test_file)]}
    assert "Test node does not exist" in capsys.readouterr().err


def test_missing_file_is_still_skipped(selector, capsys):
    groups: dict[str, list[str]] = defaultdict(list)
    selector["_route_explicit_test_target"]("tests/e2e/pull_request/two_card/test_missing.py::test_missing", groups)
    assert not groups
    assert "Path does not exist" in capsys.readouterr().err
