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

"""Kernel-line rewriting used by Triton coverage collection."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend" / "triton_line_coverage.py"
_SPEC = importlib.util.spec_from_file_location("triton_line_coverage_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
line_coverage = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = line_coverage
_SPEC.loader.exec_module(line_coverage)

_KERNEL = '''\
@triton.jit
def kernel(x, IS_VARLEN: tl.constexpr):
    """doc"""
    if IS_VARLEN:
        y = helper(x)
    else:
        y = x + 2
    for i in range(y):
        z = i
    return y
'''
_FIRST_LINE = 20


def _resolve_helper(func: ast.expr) -> bool:
    return isinstance(func, ast.Name) and func.id == "helper"


def _rewritten() -> tuple[ast.FunctionDef, line_coverage.LineTable]:
    table = line_coverage.LineTable()
    source = line_coverage.transform_source(
        _KERNEL,
        _FIRST_LINE,
        _resolve_helper,
        table,
        "pkg/vllm_ascend/ops/triton/kernel.py",
    )
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    return function, table


def test_probes_follow_the_branch_that_executes() -> None:
    function, table = _rewritten()
    recorded = {lineno for _path, lineno, _absolute in table.entries}

    assert recorded == {23, 24, 26, 27, 28, 29}
    assert "def kernel" in ast.unparse(function)
    assert function.args.args[-1].arg == line_coverage.COV_PTR

    docstring, if_store, if_node, for_store, for_node, return_store, return_node = function.body
    assert isinstance(docstring, ast.Expr)
    assert isinstance(if_node, ast.If)
    assert _store_line(if_store, table) == 23
    assert _store_line(if_node.body[0], table) == 24
    assert _store_line(if_node.orelse[0], table) == 26
    assert isinstance(for_node, ast.For)
    assert _store_line(for_store, table) == 27
    assert _store_line(for_node.body[0], table) == 28
    assert isinstance(return_node, ast.Return)
    assert _store_line(return_store, table) == 29

    helper_call = if_node.body[1].value
    assert isinstance(helper_call, ast.Call)
    assert isinstance(helper_call.args[-1], ast.Name)
    assert helper_call.args[-1].id == line_coverage.COV_PTR

    range_call = for_node.iter
    assert isinstance(range_call, ast.Call)
    assert all(not (isinstance(arg, ast.Name) and arg.id == line_coverage.COV_PTR) for arg in range_call.args)


def test_local_name_is_not_treated_as_a_kernel_call() -> None:
    source = "def kernel(helper):\n    return helper(1)\n"
    table = line_coverage.LineTable()
    rewritten = line_coverage.transform_source(source, 1, _resolve_helper, table, "vllm_ascend/ops/triton/kernel.py")
    call = ast.parse(rewritten).body[0].body[-1].value
    assert isinstance(call, ast.Call)
    assert len(call.args) == 1


def test_lines_from_hits_keep_only_executed_probes() -> None:
    entries = [
        ("ops/triton/kernel.py", 23, "/src/vllm_ascend/ops/triton/kernel.py"),
        ("ops/triton/kernel.py", 24, "/src/vllm_ascend/ops/triton/kernel.py"),
        ("ops/triton/utils.py", 74, "/src/vllm_ascend/ops/triton/utils.py"),
    ]
    covered = line_coverage.lines_from_hits([1, 0, 1], entries)
    assert covered == {"ops/triton/kernel.py": {23}, "ops/triton/utils.py": {74}}


def test_repo_relative_path_strips_package_prefix() -> None:
    assert line_coverage.repo_relative_path(r"C:\work\vllm_ascend\ops\triton\kernel.py") == "ops/triton/kernel.py"
    path = "/src/vllm-ascend/vllm_ascend/ops/triton/kernel.py"
    assert line_coverage.repo_relative_path(path) == "ops/triton/kernel.py"


def test_sidecar_records_executed_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERAGE_FILE", str(tmp_path / "coverage"))
    line_coverage._ACCUMULATED.clear()
    entries = [("ops/triton/kernel.py", 24, "/src/vllm_ascend/ops/triton/kernel.py")]
    line_coverage._record([1], entries)
    payload = json.loads((tmp_path / "triton_lines.json").read_text(encoding="utf-8"))
    assert payload == {"ops/triton/kernel.py": [24]}


def test_install_stays_off_without_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(line_coverage.ENV_VAR, raising=False)
    assert line_coverage.install() is False


def _store_line(statement: ast.stmt, table: line_coverage.LineTable) -> int:
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    offset = call.args[0]
    assert isinstance(offset, ast.BinOp)
    assert isinstance(offset.right, ast.Constant)
    probe = int(offset.right.value)
    _relative, lineno, _absolute = table.entries[probe]
    return lineno
