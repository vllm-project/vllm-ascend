# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from tests.coverage_support.triton_coverage import clear_recorded_state, install

_KERNEL_SOURCE = """\
def unused(value):
    return value + 5


def plain(value):
    return value + 3


def helper(value):
    return value + 1


def kernel(value):
    if value:
        out = helper(value)
    else:
        out = plain(0)
    return out
"""


class JITFunction:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        del args, kwargs
        if getattr(self.fn, "_boom", False):
            raise RuntimeError("launch failed")
        return "launched"


JITFunction.__module__ = "triton.runtime.jit"


@pytest.fixture(autouse=True)
def _isolated_triton_coverage(monkeypatch):
    pytest.importorskip("coverage")
    clear_recorded_state()
    jit_module = types.ModuleType("triton.runtime.jit")
    jit_module.JITFunction = JITFunction
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", jit_module)
    install()
    yield
    clear_recorded_state()


def _line_containing(source: str, text: str) -> int:
    for line_number, line in enumerate(source.splitlines(), start=1):
        if text in line:
            return line_number
    raise AssertionError(f"{text!r} is not in the kernel source")


def _load_module(directory: Path, source: str):
    directory.mkdir(parents=True)
    path = directory / "kernel_mod.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"kernel_mod_{directory.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.helper = JITFunction(module.helper)
    module.kernel = JITFunction(module.kernel)
    return module, path


def _measured_lines(coverage_module, cov, path: Path) -> set[int]:
    del coverage_module
    cov.stop()
    cov.save()
    data = cov.get_data()
    target = os.path.normcase(str(path))
    for filename in data.measured_files():
        if os.path.normcase(filename) == target:
            return set(data.lines(filename) or [])
    return set()


def test_launched_kernel_lines_are_measured_without_executing_the_body(tmp_path):
    coverage = pytest.importorskip("coverage")
    module, path = _load_module(tmp_path / "pkg" / "vllm_ascend", _KERNEL_SOURCE)
    cov = coverage.Coverage(branch=True, data_file=str(tmp_path / "coverage"), config_file=False)
    cov.start()
    assert module.kernel.run(1) == "launched"

    measured = _measured_lines(coverage, cov, path)
    assert _line_containing(_KERNEL_SOURCE, "return value + 1") in measured
    assert _line_containing(_KERNEL_SOURCE, "out = helper(value)") in measured
    assert _line_containing(_KERNEL_SOURCE, "out = plain(0)") in measured
    assert _line_containing(_KERNEL_SOURCE, "return out") in measured
    assert _line_containing(_KERNEL_SOURCE, "return value + 5") not in measured
    assert _line_containing(_KERNEL_SOURCE, "return value + 3") not in measured


def test_kernels_outside_vllm_ascend_are_not_measured(tmp_path):
    coverage = pytest.importorskip("coverage")
    module, path = _load_module(tmp_path / "other_pkg", _KERNEL_SOURCE)
    cov = coverage.Coverage(branch=True, data_file=str(tmp_path / "coverage"), config_file=False)
    cov.start()
    module.kernel.run(1)

    assert _line_containing(_KERNEL_SOURCE, "return out") not in _measured_lines(coverage, cov, path)


def test_failed_launch_still_records_the_kernel(tmp_path):
    coverage = pytest.importorskip("coverage")
    module, path = _load_module(tmp_path / "pkg" / "vllm_ascend", _KERNEL_SOURCE)
    module.kernel.fn._boom = True
    cov = coverage.Coverage(branch=True, data_file=str(tmp_path / "coverage"), config_file=False)
    cov.start()
    with pytest.raises(RuntimeError, match="launch failed"):
        module.kernel.run(1)

    measured = _measured_lines(coverage, cov, path)
    assert _line_containing(_KERNEL_SOURCE, "return out") in measured
