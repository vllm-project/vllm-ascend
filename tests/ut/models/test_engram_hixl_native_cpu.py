# SPDX-License-Identifier: Apache-2.0
"""Exercise the native HIXL configuration boundary without an NPU."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_resource_file_is_bounded_without_hardware(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the native HIXL CPU harness")
    assert compiler is not None
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path / "engram_hixl_cpu"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root / 'tests/native/engram_hixl_cpu_stubs/include'}",
            str(root / "tests/native/test_engram_hixl_local_comm_res.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    result = subprocess.run([str(executable)], check=True, text=True, capture_output=True, timeout=30)
    assert "production native.cpp CPU harness passed" in result.stdout
