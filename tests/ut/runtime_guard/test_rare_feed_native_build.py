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

# mypy: ignore-errors
"""Guarantee rare_feed_native is a first-class setuptools Extension (wheel path).

``COMPILE_CUSTOM_KERNELS=0`` (CPU CI / docs) and ``=1`` (Ascend image / wheel)
must both package the module via setuptools — not only a cmake sidecar that
can be omitted from the wheel because ``*.so`` is gitignored.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RARE_PKG = "vllm_ascend.observability.runtime_guard.detector.rare"
RARE_EXT_NAME = f"{RARE_PKG}.rare_feed_native"
SRC = REPO_ROOT / "vllm_ascend/observability/runtime_guard/detector/rare/rare_feed_native.c"
SETUP_PY = REPO_ROOT / "setup.py"
CMAKE = REPO_ROOT / "CMakeLists.txt"


def test_setup_always_registers_plain_extension():
    text = SETUP_PY.read_text(encoding="utf-8")
    assert "_RARE_FEED_NATIVE_EXT = Extension(" in text
    assert 'name="vllm_ascend.observability.runtime_guard.detector.rare.rare_feed_native"' in text
    # Must start from the plain Extension, then optionally append CMake.
    assert "ext_modules: list[Extension] = [_RARE_FEED_NATIVE_EXT]" in text
    assert 'ext_modules.append(CMakeExtension(name="vllm_ascend.vllm_ascend_C"))' in text
    # Regression: do not replace the list with cmake-only (drops rare from wheel).
    assert 'ext_modules = [CMakeExtension(name="vllm_ascend.vllm_ascend_C")]' not in text


def test_cmake_does_not_own_rare_feed_native():
    text = CMAKE.read_text(encoding="utf-8")
    assert "add_library(rare_feed_native" not in text


def test_source_present():
    assert SRC.is_file(), f"missing {SRC}"


def _built_so_candidates() -> list[Path]:
    rare_dir = REPO_ROOT / "vllm_ascend/observability/runtime_guard/detector/rare"
    return sorted(rare_dir.glob("rare_feed_native*.so")) + sorted(rare_dir.glob("rare_feed_native*.pyd"))


def test_module_loads_after_inplace_build():
    """COMPILE_CUSTOM_KERNELS=0 inplace build must produce a loadable .so."""
    env = os.environ.copy()
    env["COMPILE_CUSTOM_KERNELS"] = "0"
    env.setdefault("SOC_VERSION", "ascend910b1")
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(REPO_ROOT),
        env=env,
    )
    sos = _built_so_candidates()
    assert sos, "build_ext --inplace did not produce rare_feed_native*.so"
    # Load by path under the real init name so CPU CI does not need torch_npu.
    import importlib.util

    spec = importlib.util.spec_from_file_location("rare_feed_native", sos[0])
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "make_ctx") or hasattr(mod, "feed")


def test_rare_native_loader_finds_extension():
    import importlib.machinery
    import types

    if "torch_npu" not in sys.modules:
        stub = types.ModuleType("torch_npu")
        stub.__spec__ = importlib.machinery.ModuleSpec(name="torch_npu", loader=None)
        sys.modules["torch_npu"] = stub

    from vllm_ascend.observability.runtime_guard.detector.rare import rare_native

    rare_native._MOD = None
    rare_native._MOD_TRIED = False
    mod = rare_native._module()
    if mod is None:
        pytest.skip("rare_feed_native not built in this environment yet")
    assert mod is not None
    assert "rare_feed_native" in (getattr(mod, "__file__", "") or "")


def test_bdist_wheel_includes_rare_feed_native(tmp_path: Path):
    """Wheel built with COMPILE_CUSTOM_KERNELS=0 must contain the Extension .so."""
    env = os.environ.copy()
    env["COMPILE_CUSTOM_KERNELS"] = "0"
    env.setdefault("SOC_VERSION", "ascend910b1")
    dist = tmp_path / "dist"
    dist.mkdir()
    subprocess.check_call(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(dist),
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )
    wheels = list(dist.glob("*.whl"))
    assert wheels, "no wheel produced"
    import zipfile

    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    hits = [n for n in names if "rare_feed_native" in n and n.endswith((".so", ".pyd"))]
    assert hits, f"rare_feed_native missing from wheel; sample={names[:20]}"
