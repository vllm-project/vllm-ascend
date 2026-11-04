# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU coverage for lazy loading of the optional Ascend wheel."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[4]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loader(tmp_path, monkeypatch):
    module = load_file("test_loader", ROOT / "vllm_ascend/deepgemm.py")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(envs=SimpleNamespace(VLLM_CACHE_ROOT=str(tmp_path))))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
    monkeypatch.delenv("DG_JIT_CACHE_DIR", raising=False)
    return module


def npu_module():
    return SimpleNamespace(
        get_npu_arch=lambda: 950,
        fp8_fp4_mega_moe=lambda: None,
        transform_weights_for_mega_moe=lambda: None,
        SymmBuffer=lambda base=None: None,
        __version__="test",
        __file__="test-npu.so",
    )


@pytest.mark.parametrize(
    "source,is_npu",
    [
        ("from ._C import get_npu_arch", True),
        ("from ._C import get_num_sms", False),
        ("# get_npu_arch\nraise RuntimeError('CUDA import must not run')", False),
    ],
)
def test_external_identification(loader, tmp_path, source, is_npu):
    initializer = tmp_path / "__init__.py"
    initializer.write_text(source)
    assert loader._is_npu_package(SimpleNamespace(origin=str(initializer))) is is_npu


def test_selection_and_cache(loader, monkeypatch):
    monkeypatch.setattr(loader, "_is_npu_package", lambda spec: True)
    with (
        patch.object(loader.importlib.util, "find_spec", return_value=None),
        patch.object(loader.importlib, "import_module", return_value=npu_module()) as imported,
    ):
        assert loader.load_deepgemm() is loader.load_deepgemm()
        imported.assert_called_once_with("deep_gemm")
    assert os.environ["DG_JIT_CACHE_DIR"].endswith(str(Path("deep_gemm/ascend")))


def test_native_failure_does_not_load_second_extension(loader, monkeypatch):
    monkeypatch.setattr(loader, "_is_npu_package", lambda spec: True)
    with (
        patch.object(loader.importlib.util, "find_spec", return_value=None),
        patch.object(
            loader.importlib, "import_module", side_effect=RuntimeError("native registration failed")
        ) as imported,
        pytest.raises(RuntimeError, match="Cannot initialize") as error,
    ):
        loader.load_deepgemm()
    imported.assert_called_once_with("deep_gemm")
    assert "native registration failed" in str(error.value.__cause__)


def test_api_contract_rejects_old_buffer(loader):
    module = npu_module()
    module.SymmBuffer = lambda: None
    with pytest.raises(RuntimeError, match="base"):
        loader._validate(module)


def test_already_loaded_cuda_is_not_mixed_with_npu(loader, monkeypatch):
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace())
    with (
        patch.object(loader.importlib, "import_module") as imported,
        pytest.raises(RuntimeError, match="matching deepgemm_ascend wheel"),
    ):
        loader.load_deepgemm()
    imported.assert_not_called()


def test_default_cache_preserves_existing_setting(loader, monkeypatch):
    monkeypatch.setattr(loader, "_is_npu_package", lambda spec: True)
    monkeypatch.setenv("DG_JIT_CACHE_DIR", "/existing/cache")
    with (
        patch.object(loader.importlib.util, "find_spec", return_value=None),
        patch.object(loader.importlib, "import_module", return_value=npu_module()),
    ):
        loader.load_deepgemm()
    assert os.environ["DG_JIT_CACHE_DIR"] == "/existing/cache"


@pytest.mark.parametrize("source", [None, "from ._C import get_num_sms"])
def test_missing_or_cuda_package_is_not_imported(loader, tmp_path, source):
    spec = None
    if source is not None:
        initializer = tmp_path / "__init__.py"
        initializer.write_text(source)
        spec = SimpleNamespace(origin=str(initializer))
    with (
        patch.object(loader.importlib.util, "find_spec", return_value=spec),
        patch.object(loader.importlib, "import_module") as imported,
    ):
        assert not loader.is_deepgemm_available()
        with pytest.raises(RuntimeError, match="matching deepgemm_ascend wheel"):
            loader.load_deepgemm()
    imported.assert_not_called()


def test_availability_probe_does_not_import_native_package(tmp_path):
    package = tmp_path / "deep_gemm"
    package.mkdir()
    (package / "__init__.py").write_text("from ._C import get_npu_arch\nraise AssertionError('must remain lazy')\n")
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(tmp_path)!r})
from vllm_ascend.deepgemm import is_deepgemm_available
assert is_deepgemm_available()
assert 'deep_gemm' not in sys.modules
assert 'torch_npu' not in sys.modules
assert 'vllm_ascend.ops' not in sys.modules
assert 'vllm.config' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
