# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU coverage for native-package selection and wheel resource preservation."""

import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[4]


def test_build_entrypoint_does_not_shadow_stdlib(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/deepgemm/build.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--destination" in result.stdout


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loader(tmp_path, monkeypatch):
    module = load_file("test_loader", ROOT / "vllm_ascend/ops/fused_moe/deepgemm_loader.py")
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


@pytest.mark.parametrize("external", [False, True])
def test_selection_and_cache(loader, monkeypatch, external):
    monkeypatch.setattr(loader, "_is_npu_package", lambda spec: external)
    with (
        patch.object(loader.importlib.util, "find_spec", return_value=None),
        patch.object(loader.importlib, "import_module", return_value=npu_module()) as imported,
    ):
        assert loader.load_deepgemm() is loader.load_deepgemm()
        imported.assert_called_once_with("deep_gemm" if external else loader._BUNDLED)
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
        pytest.raises(RuntimeError, match="already loaded"),
    ):
        loader.load_deepgemm()
    imported.assert_not_called()


def test_packaged_headers_and_no_build_environment(tmp_path):
    builder = load_file("test_builder", ROOT / "tools/deepgemm/build.py")
    wheel = tmp_path / "operator.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("deep_gemm/_C.cpython-test.so", b"extension")
        archive.writestr("deep_gemm/include/deep_gemm/mega_moe.hpp", b"header")
        archive.writestr("deep_gemm/envs.py", "persistent_envs={'DG_JIT_CACHE_DIR':'/build-machine'}")
        archive.writestr("deep_gemm-0.1.dist-info/METADATA", "not installed separately")
    destination = tmp_path / "vendored"
    builder.vendor_wheel(wheel, destination)
    assert (destination / "include/deep_gemm/mega_moe.hpp").read_bytes() == b"header"
    assert (destination / "envs.py").read_text() == "persistent_envs = {}\n"
    assert not list(destination.glob("*.dist-info"))


def test_default_cache_preserves_existing_setting(loader, monkeypatch):
    monkeypatch.setenv("DG_JIT_CACHE_DIR", "/existing/cache")
    with (
        patch.object(loader.importlib.util, "find_spec", return_value=None),
        patch.object(loader.importlib, "import_module", return_value=npu_module()),
    ):
        loader.load_deepgemm()
    assert os.environ["DG_JIT_CACHE_DIR"] == "/existing/cache"


def test_device_header_fix_is_explicit_and_version_scoped(tmp_path):
    builder = load_file("test_builder", ROOT / "tools/deepgemm/build.py")
    common = tmp_path / "include/deep_gemm/common.hpp"
    comm = tmp_path / "include/deep_gemm/comm/mega_moe.hpp"
    comm.parent.mkdir(parents=True)
    common.write_text("#include <utility>\n")
    comm.write_text("#include <simt_api/device_functions.h>\n")
    hashes = builder.fix_device_includes(tmp_path)
    assert len(hashes) == 2
    assert common.read_text() == "#include <type_traits>\n"
    assert "#include <utils/debug/asc_printf.h>" in comm.read_text()
    with pytest.raises(RuntimeError, match="no longer applies"):
        builder.fix_device_includes(tmp_path)


def test_unrelated_tilelang_kernel_is_not_imported_at_package_load(tmp_path):
    builder = load_file("test_builder", ROOT / "tools/deepgemm/build.py")
    initializer = tmp_path / "__init__.py"
    initializer.write_text("from .tilelang_ops import tf32_hc_prenorm_gemm\n")
    hashes = builder.defer_tilelang_import(tmp_path)
    # There is deliberately no tilelang_ops module in this test package.
    package = load_file("optional_kernel_probe", initializer)
    assert callable(package.tf32_hc_prenorm_gemm)
    assert "__init__.py" in hashes
    with pytest.raises(ModuleNotFoundError):
        package.tf32_hc_prenorm_gemm()
