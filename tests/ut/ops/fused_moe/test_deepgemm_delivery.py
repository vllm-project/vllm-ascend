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
    assert spec is not None and spec.loader is not None
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


def test_selection_and_cache(loader, monkeypatch, tmp_path):
    # An installed external package must never affect the bundled backend.
    (tmp_path / "deep_gemm.py").write_text("raise AssertionError('external import')")
    monkeypatch.syspath_prepend(str(tmp_path))
    with patch.object(loader.importlib, "import_module", return_value=npu_module()) as imported:
        assert loader.load_deepgemm() is loader.load_deepgemm()
        imported.assert_called_once_with(loader._BUNDLED)
    assert os.environ["DG_JIT_CACHE_DIR"].endswith(str(Path("deep_gemm/ascend")))


def test_native_failure_does_not_load_second_extension(loader):
    with (
        patch.object(
            loader.importlib, "import_module", side_effect=RuntimeError("native registration failed")
        ) as imported,
        pytest.raises(RuntimeError, match="native registration failed"),
    ):
        loader.load_deepgemm()
    imported.assert_called_once_with(loader._BUNDLED)


def test_missing_bundled_build_is_actionable(loader):
    with (
        patch.object(loader.importlib, "import_module", side_effect=ModuleNotFoundError(name=loader._BUNDLED)),
        pytest.raises(RuntimeError, match="was not built"),
    ):
        loader.load_deepgemm()


def test_missing_transitive_dependency_preserves_cause(loader):
    with (
        patch.object(loader.importlib, "import_module", side_effect=ModuleNotFoundError(name="missing_dependency")),
        pytest.raises(ModuleNotFoundError) as error,
    ):
        loader.load_deepgemm()
    assert error.value.name == "missing_dependency"


@pytest.mark.parametrize("external", [SimpleNamespace(), npu_module()])
def test_already_loaded_extension_is_not_loaded_twice(loader, monkeypatch, external):
    monkeypatch.setitem(sys.modules, "deep_gemm", external)
    with (
        patch.object(loader.importlib, "import_module") as imported,
        pytest.raises(RuntimeError, match="already loaded"),
    ):
        loader.load_deepgemm()
    imported.assert_not_called()


def test_wheel_payload_is_preserved_byte_for_byte(tmp_path):
    builder = load_file("test_builder", ROOT / "tools/deepgemm/build.py")
    wheel = tmp_path / "operator.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("deep_gemm/_C.cpython-test.so", b"extension")
        archive.writestr("deep_gemm/include/deep_gemm/mega_moe.hpp", b"header")
        archive.writestr("deep_gemm/envs.py", "# Pre-installed environment variables\npersistent_envs = dict()\n")
        archive.writestr("deep_gemm-0.1.dist-info/METADATA", "not installed separately")
    destination = tmp_path / "vendored"
    builder.vendor_wheel(wheel, destination)
    assert (destination / "include/deep_gemm/mega_moe.hpp").read_bytes() == b"header"
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("deep_gemm/"):
                assert (destination / Path(name).relative_to("deep_gemm")).read_bytes() == archive.read(name)
    assert not list(destination.glob("*.dist-info"))


def test_default_cache_preserves_existing_setting(loader, monkeypatch):
    monkeypatch.setenv("DG_JIT_CACHE_DIR", "/existing/cache")
    with (
        patch.object(loader.importlib, "import_module", return_value=npu_module()),
    ):
        loader.load_deepgemm()
    assert os.environ["DG_JIT_CACHE_DIR"] == "/existing/cache"


def test_sdk_toolchain_is_scoped_to_build_and_recordable(tmp_path, monkeypatch):
    builder = load_file("test_builder_env", ROOT / "tools/deepgemm/build.py")
    compiler = tmp_path / "tools/hcc/bin/aarch64-target-linux-gnu-g++"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("CFLAGS", "-g1")
    monkeypatch.setenv("DG_JIT_CACHE_DIR", "/build-cache")
    monkeypatch.setattr(builder.shutil, "which", lambda name: None)
    monkeypatch.setattr(builder.Path, "is_file", lambda path: path == compiler)
    env = builder.native_build_env()
    assert env["CXX"] == str(compiler)
    assert env["CFLAGS"] == "-g1 -idirafter /usr/include"
    assert "-static-libstdc++" in env["LDFLAGS"]
    assert "DG_JIT_CACHE_DIR" not in env
    assert "CXX" not in os.environ
    assert os.environ["DG_JIT_CACHE_DIR"] == "/build-cache"


def test_explicit_native_toolchain_is_preserved(monkeypatch):
    builder = load_file("test_builder_env", ROOT / "tools/deepgemm/build.py")
    monkeypatch.setenv("CXX", "/custom/bin/clang++")
    monkeypatch.setenv("CFLAGS", "-O2")
    monkeypatch.setenv("LDFLAGS", "-Wl,--as-needed")
    env = builder.native_build_env()
    assert env["CXX"] == "/custom/bin/clang++"
    assert env["CFLAGS"] == "-O2"
    assert env["LDFLAGS"] == "-Wl,--as-needed"


@pytest.mark.parametrize("deepgemm", [False, True])
def test_editable_build_preserves_native_packages(tmp_path, deepgemm):
    import __future__

    import ast
    from typing import Any

    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    # Load the production command without executing setup.py's installation.
    tree = ast.parse((ROOT / "setup.py").read_text())
    command = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "cmake_build_ext")
    source = tmp_path / "source"
    (source / "vllm_ascend").mkdir(parents=True)
    namespace: dict[str, Any] = {"build_ext": build_ext, "Path": Path, "ROOT_DIR": str(source)}
    exec(
        compile(
            ast.Module(body=[command], type_ignores=[]), "setup.py", "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    names = ["vllm_ascend.vllm_ascend_C"]
    if deepgemm:
        names.append("vllm_ascend.third_party.deep_gemm._C")
    distribution = Distribution(
        {"ext_modules": [Extension(name, sources=[]) for name in names], "package_dir": {"": str(source)}}
    )
    build = namespace["cmake_build_ext"](distribution)
    build.ensure_finalized()
    build.build_lib = str(tmp_path / "build")
    for name in names:
        extension = Path(build.build_lib) / build.get_ext_filename(name)
        extension.parent.mkdir(parents=True, exist_ok=True)
        extension.write_bytes(b"native-extension")
    relative = Path("vllm_ascend/third_party/deep_gemm")
    resources = {
        "__init__.py": b"",
        "include/deep_gemm/mega_moe.hpp": b"header",
        "ascend_build.json": b"{}",
        "LICENSE": b"license",
    }
    if deepgemm:
        for name, content in resources.items():
            resource = Path(build.build_lib) / relative / name
            resource.parent.mkdir(parents=True, exist_ok=True)
            resource.write_bytes(content)
    assert not (source / relative).exists()
    build.copy_extensions_to_source()
    for name in names:
        assert (source / build.get_ext_filename(name)).read_bytes() == b"native-extension"
    if deepgemm:
        for name, content in resources.items():
            assert (source / relative / name).read_bytes() == content
    else:
        assert not (source / relative).exists()
