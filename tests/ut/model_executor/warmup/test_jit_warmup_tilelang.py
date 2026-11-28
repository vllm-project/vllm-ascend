# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.model_executor.warmup.helpers import make_mock_worker, make_registry
from vllm_ascend.ops.tilelang import ascend_jit_warmup as jit
from vllm_ascend.ops.triton import rms_norm

kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


class CopyKernel(jit.AscendTileLangJitKernel[int]):
    def get_prim_func(self, compile_key):
        return ("copy", compile_key)

    def get_warmup_keys(self, sizes):
        return sizes

    def get_compile_options(self, compile_key):
        return {"out_idx": [1]}


@pytest.fixture
def npu_runtime():
    compiler = MagicMock()
    with patch.object(torch, "npu", create=True) as npu, patch.dict(sys.modules, {"tilelang": compiler}):
        npu.current_device.return_value = 0
        npu.is_current_stream_capturing.return_value = False
        yield npu, compiler


def test_warmup_and_launch_share_compilation(npu_runtime):
    _, compiler = npu_runtime
    kernel = CopyKernel()
    registry = make_registry()
    with registry.activate():
        kernel.register_warmup([128, 256, 128])
    registry.warmup()
    assert compiler.compile.call_count == 2
    compiler.compile.assert_any_call(("copy", 128), target="ascend", execution_backend="tvm_ffi", out_idx=[1])
    assert kernel.get_kernel(128) is compiler.compile.return_value
    assert compiler.compile.call_count == 2


def test_capture_allows_cached_buckets_and_rejects_a_miss(npu_runtime):
    npu, compiler = npu_runtime
    kernel = CopyKernel()
    kernel.compile(128)
    npu.is_current_stream_capturing.return_value = True
    assert kernel.get_kernel(128) is compiler.compile.return_value
    with pytest.raises(RuntimeError, match="not warmed"):
        kernel.get_kernel(129)
    compiler.compile.assert_called_once()


def test_compiler_failure_is_propagated_and_not_cached(npu_runtime):
    _, compiler = npu_runtime
    kernel = CopyKernel()
    compiler.compile.side_effect = RuntimeError("compiler diagnostic")
    with pytest.raises(RuntimeError, match="compiler diagnostic"):
        kernel.compile(128)
    assert not kernel._compiled_cache


def test_missing_tilelang_is_reported_at_compilation(npu_runtime):
    kernel = CopyKernel()
    with patch.dict(sys.modules, {"tilelang": None}), pytest.raises(ModuleNotFoundError, match="tilelang"):
        kernel.compile(128)
    assert not kernel._compiled_cache


def test_executor_cannot_be_reused_on_another_device(npu_runtime):
    npu, _ = npu_runtime
    kernel = CopyKernel()
    kernel.compile(128)
    npu.current_device.return_value = 1
    with pytest.raises(RuntimeError, match="each NPU worker device"):
        kernel.get_kernel(128)


@pytest.mark.parametrize("has_triton", [False, True])
def test_worker_retains_tilelang_registrations(has_triton, npu_runtime):
    kernel = CopyKernel()
    worker = make_mock_worker()
    registry = worker.model_runner.jit_warmup_registry
    with registry.activate():
        kernel.register_warmup([128])
    registry._registrations[object()] = [((), {})]
    # Triton late registration is covered by its existing orchestration tests.
    with (
        patch.object(kw, "HAS_TRITON", has_triton),
        patch.object(kw, "HAS_TILELANG", True),
        patch.object(kw, "_register_dcp_warmup"),
        patch.object(kw, "_register_dspark_swa_indices_warmup", return_value=0),
    ):
        kw.kernel_warmup(worker)
    assert list(registry._registrations) == [kernel]
    assert 128 in kernel._compiled_cache


def test_disabled_warmup_does_not_load_tilelang(npu_runtime):
    _, compiler = npu_runtime
    worker = make_mock_worker(enable_jit_warmup=False)
    registry = worker.model_runner.jit_warmup_registry
    with registry.activate():
        CopyKernel().register_warmup([128])
    with patch.dict(sys.modules, {"tilelang": None}):
        kw.kernel_warmup(worker)
    compiler.compile.assert_not_called()


@pytest.mark.parametrize(
    "has_triton,has_tilelang,tilelang_first",
    [(False, False, False), (False, True, False), (True, False, False), (True, True, False), (True, True, True)],
)
def test_worker_filters_unavailable_backends(tilelang_first, has_triton, has_tilelang, npu_runtime):
    worker = make_mock_worker()
    registry = worker.model_runner.jit_warmup_registry
    tilelang_kernel = CopyKernel()
    triton_kernel = rms_norm._TRITON_Q_RMS_KERNEL
    registrations = [(triton_kernel, (worker.vllm_config,)), (tilelang_kernel, ([128],))]
    if tilelang_first:
        registrations.reverse()
    with registry.activate():
        for kernel, args in registrations:
            kernel.register_warmup(*args)
    registry._registrations[object()] = [((), {})]
    registry.warmup = MagicMock()
    with (
        patch.object(kw, "HAS_TRITON", has_triton),
        patch.object(kw, "HAS_TILELANG", has_tilelang),
        patch.object(kw, "_register_dcp_warmup") as register_dcp,
        patch.object(kw, "_register_dspark_swa_indices_warmup", return_value=0) as register_dspark,
    ):
        kw.kernel_warmup(worker)
    available_kernels = []
    if has_triton:
        available_kernels.append(triton_kernel)
        register_dcp.assert_called_once_with(worker)
        register_dspark.assert_called_once_with(worker)
    else:
        register_dcp.assert_not_called()
        register_dspark.assert_not_called()
    if has_tilelang:
        available_kernels.append(tilelang_kernel)
    assert list(registry._registrations) == [kernel for kernel, _ in registrations if kernel in available_kernels]
    assert registry.warmup.call_count == int(bool(available_kernels))


@pytest.mark.parametrize("available", [False, True])
def test_package_detection_does_not_import_backend(available, tmp_path):
    if available:
        (tmp_path / "tilelang.py").write_text("raise AssertionError('TileLang must not be imported')\n")
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"tilelang", "torch", "torch_npu"}
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(
        "tilelang_probe", Path(__file__).resolve().parents[4] / "vllm_ascend/ops/tilelang/__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        patch.object(sys, "path", [str(tmp_path)]),
        patch.dict(sys.modules),
        patch("builtins.__import__", side_effect=checked_import),
    ):
        sys.modules.pop("tilelang", None)
        spec.loader.exec_module(module)
        assert "tilelang" not in sys.modules
    assert module.HAS_TILELANG is available
