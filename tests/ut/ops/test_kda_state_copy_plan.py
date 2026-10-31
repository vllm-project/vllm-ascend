# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from vllm_ascend.ops import kda_state_copy_plan as production

kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


@pytest.mark.parametrize("retain_constexpr", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("to_cache", [False, True])
@pytest.mark.parametrize("enable_jit_warmup", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_prepared_launch_matches_compiled_signature(
    monkeypatch, retain_constexpr, index_dtype, to_cache, state_dtype, enable_jit_warmup
):
    launches = []
    compilations = []

    class CompiledKernel:
        def __init__(self, constants):
            self.constants = constants
            signature = dict.fromkeys(range(9), "i64")
            if retain_constexpr:
                signature.update(dict.fromkeys(range(9, 12), "constexpr"))
            self.src = SimpleNamespace(signature=signature)

        def __getitem__(self, grid):
            def launch(*args):
                assert len(args) == (12 if retain_constexpr else 9)
                if retain_constexpr:
                    assert args[9:] == tuple(self.constants[k] for k in ("TO_CACHE", "HAS_FLAGS", "BLOCK_SIZE"))
                launches.append((grid, args))

            return launch

    class Compiler:
        pre_run_hooks = ()

        def warmup(self, *, grid, **inputs):
            assert grid == (1,)
            compilations.append(inputs)
            return CompiledKernel(inputs)

    monkeypatch.setattr(production, "_kda_state_copy_kernel", Compiler())
    monkeypatch.setattr(production, "_validate_cache_layout", lambda state: (6, 4, 6))
    monkeypatch.setattr(production, "_configuration", lambda: ())
    monkeypatch.setattr(
        production.torch,
        "npu",
        SimpleNamespace(device=lambda device: nullcontext(), synchronize=lambda: None, current_device=lambda: None),
        raising=False,
    )
    state = torch.zeros((4, 1, 2, 3), dtype=state_dtype)
    packed = torch.empty((2, 1, 2, 3), dtype=state_dtype)
    indices = torch.zeros(2, dtype=index_dtype)
    flags = torch.ones(2, dtype=torch.bool)
    registry = JitWarmupRegistry(None)
    with registry.activate():
        plan = production.KDAStateCopyPlan.prepare(state, 2)
    plan.seal()
    assert not compilations
    assert len(registry) == 4
    monkeypatch.setattr(kw, "HAS_TRITON", True)
    monkeypatch.setattr(kw, "_register_dspark_swa_indices_warmup", lambda worker: 0)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup)),
        model_runner=SimpleNamespace(jit_warmup_registry=registry),
    )
    kw.kernel_warmup(worker)
    assert len(compilations) == (4 if enable_jit_warmup else 0)
    for inputs in compilations:
        assert isinstance(inputs["cache_ptr"], production.TritonWarmupTensor)
        assert inputs["cache_ptr"].dtype == state_dtype
        assert inputs["flags_ptr"].dtype == (inputs["indices_ptr"].dtype if inputs["TO_CACHE"] else torch.bool)

    # No serving-time JIT or signature inspection after registry warmup.
    if enable_jit_warmup:
        monkeypatch.setattr(plan._kernel, "compile", lambda key: pytest.fail("unexpected compilation"))
    plan._launch(state, packed, indices, flags, to_cache=to_cache)

    assert len(compilations) == (4 if enable_jit_warmup else 1)
    assert len(launches) == 1
    grid, args = launches[0]
    assert grid == (2, 1, 1)
    assert args[0] is state and args[1] is packed and args[2] is indices
    assert args[3] is (indices if to_cache else flags)
    assert args[4:9] == (4, 6, 6, 1, 0 if to_cache else 1)


def test_cache_initialization_only_registers_shared_layouts(monkeypatch):
    compiler = SimpleNamespace(
        pre_run_hooks=(),
        warmup=lambda **kwargs: pytest.fail("cache initialization must only register metadata"),
    )
    monkeypatch.setattr(production, "_kda_state_copy_kernel", compiler)
    monkeypatch.setattr(production, "supports_kda_state_copy", lambda state: True)
    monkeypatch.setattr(production, "_validate_cache_layout", lambda state: (6, 4, 6))
    monkeypatch.setattr(production, "_configuration", lambda: ())
    state = torch.zeros((4, 1, 2, 3))
    layers = {
        name: SimpleNamespace(_requires_kda_state_copy=True, kv_cache=(None, state)) for name in ("kda.0", "kda.1")
    }
    registry = JitWarmupRegistry(None)
    with registry.activate(), monkeypatch.context() as allocations:
        for factory in ("empty", "zeros", "ones", "tensor"):
            allocations.setattr(torch, factory, lambda *a, **kw: pytest.fail("unexpected tensor allocation"))
        production.initialize_kda_state_copy(layers, 2)
    plans = [layer._ascend_kda_state_copy for layer in layers.values()]
    assert len(registry) == 4
    assert plans[0]._kernel is plans[1]._kernel
    assert not plans[0]._kernel._compiled_cache
    assert all(layer._kda_state_copy_ready for layer in layers.values())
    assert [plan._layer_name for plan in plans] == ["kda.0", "kda.1"]


@pytest.mark.parametrize("retain_constexpr", [False, True])
@pytest.mark.parametrize("prewarm", [False, True])
def test_fallback_registry_and_lazy_compilation(monkeypatch, retain_constexpr, prewarm):
    compilations = []
    compiled = SimpleNamespace(src=SimpleNamespace(signature={0: "constexpr"} if retain_constexpr else {}))

    def warmup(*, grid, **inputs):
        compilations.append(inputs)
        assert grid == (1,)
        assert all(
            isinstance(inputs[name], production.TritonWarmupTensor) for name in ("src_ptrs", "dst_ptrs", "sizes")
        )
        return compiled

    monkeypatch.setattr(production, "batch_memcpy_kernel", SimpleNamespace(pre_run_hooks=(), warmup=warmup))
    monkeypatch.setattr(production, "_configuration", lambda: ())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False)
    state = SimpleNamespace(
        device=SimpleNamespace(type="npu"),
        ndim=4,
        shape=(4, 1, 2, 3),
        stride=lambda dim=None: (6, 6, 3, 1) if dim is None else (6, 6, 3, 1)[dim],
        dtype=torch.float16,
        element_size=lambda: 2,
        data_ptr=lambda: 0,
    )
    # A hashable device identity, like the real torch.device used by workers.
    state.device = type("NPUDevice", (), {"type": "npu"})()
    registry = JitWarmupRegistry(None)
    with registry.activate():
        plan = production.StridedKDAFallbackPlan.prepare(state, 2)
    plan.seal()
    assert not compilations
    assert len(registry) == 1
    if prewarm:
        registry.warmup()
        assert len(compilations) == 1
    executor, constants = plan._kernel.executor(plan._compile_key)
    assert executor is compiled
    assert constants == ((8192,) if retain_constexpr else ())
    assert plan._kernel.executor(plan._compile_key) == (executor, constants)
    assert len(compilations) == 1


def test_registry_compile_failure_does_not_cache_executor(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False)
    owner = production.StateCopyJitKernel(SimpleNamespace(warmup=fail), ("BLOCK_SIZE",))
    key = production.StateCopyCompileKey(torch.device("cpu"), (("BLOCK_SIZE", 8192),))
    registry = JitWarmupRegistry(None)
    with registry.activate():
        owner.register_warmup(key)
    with pytest.raises(RuntimeError, match="compiler failed"):
        registry.warmup()
    assert not owner._compiled_cache
    assert not owner.supports_parallel_compile
