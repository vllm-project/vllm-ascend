# SPDX-License-Identifier: Apache-2.0
"""CPU-only host-contract tests; stubs do not establish NPU correctness.

Flow: load a fresh module with isolated backend stubs -> prepare -> seal ->
exercise accepted and rejected signatures without a device or compiler.
"""

import importlib.util
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/kda_state_copy.py"


@dataclass(frozen=True)
class Device:
    """Hashable NPU identity used by the dispatch signature."""

    type: str = "npu"
    index: int = 0


class Tensor:
    """Minimal tensor metadata; no payload computation is simulated."""

    def __init__(self, shape, strides, dtype="float32", pointer=4096, contiguous=True):
        self.shape = shape
        self.strides = strides
        self.dtype = dtype
        self.pointer = pointer
        self.contiguous = contiguous
        self.device = Device()

    @property
    def ndim(self):
        """Return rank, as a real Torch tensor does."""
        return len(self.shape)

    def stride(self, axis=None):
        """Expose element strides, never byte strides."""
        return self.strides if axis is None else self.strides[axis]

    def is_contiguous(self):
        """Return the fixture's declared packed layout."""
        return self.contiguous

    def data_ptr(self):
        """Return the fixture's effective, storage-offset-adjusted pointer."""
        return self.pointer

    def numel(self):
        """Return vector size for initial-state flags."""
        assert self.ndim == 1
        return self.shape[0]


class Kernel:
    """Track preparation, JIT entry and compiled-launcher use separately."""

    def __init__(self):
        self.pre_run_hooks = []
        self.compiles = 0
        self.launches = 0
        self.run = self.compile

    def __getitem__(self, grid):
        """Model JIT indexing; the returned callable consults the live run hook."""
        return lambda *args, **kwargs: self.run(*args, **kwargs)

    def compile(self, *args, **kwargs):
        """Record a preparation compile and its first execution."""
        self.compiles += 1
        self.launches += 1
        return self

    # Compiled kernels and JIT kernels have different indexing semantics.
    def compiled_launcher(self, *args, **kwargs):
        """Count a direct launch without visiting the compilation entry."""
        self.launches += 1


class Compiled:
    """Expose the compiled-kernel grid binding used by the implementation."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        """Bind a grid to the direct-launch counter."""
        return self.kernel.compiled_launcher


@pytest.fixture
def module(monkeypatch):
    """Import real host code in isolation; restore all module stubs afterwards."""
    kernel = Kernel()

    def compile_kernel(*args, **kwargs):
        """Wrap preparation output as a compiled kernel, not another JIT object."""
        kernel.compile(*args, **kwargs)
        return Compiled(kernel)

    kernel.run = compile_kernel
    torch = ModuleType("torch")
    for name in ("float32", "bfloat16", "int32", "int64", "bool"):
        setattr(torch, name, name)
    # Populate dynamic module namespaces without pretending ModuleType declares
    # vendor-specific Torch/Triton attributes in its static interface.
    torch.__dict__.update(
        __version__="test",
        npu=SimpleNamespace(current_device=lambda: 0, device=lambda _: nullcontext(), synchronize=lambda: None),
    )
    utils = ModuleType("vllm.triton_utils")
    utils.__dict__.update(
        triton=SimpleNamespace(jit=lambda _: kernel, cdiv=lambda n, d: (n + d - 1) // d, __version__="test"),
        tl=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", utils)
    spec = importlib.util.spec_from_file_location("isolated_kda_state_copy", SOURCE)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inputs(selected=2, payload=4):
    """Return dense payloads in gapped cache pages and contiguous packed rows."""
    return (
        Tensor((8, 1, 1, payload), (payload + 8, payload, payload, 1)),
        Tensor((selected, 1, 1, payload), (payload, payload, payload, 1)),
        Tensor((selected,), (1,), "int32"),
    )


def test_requires_prepare_and_seal(module):
    """Neither an empty registry nor preparation alone enables serving."""
    args = inputs()
    with pytest.raises(RuntimeError, match="prepare"):
        module.copy_kda_states_triton(*args)
    with pytest.raises(RuntimeError, match="empty launcher"):
        module.seal_kda_states_triton()
    module.prepare_kda_states_triton(*args)
    with pytest.raises(RuntimeError, match="seal"):
        module.copy_kda_states_triton(*args)
    module.seal_kda_states_triton()
    module.seal_kda_states_triton()
    module.copy_kda_states_triton(*args)
    assert module._kda_state_copy_kernel.compiles == 1
    assert module._kda_state_copy_kernel.launches == 2
    with pytest.raises(RuntimeError, match="preparation is forbidden"):
        module.prepare_kda_states_triton(*args)
    with pytest.raises(RuntimeError, match="JIT entry is disabled"):
        module._kda_state_copy_kernel.run()


@pytest.mark.parametrize("change", ["payload", "alignment", "index_dtype", "block", "direction", "flags"])
def test_unknown_signature_never_compiles(module, change):
    """Unprepared scalar, constexpr, dtype and pointer classes fail closed."""
    module.prepare_kda_states_triton(*inputs())
    module.seal_kda_states_triton()
    args = inputs(payload=8) if change == "payload" else inputs()
    kwargs: dict[str, object] = {}
    if change == "alignment":
        args[0].pointer += 4
    elif change == "index_dtype":
        args[2].dtype = "int64"
    elif change == "block":
        kwargs["block_size"] = 1024
    elif change == "direction":
        kwargs["to_cache"] = True
    elif change == "flags":
        kwargs["has_initial_state"] = Tensor((2,), (1,), "bool")
    with pytest.raises(RuntimeError, match="unprepared"):
        module.copy_kda_states_triton(*args, **kwargs)
    assert module._kda_state_copy_kernel.compiles == 1
    assert module._kda_state_copy_kernel.launches == 1


def test_different_tensor_identity_reuses_prepared_launcher(module):
    """Aligned pointer changes do not make the registry cache tensor identity."""
    module.prepare_kda_states_triton(*inputs())
    module.seal_kda_states_triton()
    args = inputs()
    for tensor in args:
        tensor.pointer += 32
    module.copy_kda_states_triton(*args)
    assert module._kda_state_copy_kernel.compiles == 1
    assert module._kda_state_copy_kernel.launches == 2


def test_no_eviction_on_budget_exhaustion(module, monkeypatch):
    """Reject excess startup signatures before invoking JIT or dropping entries."""
    monkeypatch.setattr(module, "_MAX_SIGNATURES", 1)
    module.prepare_kda_states_triton(*inputs())
    with pytest.raises(RuntimeError, match="budget exhausted"):
        module.prepare_kda_states_triton(*inputs(payload=8))
    assert len(module._LAUNCHERS) == 1
    assert module._kda_state_copy_kernel.compiles == 1


@pytest.mark.parametrize("block", [0, -1, 3, True, 1.5])
def test_invalid_block_size(module, block):
    """Block size must be an integer power of two, excluding bool."""
    with pytest.raises(ValueError, match="power of two"):
        module.prepare_kda_states_triton(*inputs(), block_size=block)
    assert module._kda_state_copy_kernel.compiles == 0


@pytest.mark.parametrize("invalid", ["dtype", "packed", "indices", "pages", "flags", "direction"])
def test_metadata_errors_do_not_compile(module, invalid):
    """Reject invalid metadata rather than attempting a device launch."""
    args = inputs()
    kwargs: dict[str, object] = {}
    if invalid == "dtype":
        args[0].dtype = "float16"
    elif invalid == "packed":
        args[1].contiguous = False
    elif invalid == "indices":
        args[2].dtype = "float32"
    elif invalid == "pages":
        args[0].strides = (1, 4, 4, 1)
    elif invalid == "flags":
        kwargs["has_initial_state"] = Tensor((3,), (1,), "bool")
    elif invalid == "direction":
        kwargs["to_cache"] = 1
    with pytest.raises((RuntimeError, TypeError)):
        module.prepare_kda_states_triton(*args, **kwargs)
    assert module._kda_state_copy_kernel.compiles == 0


def test_empty_selection_validates_without_launch(module):
    """Empty copies are no-ops only after valid metadata and lifecycle checks."""
    module.prepare_kda_states_triton(*inputs())
    module.seal_kda_states_triton()
    module.copy_kda_states_triton(*inputs(selected=0))
    args = inputs(selected=0)
    args[1].contiguous = False
    with pytest.raises(RuntimeError, match="contiguous"):
        module.copy_kda_states_triton(*args)
    assert module._kda_state_copy_kernel.launches == 1


def test_runtime_drift_and_hooks_rejected(module, monkeypatch):
    """Detect compiler configuration drift and newly installed JIT hooks."""
    module.prepare_kda_states_triton(*inputs())
    module.seal_kda_states_triton()
    with monkeypatch.context() as env:
        env.setenv("TRITON_KDA_TEST_DRIFT", "changed")
        with pytest.raises(RuntimeError, match="configuration changed"):
            module.copy_kda_states_triton(*inputs())
    module._kda_state_copy_kernel.pre_run_hooks.append(object())
    with pytest.raises(RuntimeError, match="pre-run hooks"):
        module.copy_kda_states_triton(*inputs())
    assert module._kda_state_copy_kernel.launches == 1


def test_dynamic_selected_reuses_compiled_kernel(module):
    """Request count changes the grid, not the compiled signature."""
    module.prepare_kda_states_triton(*inputs(selected=2))
    module.seal_kda_states_triton()
    module.copy_kda_states_triton(*inputs(selected=65))
    assert len(module._LAUNCHERS) == 1
    assert module._kda_state_copy_kernel.compiles == 1
    assert module._kda_state_copy_kernel.launches == 2


def test_default_block_matches_explicit_acceptance_configuration(module):
    """The default call and explicit 8192 block share one signature."""
    module.prepare_kda_states_triton(*inputs())
    module.seal_kda_states_triton()
    module.copy_kda_states_triton(*inputs(), block_size=8192)
    assert module._kda_state_copy_kernel.compiles == 1
    assert module._kda_state_copy_kernel.launches == 2
