# SPDX-License-Identifier: Apache-2.0
"""CPU-only host-contract tests for the sole worker-owned lifecycle.

Flow: stub NPU allocation/compilation -> prepare disposable scratch -> seal ->
exercise compiled launches and rejected layouts. No payload math is simulated.
"""

import importlib.util
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Device:
    """Hashable fake NPU identity for cache signatures."""

    type: str = "npu"
    index: int = 0


class Tensor:
    """Minimal aligned tensor metadata for startup and steady-state checks."""

    def __init__(self, shape, strides=None, dtype="float32", pointer=4096, device=None):
        self.shape = tuple(shape)
        self.strides: tuple[int, ...] = tuple(strides) if strides is not None else self._dense(self.shape)
        self.dtype = dtype
        self.pointer = pointer
        self.device = Device() if device is None else device

    @staticmethod
    def _dense(shape):
        """Derive contiguous strides without allocating payload storage."""
        strides: list[int] = []
        running = 1
        for size in reversed(shape):
            strides.insert(0, running)
            running *= size
        return tuple(strides)

    @property
    def ndim(self):
        """Return tensor rank."""
        return len(self.shape)

    def stride(self, axis=None):
        """Return element strides."""
        return self.strides if axis is None else self.strides[axis]

    def is_contiguous(self):
        """Compare layout with dense strides."""
        return self.strides == self._dense(self.shape)

    def data_ptr(self):
        """Return the effective storage-offset-adjusted pointer."""
        return self.pointer

    def numel(self):
        """Count metadata elements, including empty selections."""
        result = 1
        for size in self.shape:
            result *= size
        return result

    def element_size(self):
        """Model the input dtype item size."""
        return 2 if self.dtype == "bfloat16" else 4

    def narrow(self, axis, start, length):
        """Return a scratch view with the requested pointer alignment."""
        assert axis == 0
        return Tensor((length,), dtype=self.dtype, pointer=self.pointer + start * self.element_size())


class Kernel:
    """Track four startup variants separately from compiled launches."""

    def __init__(self):
        self.pre_run_hooks = []
        self.compiles = 0
        self.launches = []

    def __getitem__(self, grid):
        """Enter JIT only during preparation."""

        def compile(*args, **kwargs):
            self.compiles += 1
            return Compiled(self)

        return compile


class Compiled:
    """Bind each request grid directly without entering the JIT object."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        """Return a direct compiled launcher for the dynamic grid."""
        return lambda *args: self.kernel.launches.append(grid)


@pytest.fixture
def modules(monkeypatch):
    """Import the actual kernel and production plan with isolated CPU stubs."""
    kernel = Kernel()
    torch = ModuleType("torch")
    for name in ("float32", "bfloat16", "int32", "int64", "bool"):
        setattr(torch, name, name)

    def allocate(shape, dtype=None, device=None):
        """Create metadata-only contiguous output with an aligned pointer."""
        return Tensor((shape,) if isinstance(shape, int) else shape, dtype=dtype, device=device)

    torch.__dict__.update(
        __version__="test",
        Tensor=Tensor,
        empty=allocate,
        zeros=allocate,
        ones=allocate,
        npu=SimpleNamespace(current_device=lambda: 0, device=lambda _: nullcontext(), synchronize=lambda: None),
        compiler=SimpleNamespace(is_compiling=lambda: False),
    )
    triton = ModuleType("vllm.triton_utils")
    triton.__dict__.update(
        triton=SimpleNamespace(jit=lambda _: kernel, __version__="test"),
        tl=SimpleNamespace(),
    )
    stubs = {
        "torch": torch,
        "vllm": ModuleType("vllm"),
        "vllm.triton_utils": triton,
        "vllm.forward_context": ModuleType("vllm.forward_context"),
        "vllm.utils": ModuleType("vllm.utils"),
        "vllm.utils.torch_utils": ModuleType("vllm.utils.torch_utils"),
        "vllm_ascend": ModuleType("vllm_ascend"),
        "vllm_ascend.ops": ModuleType("vllm_ascend.ops"),
        "vllm_ascend.ops.triton": ModuleType("vllm_ascend.ops.triton"),
        "vllm_ascend.ops.triton.batch_memcpy": ModuleType("vllm_ascend.ops.triton.batch_memcpy"),
        "vllm_ascend.utils": ModuleType("vllm_ascend.utils"),
    }
    stubs["vllm.forward_context"].__dict__["get_forward_context"] = lambda: None
    stubs["vllm.utils.torch_utils"].__dict__["direct_register_custom_op"] = lambda **kwargs: None
    stubs["vllm_ascend.ops.triton.batch_memcpy"].__dict__["batch_memcpy_kernel"] = kernel
    stubs["vllm_ascend.utils"].__dict__["is_950"] = lambda: True
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    source = ROOT / "vllm_ascend/ops/triton/kda_state_copy.py"
    spec = importlib.util.spec_from_file_location("vllm_ascend.ops.triton.kda_state_copy", source)
    assert spec is not None and spec.loader is not None
    lowlevel = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, lowlevel)
    spec.loader.exec_module(lowlevel)
    source = ROOT / "vllm_ascend/ops/kda_state_copy.py"
    spec = importlib.util.spec_from_file_location("isolated_production_kda", source)
    assert spec is not None and spec.loader is not None
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    return production, kernel


def state(payload=4, pointer=4096, dtype="float32"):
    """Return a dense inner payload within gapped cache pages."""
    return Tensor((8, 1, 1, payload), (payload + 8, payload, payload, 1), dtype, pointer)


def prepared(modules, *, payload=4, limit=128):
    """Prepare and seal one real production plan on disposable fake scratch."""
    production, kernel = modules
    plan = production.KDAStateCopyPlan.prepare(state(payload), limit)
    assert kernel.compiles == 4
    plan.seal()
    return plan


def test_unsealed_plan_and_layout_rejected_before_launch(modules):
    """An unsealed plan or changed layout cannot enter a compiled launch."""
    production, kernel = modules
    plan = production.KDAStateCopyPlan.prepare(state(), 128)
    with pytest.raises(RuntimeError, match="sealed"):
        plan._indices(state(), Tensor((2,), dtype="int32"))
    plan.seal()
    with pytest.raises(RuntimeError, match="unprepared"):
        plan._indices(state(payload=8), Tensor((2,), dtype="int32"))
    with pytest.raises(RuntimeError, match="unprepared"):
        plan._indices(state(pointer=4100), Tensor((2,), dtype="int32"))
    assert not kernel.launches


def test_dynamic_count_identity_and_four_variants(modules):
    """Both index dtypes/directions and changing selected counts reuse JIT output."""
    plan = prepared(modules)
    _, kernel = modules
    for dtype in ("int32", "int64"):
        for to_cache in (False, True):
            for count in (0, 1, 65):
                indices = plan._indices(state(), Tensor((count,), dtype=dtype, pointer=4128))
                plan._launch(
                    state(),
                    Tensor((count, 1, 1, 4)),
                    indices,
                    Tensor((count,), dtype="bool"),
                    to_cache=to_cache,
                )
    assert kernel.compiles == 4
    assert len(kernel.launches) == 8
    assert set(kernel.launches) == {(1, 1, 1), (65, 1, 1)}


@pytest.mark.parametrize("invalid", ["dtype", "pages", "inner", "limit", "hooks"])
def test_invalid_preparation_never_compiles(modules, invalid):
    """Reject invalid layouts/limits/hooks before startup compilation."""
    production, kernel = modules
    cache = state()
    maximum = 128
    if invalid == "dtype":
        cache.dtype = "int64"
    elif invalid == "pages":
        cache.strides = (1, 4, 4, 1)
    elif invalid == "inner":
        cache.strides = (12, 4, 4, 2)
    elif invalid == "limit":
        maximum = 0
    else:
        kernel.pre_run_hooks.append(object())
    with pytest.raises((RuntimeError, ValueError)):
        production.KDAStateCopyPlan.prepare(cache, maximum)
    assert kernel.compiles == 0


def test_seal_configuration_drift_and_hooks(modules, monkeypatch):
    """Configuration drift is checked at seal, but never scanned per copy."""
    production, kernel = modules
    plan = production.KDAStateCopyPlan.prepare(state(), 128)
    with monkeypatch.context() as changed:
        changed.setenv("TRITON_KDA_TEST_DRIFT", "changed")
        with pytest.raises(RuntimeError, match="configuration changed"):
            plan.seal()
    kernel.pre_run_hooks.append(object())
    with pytest.raises(RuntimeError, match="pre-run hooks"):
        plan.seal()
    kernel.pre_run_hooks.clear()
    plan.seal()
    with monkeypatch.context() as changed:
        changed.setattr(production, "_configuration", lambda: (_ for _ in ()).throw(AssertionError("hot-path scan")))
        plan._indices(state(), Tensor((2,), dtype="int32"))


def test_selected_limit_and_index_metadata_rejected(modules):
    """Count, dtype, and device metadata fail closed before a launch."""
    plan = prepared(modules, limit=2)
    invalid = (
        Tensor((3,), dtype="int32"),
        Tensor((1,), dtype="float32"),
        Tensor((1,), dtype="int32", device=Device(index=1)),
    )
    for indices in invalid:
        with pytest.raises(RuntimeError):
            plan._indices(state(), indices)


@pytest.mark.parametrize("invalid", ["packed_dtype", "packed_shape", "packed_layout", "indices_rank", "indices_dtype"])
def test_shared_metadata_rejects_invalid_buffers(modules, invalid):
    """The remaining common validator rejects mismatched standalone buffers."""
    production, kernel = modules
    packed = Tensor((2, 1, 1, 4))
    indices = Tensor((2,), dtype="int32")
    if invalid == "packed_dtype":
        packed.dtype = "bfloat16"
    elif invalid == "packed_shape":
        packed.shape = (3, 1, 1, 4)
    elif invalid == "packed_layout":
        packed.strides = (8, 4, 4, 1)
    elif invalid == "indices_rank":
        indices.shape = (1, 2)
    else:
        indices.dtype = "float32"
    with pytest.raises(RuntimeError):
        production._validate_inputs(state(), packed, indices)
    assert kernel.compiles == 0


def test_flags_and_unaligned_normalized_buffer_fail_closed(modules):
    """Invalid flag count and unexpected pointer class cannot launch a copy."""
    plan = prepared(modules)
    _, kernel = modules
    with pytest.raises(RuntimeError, match="flags"):
        plan.gather(state(), Tensor((2,), dtype="int32"), Tensor((3,), dtype="bool"))
    indices = plan._indices(state(), Tensor((2,), dtype="int32"))
    with pytest.raises(RuntimeError, match="unaligned"):
        plan._launch(
            state(),
            Tensor((2, 1, 1, 4), pointer=4100),
            indices,
            Tensor((2,), dtype="bool"),
            to_cache=False,
        )
    assert not kernel.launches
