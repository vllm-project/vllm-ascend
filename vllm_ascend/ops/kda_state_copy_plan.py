# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned prepared plans for PR17301-aligned Kimi prefill state copies.

Flow: bound cache metadata -> register compile keys -> seal and attach plans ->
JitWarmupRegistry compiles -> direct compiled launches. Plans retain metadata,
not tensors or stream handles; preparation never accesses cache contents.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel
from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.ops.triton.kda_state_copy_kernel import (
    DEFAULT_KDA_BLOCK_SIZE,
    _configuration,
    _kda_state_copy_kernel,
    _validate_cache_layout,
)
from vllm_ascend.utils import is_950

if TYPE_CHECKING:
    from vllm_ascend.ops.triton.kda_state_copy_kernel import _CompiledKernel

_FALLBACK_COPY_BLOCK_SIZE = 8192


@dataclass(frozen=True)
class StateCopyCompileKey:
    device: torch.device
    arguments: tuple[tuple[str, Any], ...]


class StateCopyJitKernel(AscendTritonJitKernel[StateCopyCompileKey]):
    """Worker-owned compiled launchers scheduled by JitWarmupRegistry.

    Metadata-only warmup compiles without allocating or launching device data.
    The cached executor includes its ABI constants, resolved once at compile
    time. With JIT warmup disabled, the framework cache compiles on first use.
    """

    def __init__(self, kernel: Any, constexpr_names: tuple[str, ...]) -> None:
        super().__init__()
        self.kernel = kernel
        self._constexpr_names = constexpr_names

    @property
    def supports_parallel_compile(self) -> bool:
        # Signature inspection needs a completed CompiledKernel rather than
        # the futures returned inside Triton's asynchronous compile pool.
        return False

    def get_warmup_keys(self, key: StateCopyCompileKey) -> list[StateCopyCompileKey]:
        return [key]

    def warmup_inputs(self, compile_key: StateCopyCompileKey) -> dict[str, Any]:
        return dict(compile_key.arguments)

    def compile(self, compile_key: StateCopyCompileKey) -> None:
        if compile_key in self._compiled_cache:
            return
        inputs = self.warmup_inputs(compile_key)
        with torch.npu.device(compile_key.device):
            compiled = self.kernel.warmup(grid=(1,), **inputs)
        constants = (
            tuple(inputs[name] for name in self._constexpr_names)
            if "constexpr" in compiled.src.signature.values()
            else ()
        )
        self._compiled_cache[compile_key] = compiled, constants

    def executor(self, compile_key: StateCopyCompileKey) -> tuple[_CompiledKernel, tuple[Any, ...]]:
        return self._get_or_compile(compile_key)


def supports_kda_state_copy(state: torch.Tensor) -> bool:
    """Match PR #17301's A5 FP32/BF16 dense-payload capability predicate.

    The native-extension presence check is replaced by the bundled Triton
    implementation. Device discovery happens at startup, not per copy.
    """
    if state.device.type != "npu" or not is_950() or state.dtype not in (torch.float32, torch.bfloat16):
        return False
    if state.ndim != 4 or state.shape[0] == 0:
        return False
    payload = 1
    for size, stride in zip(reversed(state.shape[1:]), reversed(state.stride()[1:])):
        if size <= 0 or (size > 1 and stride != payload):
            return False
        payload *= size
    return state.stride(0) >= payload


def cache_signature(state: torch.Tensor) -> tuple:
    """Identify an immutable cache layout, not its allocation or contents.

    Storage offsets are already reflected in data_ptr. Retain the alignment
    class required by the compiler, but allow a same-layout cache to be rebound
    after a worker memory-pool wakeup.
    """
    return state.device, state.dtype, tuple(state.shape), tuple(state.stride()), state.data_ptr() % 16


def _canonical(tensor: torch.Tensor, dtype: torch.dtype, device: torch.device, *, vector: bool = False) -> torch.Tensor:
    """Normalize metadata/packed buffers to one known specialization.

    contiguous() alone is insufficient: contiguous slices may be misaligned,
    and a singleton vector can be contiguous with a non-unit stride. Copy only
    when conversion, layout or alignment requires it. Never canonicalize cache
    tensors: their page gaps and storage offsets are part of the contract.
    """
    if tensor.device != device or tensor.dtype != dtype:
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
    if not tensor.is_contiguous() or tensor.data_ptr() % 16 or (vector and tensor.stride(0) != 1):
        result = torch.empty(tensor.shape, dtype=dtype, device=device)
        result.copy_(tensor)
        tensor = result
    return tensor


class KDAStateCopyPlan:
    """Four compile keys per bound cache layout, registered at startup.

    Variants retain aligned contiguous INT32 or INT64 indices; gather uses aligned BOOL
    flags. Thus batch sizes, sliced metadata and INT32/INT64 source indices do
    not multiply the registry. Payload/cache strides stay fixed for a worker;
    selected counts from zero through scheduler.max_num_seqs are supported.
    The plan is request-stateless and has no eviction. The registry precompiles
    its keys before model runs; disabling JIT warmup permits lazy compilation.
    """

    # Preparation initializes layout/launch metadata; publication binds the name.
    _sealed: bool
    _signature: tuple
    _max_selected: int
    _configuration: tuple[str, str, tuple[tuple[str, str], ...]]
    _payload: int
    _scalars: tuple[int, int, int, int]
    _tiles: int
    _compile_keys: Mapping[tuple[torch.dtype, bool], StateCopyCompileKey]
    _kernel: StateCopyJitKernel
    _layer_name: str

    @classmethod
    def prepare(cls, state: torch.Tensor, max_selected: int) -> KDAStateCopyPlan:
        """Bind cache metadata and register four compile keys without JIT work."""
        if type(max_selected) is not int or max_selected <= 0:
            raise ValueError("max_selected must be a positive scheduler limit")
        payload, rows, page_stride = _validate_cache_layout(state)
        if _kda_state_copy_kernel.pre_run_hooks:
            raise RuntimeError("JIT pre-run hooks are unsupported by strict KDA")
        plan = cls()
        plan._sealed = False
        plan._signature = cache_signature(state)
        plan._max_selected = max_selected
        plan._configuration = _configuration()
        plan._payload = payload
        plan._scalars = (rows, page_stride, payload, 1)
        plan._tiles = (payload + DEFAULT_KDA_BLOCK_SIZE - 1) // DEFAULT_KDA_BLOCK_SIZE
        plan._kernel = StateCopyJitKernel(_kda_state_copy_kernel, ("TO_CACHE", "HAS_FLAGS", "BLOCK_SIZE"))
        compile_keys = {}
        for index_dtype in (torch.int32, torch.int64):
            for to_cache in (False, True):
                inputs = dict(
                    cache_ptr=TritonWarmupTensor(state.dtype, aligned=state.data_ptr() % 16 == 0),
                    packed_ptr=TritonWarmupTensor(state.dtype),
                    indices_ptr=TritonWarmupTensor(index_dtype),
                    flags_ptr=TritonWarmupTensor(index_dtype if to_cache else torch.bool),
                    cache_rows=rows,
                    cache_stride_elements=page_stride,
                    payload_elements=payload,
                    index_stride=1,
                    flag_stride=0 if to_cache else 1,
                    TO_CACHE=to_cache,
                    HAS_FLAGS=not to_cache,
                    BLOCK_SIZE=DEFAULT_KDA_BLOCK_SIZE,
                )
                key = StateCopyCompileKey(state.device, tuple(inputs.items()))
                compile_keys[index_dtype, to_cache] = key
                plan._kernel.register_warmup(key)
        plan._compile_keys = MappingProxyType(compile_keys)
        return plan

    def seal(self) -> None:
        """Seal the immutable compile-key table after configuration checks.

        Serving resolves cached compiled executors through the kernel owner.
        Independent worker lifecycles can coexist in one process.
        """
        if self._configuration != _configuration():
            raise RuntimeError("KDA compiler/runtime configuration changed during preparation")
        if _kda_state_copy_kernel.pre_run_hooks:
            raise RuntimeError("JIT pre-run hooks are unsupported by strict KDA")
        self._sealed = True

    def _indices(self, state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Fail closed before launch, then normalize supported index metadata."""
        if not self._sealed:
            raise RuntimeError("strict KDA plan must be sealed before serving")
        if cache_signature(state) != self._signature:
            raise RuntimeError("unprepared KDA cache layout; reinitialize the worker")
        # Compiler environment is a startup contract, checked by seal(). A
        # stored compiled kernel cannot recompile when environment values change.
        # Runtime environment mutation is unsupported; restart/reprepare instead.
        if _kda_state_copy_kernel.pre_run_hooks:
            raise RuntimeError("KDA JIT hooks changed after seal")
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64) or indices.device != state.device:
            raise RuntimeError("KDA indices must be a same-device INT32/INT64 vector")
        if indices.numel() > self._max_selected:
            raise RuntimeError("KDA selected count exceeds scheduler.max_num_seqs")
        return _canonical(indices, indices.dtype, state.device, vector=True)

    def _launch(self, state, packed, indices, flags, *, to_cache: bool) -> None:
        """Bind only runtime grid/stream information to a precompiled kernel."""
        if not indices.numel():
            return
        tensors = (state, packed, indices, indices if to_cache else flags)
        # Enforce precisely the pointer class used in preparation; do not ask
        # the JIT to specialize an unexpected allocator or normalized buffer.
        if any(t.data_ptr() % 16 for t in tensors[1:]):
            raise RuntimeError("unaligned normalized KDA buffer")
        with nullcontext() if torch.npu.current_device() == state.device.index else torch.npu.device(state.device):
            compiled, constants = self._kernel.executor(self._compile_keys[indices.dtype, to_cache])
            compiled[(indices.numel(), self._tiles, 1)](
                *tensors,
                *self._scalars,
                0 if to_cache else 1,
                *constants,
            )

    def gather(self, state: torch.Tensor, indices: torch.Tensor, flags: torch.Tensor | None) -> torch.Tensor:
        """Gather/clear initial states, including invalid-row zero semantics."""
        if torch.compiler.is_compiling():
            return torch.ops.vllm.kda_state_gather(state, indices, flags, self._layer_name)
        indices = self._indices(state, indices)
        if flags is None:
            flags = torch.ones(indices.numel(), dtype=torch.bool, device=state.device)
        else:
            if flags.ndim != 1:
                flags = flags.reshape(-1)
            if flags.numel() != indices.numel():
                raise RuntimeError("KDA flags must match the selected count")
            flags = _canonical(flags, torch.bool, state.device, vector=True)
        packed = torch.empty((indices.numel(), *state.shape[1:]), dtype=state.dtype, device=state.device)
        self._launch(state, packed, indices, flags, to_cache=False)
        return packed

    def scatter(self, state: torch.Tensor, packed: torch.Tensor, indices: torch.Tensor) -> None:
        """Scatter final states, casting to cache dtype as the old path did.

        Valid destination indices must be unique. Initial-state flags never
        suppress final writes; invalid destinations are masked in the kernel.
        """
        if torch.compiler.is_compiling():
            torch.ops.vllm.kda_state_scatter(state, packed, indices, self._layer_name)
            return
        indices = self._indices(state, indices)
        if tuple(packed.shape) != (indices.numel(), *state.shape[1:]) or packed.device != state.device:
            raise RuntimeError("KDA final states must match the selected count, cache payload and device")
        packed = _canonical(packed, state.dtype, state.device)
        self._launch(state, packed, indices, None, to_cache=True)


class StridedKDAFallbackPlan:
    """PR #17301 byte-copy fallback registered with the JIT warmup framework.

    Used for unsupported contiguous and strided NPU caches. Keep the original
    byte-pointer arithmetic and invalid-row masking, but never enter the
    batch_memcpy JIT after registry warmup. Clearing uses Torch to avoid a second Triton
    warmup contract on unsupported devices. This is not the optimized A5 path.
    """

    # The compiled byte-copy kernel is shared, but each layer owns its binding.
    _sealed: bool
    _signature: tuple
    _max_selected: int
    _state_bytes: int
    _configuration: tuple[str, str, tuple[tuple[str, str], ...]]
    _kernel: StateCopyJitKernel
    _compile_key: StateCopyCompileKey
    _layer_name: str

    @classmethod
    def prepare(cls, state: torch.Tensor, max_selected: int) -> StridedKDAFallbackPlan:
        """Validate pages and register one byte-copy key without live pointers."""
        if type(max_selected) is not int or max_selected <= 0:
            raise ValueError("max_selected must be a positive scheduler limit")
        if state.device.type != "npu" or state.ndim != 4 or state.shape[0] <= 0:
            raise ValueError("strided KDA fallback requires nonempty NPU [N,H,V,K]")
        payload = 1
        for size, stride in zip(reversed(state.shape[1:]), reversed(state.stride()[1:])):
            if size <= 0 or (size > 1 and stride != payload):
                raise ValueError("strided KDA fallback requires dense inner payload")
            payload *= size
        if state.stride(0) < payload:
            raise ValueError("strided KDA fallback pages must not overlap")
        plan = cls()
        plan._signature = cache_signature(state)
        plan._max_selected = max_selected
        plan._state_bytes = payload * state.element_size()
        plan._sealed = False
        plan._configuration = _configuration()
        if batch_memcpy_kernel.pre_run_hooks:
            raise RuntimeError("JIT pre-run hooks are unsupported by KDA fallback")
        plan._kernel = StateCopyJitKernel(batch_memcpy_kernel, ("BLOCK_SIZE",))
        inputs = dict(
            src_ptrs=TritonWarmupTensor(torch.int64),
            dst_ptrs=TritonWarmupTensor(torch.int64),
            sizes=TritonWarmupTensor(torch.int64),
            BLOCK_SIZE=_FALLBACK_COPY_BLOCK_SIZE,
        )
        plan._compile_key = StateCopyCompileKey(state.device, tuple(inputs.items()))
        plan._kernel.register_warmup(plan._compile_key)
        return plan

    def seal(self):
        """Reject startup configuration drift before publishing the fallback."""
        if self._configuration != _configuration() or batch_memcpy_kernel.pre_run_hooks:
            raise RuntimeError("KDA fallback configuration/hooks changed during preparation")
        self._sealed = True

    def _copy(self, state, packed, indices, *, to_cache):
        """Build byte-pointer vectors as the PR does and use a compiled launch."""
        if not self._sealed or cache_signature(state) != self._signature:
            raise RuntimeError("unprepared KDA fallback cache layout")
        if batch_memcpy_kernel.pre_run_hooks:
            raise RuntimeError("KDA fallback JIT hooks changed after seal")
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64) or indices.device != state.device:
            raise ValueError("KDA fallback requires same-device INT32/INT64 indices")
        count = indices.numel()
        if count > self._max_selected:
            raise ValueError("KDA selected count exceeds scheduler limit")
        if packed.shape != (count, *state.shape[1:]) or packed.device != state.device:
            raise ValueError("KDA fallback packed shape/device mismatch")
        if not count:
            return
        indices = indices.to(torch.int64)
        valid = (indices >= 0) & (indices < state.shape[0])
        cache_ptrs = state.data_ptr() + indices.clamp(0, state.shape[0] - 1) * (state.stride(0) * state.element_size())
        packed_ptrs = (
            packed.data_ptr() + torch.arange(count, device=state.device, dtype=torch.int64) * self._state_bytes
        )
        sizes = valid.to(torch.int64) * self._state_bytes
        if to_cache:
            src, dst = packed_ptrs, cache_ptrs
        else:
            packed.zero_()
            src, dst = cache_ptrs, packed_ptrs
        with torch.npu.device(state.device):
            compiled, constants = self._kernel.executor(self._compile_key)
            compiled[(count, 1, 1)](src, dst, sizes, *constants)

    def gather(self, state, indices, flags):
        """Gather valid rows and clear false flags through the cached executor."""
        if torch.compiler.is_compiling():
            return torch.ops.vllm.kda_state_gather(state, indices, flags, self._layer_name)
        if flags is not None:
            flags = flags.to(device=state.device, dtype=torch.bool).reshape(-1)
            if flags.numel() != indices.numel():
                raise ValueError("KDA fallback flags must match selected count")
        packed = torch.empty((indices.numel(), *state.shape[1:]), dtype=state.dtype, device=state.device)
        self._copy(state, packed, indices, to_cache=False)
        if flags is not None:
            packed.masked_fill_(~flags[:, None, None, None], 0)
        return packed

    def scatter(self, state, packed, indices):
        """Cast/contiguize final states and preserve untouched cache pages."""
        if torch.compiler.is_compiling():
            torch.ops.vllm.kda_state_scatter(state, packed, indices, self._layer_name)
            return
        packed = packed.to(dtype=state.dtype).contiguous()
        self._copy(state, packed, indices, to_cache=True)


def initialize_kda_state_copy(static_forward_context: dict, max_selected: int) -> None:
    """Register metadata and atomically attach plans after worker cache binding.

    Kimi layers are identified by an internal class marker, not a user setting.
    Non-KDA layers and empty pipeline stages require no state-copy plans.
    The finite support set comes from actual local cache layouts, not guessed
    request shapes or an arbitrary LRU cap. Exactly four variants per unique
    layout cover every admitted selected count and both source index dtypes.
    """
    plans: dict[tuple, KDAStateCopyPlan | StridedKDAFallbackPlan] = {}
    bindings = []
    for layer_name, layer in static_forward_context.items():
        if not getattr(layer, "_requires_kda_state_copy", False):
            continue
        layer._ascend_kda_state_copy = None
        layer._kda_state_copy_ready = False
        bindings.append((layer_name, layer))
    selected_plans: list[KDAStateCopyPlan | StridedKDAFallbackPlan | None] = []
    plan_type: type[KDAStateCopyPlan] | type[StridedKDAFallbackPlan]
    for _, layer in bindings:
        kv_cache = getattr(layer, "kv_cache", None)
        if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
            raise RuntimeError("Kimi KDA cache must be bound before strict state-copy preparation")
        state = kv_cache[1]
        if not isinstance(state, torch.Tensor):
            raise RuntimeError("Kimi KDA cache state must be a torch.Tensor")
        # A5 with a supported layout uses the fused plan; A3 and all other
        # unsupported layouts use the masked, precompiled byte-copy plan.
        if supports_kda_state_copy(state):
            plan_type = KDAStateCopyPlan
        else:
            # Use the masked byte-copy plan for unsupported layouts so invalid
            # gather indices read zero and invalid scatter destinations are skipped.
            plan_type = StridedKDAFallbackPlan
        key = (plan_type, cache_signature(state))
        if key not in plans:
            plans[key] = plan_type.prepare(state, max_selected)
        selected_plans.append(plans[key])
    for prepared_plan in plans.values():
        prepared_plan.seal()
    # Publish only after every layer has prepared and sealed successfully.
    for (layer_name, layer), plan in zip(bindings, selected_plans):
        if plan is not None:
            # Resolve compiled execution through the existing worker context,
            # not a process-global plan registry. Keep each binding independent
            # while sharing the sealed launch table and immutable metadata.
            plan = copy(plan)
            plan._layer_name = layer_name
        layer._ascend_kda_state_copy = plan
        layer._kda_state_copy_ready = True


def _context_plan(layer_name: str):
    """Resolve the sealed worker-owned plan when an opaque op executes.

    Fake implementations never call this helper. No pointers or plans are
    embedded in an exported graph; the active worker owns the layer binding.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    plan = getattr(layer, "_ascend_kda_state_copy", None)
    if not getattr(layer, "_kda_state_copy_ready", False) or plan is None:
        raise RuntimeError("compiled KDA state copy requires a prepared worker layer")
    return plan


def _kda_state_gather(
    state: torch.Tensor, indices: torch.Tensor, flags: torch.Tensor | None, layer_name: str
) -> torch.Tensor:
    """Run the same eager gather behind the Dynamo/FakeTensor boundary."""
    return _context_plan(layer_name).gather(state, indices, flags)


def _fake_state_selection(state: torch.Tensor, indices: torch.Tensor) -> None:
    """Validate pointer-free metadata, preserving symbolic selected lengths.

    Actual layout/alignment and scheduler-limit checks remain in the sealed
    plan at execution. This boundary is shared by fused and byte-copy plans,
    so it must not restrict payload dtype to the fused FP32/BF16 subset.
    """
    torch._check(state.ndim == 4)
    torch._check(state.shape[0] > 0)
    torch._check(indices.ndim == 1)
    torch._check(indices.dtype in (torch.int32, torch.int64))
    torch._check(indices.device == state.device)
    payload = 1
    for size, stride in zip(reversed(state.shape[1:]), reversed(state.stride()[1:])):
        torch._check(size > 0)
        torch._check((size == 1) | (stride == payload))
        payload *= size
    torch._check(state.stride(0) >= payload)


def _kda_state_gather_fake(
    state: torch.Tensor, indices: torch.Tensor, flags: torch.Tensor | None, layer_name: str
) -> torch.Tensor:
    """Infer fresh packed output without accessing a real pointer or plan."""
    _fake_state_selection(state, indices)
    if flags is not None:
        torch._check(flags.numel() == indices.numel())
    return state.new_empty((indices.numel(), *state.shape[1:]))


def _kda_state_scatter(state: torch.Tensor, packed: torch.Tensor, indices: torch.Tensor, layer_name: str) -> None:
    """Expose cache mutation explicitly while keeping launches precompiled."""
    _context_plan(layer_name).scatter(state, packed, indices)


def _kda_state_scatter_fake(state: torch.Tensor, packed: torch.Tensor, indices: torch.Tensor, layer_name: str) -> None:
    """Validate scatter metadata; fake execution does not modify storage."""
    _fake_state_selection(state, indices)
    torch._check(packed.ndim == 4)
    torch._check(packed.shape[0] == indices.numel())
    for axis in (1, 2, 3):
        torch._check(packed.shape[axis] == state.shape[axis])
    torch._check(packed.device == state.device)


# Eager calls bypass these dispatcher entries. Dynamo sees only these opaque
# ops, with allocation/mutation schemas and Fake implementations; actual NPU
# execution still resolves the current context and uses the sealed plan.
direct_register_custom_op(
    op_name="kda_state_gather",
    op_func=_kda_state_gather,
    mutates_args=[],
    fake_impl=_kda_state_gather_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="kda_state_scatter",
    op_func=_kda_state_scatter,
    mutates_args=["state"],
    fake_impl=_kda_state_scatter_fake,
    dispatch_key="PrivateUse1",
)
