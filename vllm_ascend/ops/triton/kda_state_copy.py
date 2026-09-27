# SPDX-License-Identifier: Apache-2.0
"""Explicit-lifecycle Triton implementation of KDA state gather/clear/scatter.

This opt-in module has no automatic production dispatch integration.
This module does not replace or register any existing project operator.

Contract:
    state: NPU FP32/BF16 cache [cache_rows, H, V, K]. Inner payload is dense;
        the first-axis stride may include gaps. The tensor data pointer already
        incorporates its storage offset, so no extra storage_offset is added.
    packed_states: same-device, same-dtype contiguous [selected, H, V, K].
    indices: same-device INT32/INT64 vector [selected], possibly strided.
    has_initial_state: optional flag vector; gather clears false or invalid rows.
    scatter: ignores flag VALUES and skips invalid indices. Valid destination
        indices must be unique; uniqueness is a caller precondition, not checked
        using a device-to-host synchronization. Repeated gather indices are valid.

Only dense payload elements are read/written; cache page gaps stay untouched.
Cache and packed storage must not overlap. Concurrent writers are unsupported.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from threading import RLock
from types import MappingProxyType

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _kda_state_copy_kernel(
    cache_ptr,
    packed_ptr,
    indices_ptr,
    flags_ptr,
    cache_rows,
    cache_stride_elements,
    payload_elements,
    index_stride,
    flag_stride,
    TO_CACHE: tl.constexpr,
    HAS_FLAGS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one payload tile for one selected state.

    Grid axis 0 selects an entry in indices; axis 1 selects a payload tile.
    Promote before address multiplication to avoid 32-bit intermediate overflow.
    Invalid cache rows use a safe row address AND a false memory-access mask.
    """
    selected = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    element = tile * BLOCK_SIZE + lane
    in_payload = element < payload_elements

    cache_row = tl.load(indices_ptr + selected * index_stride).to(tl.int64)
    valid_row = (cache_row >= 0) & (cache_row < cache_rows)
    safe_row = tl.where(valid_row, cache_row, 0).to(tl.int64)

    # These offsets are in ELEMENTS, not bytes. Typed pointers handle item size.
    cache_offset = safe_row * cache_stride_elements + element
    packed_offset = selected * payload_elements + element

    if TO_CACHE:
        # Flags describe INITIAL state only; never suppress a valid final write.
        copy_mask = in_payload & valid_row
        value = tl.load(packed_ptr + packed_offset, mask=copy_mask, other=0)
        tl.store(cache_ptr + cache_offset, value, mask=copy_mask)
    else:
        should_read = valid_row
        if HAS_FLAGS:
            flag = tl.load(flags_ptr + selected * flag_stride)
            should_read = should_read & (flag != 0)
        # Masked reads become zero. Always write the full valid packed payload,
        # so invalid indices / false flags do not leave uninitialized output.
        value = tl.load(
            cache_ptr + cache_offset,
            mask=in_payload & should_read,
            other=0,
        )
        tl.store(packed_ptr + packed_offset, value, mask=in_payload)


def _validate_inputs(state, packed_states, indices):
    """Validate each call and return only this call's reusable tensor metadata.

    No tensor identity or validation result survives the call. All original
    shape, dtype, device, contiguity and nonoverlapping-page checks are retained.
    Singleton inner axes retain the original permissive stride semantics.
    """
    device = state.device
    if device.type != "npu":
        raise RuntimeError("state must be an NPU tensor")
    shape, packed_shape = state.shape, packed_states.shape
    if len(shape) != 4 or shape[0] <= 0 or len(packed_shape) != 4:
        raise RuntimeError("expected nonempty cache [N,H,V,K] and packed [S,H,V,K]")
    dtype = state.dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("state must be FP32 or BF16")
    if packed_states.dtype != dtype or packed_states.device != device:
        raise RuntimeError("packed_states must match cache dtype and device")
    if not packed_states.is_contiguous():
        raise RuntimeError("packed_states must be contiguous")
    index_dtype = indices.dtype
    if indices.ndim != 1 or index_dtype not in (torch.int32, torch.int64):
        raise RuntimeError("indices must be a one-dimensional INT32/INT64 tensor")
    if indices.device != device:
        raise RuntimeError("indices must be on the cache device")
    selected = indices.numel()
    if packed_shape[0] != selected:
        raise RuntimeError("packed row count must match the number of indices")
    if packed_shape[1:] != shape[1:]:
        raise RuntimeError("cache and packed inner shapes must match")
    strides = state.stride()
    payload = 1
    for axis in (3, 2, 1):
        size, stride = shape[axis], strides[axis]
        if size <= 0 or (size > 1 and stride != payload):
            raise RuntimeError("cache must have a dense inner [H,V,K] payload")
        payload *= size
    if strides[0] < payload:
        raise RuntimeError("cache pages must not overlap")
    return payload, device, dtype, index_dtype, selected, shape[0], strides[0], indices.stride(0)


def _copy_kda_states_triton(
    state: torch.Tensor,
    packed_states: torch.Tensor,
    indices: torch.Tensor,
    *,
    to_cache: bool = False,
    has_initial_state: torch.Tensor | None = None,
    block_size: int = 1024,
    _prepare: bool = False,
) -> None:
    """Gather/clear into packed_states, or scatter packed_states into state.

    This is an explicit alternative entry point, not an automatic fallback.
    block_size is a power-of-two tile size in ELEMENTS, defaulting to 1024.
    Callers must prepare the same explicit block size used during serving.
    The wrapper allocates no payload/output tensor. Metadata conversion below
    may allocate, matching the existing Python wrapper's flag normalization.

    Empty selections are validated and return without a kernel launch.
    Device availability, resource limits and compiler errors are not swallowed.
    Kernel launch is asynchronous; the caller controls synchronization/timing.
    """
    # This internal entry is called only while the lifecycle lock is held.
    if _prepare:
        if _SEALED:
            raise RuntimeError("preparation is forbidden after seal")
    elif not _SEALED:
        raise RuntimeError("call prepare_kda_states_triton then seal_kda_states_triton first")
    _check_configuration()
    if _kda_state_copy_kernel.pre_run_hooks:
        raise RuntimeError("JIT pre-run hooks are unsupported by strict dispatch")
    if type(to_cache) is not bool:
        raise TypeError("to_cache must be bool")
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    (payload, device, dtype, index_dtype, selected, cache_rows, cache_stride, index_stride) = _validate_inputs(
        state, packed_states, indices
    )

    flags = has_initial_state
    if flags is not None:
        # Match the existing Python wrapper's device/dtype/shape normalization.
        if flags.device != device or flags.dtype != torch.bool:
            flags = flags.to(device=device, dtype=torch.bool, non_blocking=True)
        if flags.ndim != 1:
            flags = flags.reshape(-1)
        if flags.numel() != selected:
            raise RuntimeError("initial-state flags must match the number of indices")

    if selected == 0:
        return

    has_flags = flags is not None and not to_cache
    # A valid dummy pointer avoids passing None; HAS_FLAGS removes its load.
    flags_arg = flags if has_flags else indices
    flag_stride = flags.stride(0) if has_flags else 0
    grid = (selected, triton.cdiv(payload, block_size))

    # Exact scalar values prevent reusing implicit equal-to-one specializations.
    # Pointer alignment classes cover this installed backend's specialization.
    # Never retain tensors, data pointers, streams or runtime index/flag values.
    tensors = (state, packed_states, indices, flags_arg)
    scalars = (cache_rows, cache_stride, payload, index_stride, flag_stride)
    # Validated packed dtype equals cache dtype; normalized flags are boolean.
    # Retain every pointer alignment class and exact scalar specialization.
    key = (
        device,
        (dtype, dtype, index_dtype, torch.bool if has_flags else index_dtype),
        (state.data_ptr() % 16, packed_states.data_ptr() % 16, indices.data_ptr() % 16, flags_arg.data_ptr() % 16),
        scalars,
        to_cache,
        has_flags,
        block_size,
        grid,
        os.environ.get("TRITON_DEBUG", "0"),
    )
    # Skip redundant context switching only when this thread is already on
    # the tensor device. Preserve the original guard for a mismatching device.
    with _NO_DEVICE_SWITCH if torch.npu.current_device() == device.index else torch.npu.device(device):
        runner = _LAUNCHERS.get(key)
        if runner is None:
            if not _prepare:
                raise RuntimeError("unprepared KDA signature; JIT compilation is forbidden")
            # No eviction: exceeding the budget fails BEFORE entering JIT.
            if len(_LAUNCHERS) >= _MAX_SIGNATURES:
                raise RuntimeError("preparation signature budget exhausted; no eviction")
            compiled = _kda_state_copy_kernel[grid](
                *tensors, *scalars, TO_CACHE=to_cache, HAS_FLAGS=has_flags, BLOCK_SIZE=block_size
            )
            _LAUNCHERS[key] = compiled[(grid[0], grid[1], 1)]
        else:
            runner(*tensors, *scalars)


# Lifecycle is process-local and irreversible. Private state is not a security
# boundary against monkey-patching; restarting requires a new prepare/seal pass.
_LAUNCHERS = {}
_MAX_SIGNATURES = 128
_SEALED = False
_CONFIGURATION = None
_PREPARED_DEVICES = set()
_LOCK = RLock()
_NO_DEVICE_SWITCH = nullcontext()


def _configuration():
    """Pin compile-related environment and runtime versions for this process.

    This is a fail-closed configuration check, not a portable compiler-cache key.
    Loaded backend/binaries must remain fixed; live code replacement is unsupported.
    """
    prefixes = ("TRITON_", "ASCEND_", "CANN_", "TORCH_NPU_", "NPU_", "LLVM_")
    keys = {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH"}
    return (
        torch.__version__,
        triton.__version__,
        tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith(prefixes) or k in keys)),
    )


def _check_configuration():
    """Reject configuration drift before compilation or a steady-state launch."""
    global _CONFIGURATION
    current = _configuration()
    if _CONFIGURATION is None:
        _CONFIGURATION = current
    elif current != _CONFIGURATION:
        raise RuntimeError("KDA compiler/runtime configuration changed; restart and prepare")


def _deny_jit(*args, **kwargs):
    """Hard stop accidental JIT dispatch on this kernel after sealing."""
    raise RuntimeError("sealed KDA kernel: JIT entry is disabled")


def prepare_kda_states_triton(state, packed_states, indices, **kwargs):
    """Compile AND EXECUTE one signature on disposable startup tensors.

    Gather overwrites packed_states; scatter overwrites state. Call outside graph
    capture before serving traffic. Prepare each dtype/device/layout/alignment/
    block/direction/flag-presence/selected-count class that serving will admit.
    Tensor values are not keys. Do not warm up on live state unless writes are safe.
    """
    with _LOCK:
        _copy_kda_states_triton(state, packed_states, indices, _prepare=True, **kwargs)
        _PREPARED_DEVICES.add(state.device)


def seal_kda_states_triton():
    """Synchronize startup work and irreversibly prohibit new signatures/JIT.

    Multiple seals are idempotent. No unseal, automatic warmup, eviction, or JIT
    fallback exists. Installations of pre-run hooks after seal are rejected.
    """
    global _SEALED, _LAUNCHERS
    with _LOCK:
        if _SEALED:
            return
        _check_configuration()
        if not _LAUNCHERS:
            raise RuntimeError("cannot seal an empty launcher registry")
        if _kda_state_copy_kernel.pre_run_hooks:
            raise RuntimeError("remove JIT pre-run hooks before sealing")
        for device in _PREPARED_DEVICES:
            with torch.npu.device(device):
                torch.npu.synchronize()
        _LAUNCHERS = MappingProxyType(dict(_LAUNCHERS))
        _kda_state_copy_kernel.run = _deny_jit
        _SEALED = True


def copy_kda_states_triton(state, packed_states, indices, **kwargs):
    """Run only a sealed prepared signature; never invoke the Triton JIT.

    Metadata/normalization and compiled-kernel launch still run on the host.
    Graph replay uses captured pointers: their lifetime belongs to the caller.
    No promise is made about unrelated Torch/Legacy kernels compiling elsewhere.
    """
    with _LOCK:
        _copy_kda_states_triton(state, packed_states, indices, **kwargs)
