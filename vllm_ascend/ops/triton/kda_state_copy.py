# SPDX-License-Identifier: Apache-2.0
"""KDA state-copy kernel and metadata validation for worker-owned plans.

Production selects its plan by device/cache layout. The sole prepare/seal
lifecycle lives in ops/kda_state_copy.py; independent tests use that same plan.
This module has no launcher registry.

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
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:

    class _CompiledKernel(Protocol):
        """Structural launch interface for vendor Triton kernels without stubs."""

        def __getitem__(self, grid: tuple[int, int, int]) -> Callable[..., object]: ...


DEFAULT_KDA_BLOCK_SIZE = 8192


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


def _validate_cache_layout(state: torch.Tensor) -> tuple[int, int, int]:
    """Validate a real fused cache once at startup and extract launch scalars.

    ``prepare`` builds its own disposable packed row and index vectors from
    this cache's metadata. Their shape, dtype, device and contiguous layout
    are construction guarantees, not external inputs to validate here. Actual
    request indices and packed final states are checked by the sealed plan.
    """
    if state.device.type != "npu":
        raise RuntimeError("state must be an NPU tensor")
    shape = state.shape
    if len(shape) != 4 or shape[0] <= 0:
        raise RuntimeError("expected nonempty cache [N,H,V,K]")
    if state.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("state must be FP32 or BF16")
    strides = state.stride()
    payload = 1
    for axis in (3, 2, 1):
        size, stride = shape[axis], strides[axis]
        if size <= 0 or (size > 1 and stride != payload):
            raise RuntimeError("cache must have a dense inner [H,V,K] payload")
        payload *= size
    if strides[0] < payload:
        raise RuntimeError("cache pages must not overlap")
    return payload, shape[0], strides[0]
