# SPDX-License-Identifier: Apache-2.0
"""KDA state-copy kernel and metadata validation for worker-owned plans.

Production selects its plan by device/cache layout. Bound layout metadata is
late-registered before kernel_warmup; warmup and serving share one launcher.

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

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

DEFAULT_KDA_BLOCK_SIZE = 8192


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


class KDAStateCopyKernel(AscendTritonJitKernel["KDAStateCopyKernel.CompileKey"]):
    kernel = _kda_state_copy_kernel

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        index_dtype: torch.dtype
        cache_rows: int
        cache_stride_elements: int
        payload_elements: int
        cache_aligned: bool
        to_cache: bool

    def dispatch(
        self,
        *,
        dtype: torch.dtype,
        index_dtype: torch.dtype,
        cache_rows: int,
        cache_stride_elements: int,
        payload_elements: int,
        cache_aligned: bool,
        to_cache: bool,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            index_dtype=index_dtype,
            cache_rows=cache_rows,
            cache_stride_elements=cache_stride_elements,
            payload_elements=payload_elements,
            cache_aligned=cache_aligned,
            to_cache=to_cache,
        )

    def get_warmup_keys(
        self,
        *,
        dtype: torch.dtype,
        cache_rows: int,
        cache_stride_elements: int,
        payload_elements: int,
        cache_aligned: bool,
    ) -> list[CompileKey]:
        # These scalars are fixed by the bound cache, including i64 page strides.
        # Selected count occurs only in the grid and cannot add a JIT variant.
        return self._trace_dispatch(self.dispatch)(
            dtype=dtype,
            index_dtype=(torch.int32, torch.int64),
            cache_rows=cache_rows,
            cache_stride_elements=cache_stride_elements,
            payload_elements=payload_elements,
            cache_aligned=cache_aligned,
            to_cache=(False, True),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            cache=TritonWarmupTensor(compile_key.dtype, aligned=compile_key.cache_aligned),
            packed=TritonWarmupTensor(compile_key.dtype),
            indices=TritonWarmupTensor(compile_key.index_dtype),
            flags=TritonWarmupTensor(compile_key.index_dtype if compile_key.to_cache else torch.bool),
            cache_rows=compile_key.cache_rows,
            cache_stride_elements=compile_key.cache_stride_elements,
            payload_elements=compile_key.payload_elements,
            to_cache=compile_key.to_cache,
            selected=1,
        )

    @kernel_launcher
    def __call__(
        self,
        cache: torch.Tensor,
        packed: torch.Tensor,
        indices: torch.Tensor,
        flags: torch.Tensor,
        *,
        cache_rows: int,
        cache_stride_elements: int,
        payload_elements: int,
        to_cache: bool,
        selected: int,
    ) -> LaunchSpec:
        return (selected, triton.cdiv(payload_elements, DEFAULT_KDA_BLOCK_SIZE)), dict(
            index_stride=1,
            flag_stride=0 if to_cache else 1,
            TO_CACHE=to_cache,
            HAS_FLAGS=not to_cache,
            BLOCK_SIZE=DEFAULT_KDA_BLOCK_SIZE,
        )


_KDA_STATE_COPY_KERNEL = KDAStateCopyKernel()


def _validate_cache_layout(state: torch.Tensor) -> tuple[int, int, int]:
    """Validate a real fused cache once at startup and extract launch scalars.

    Preparation reads only metadata; registry warmup uses pointer-free tensors.
    Request indices and packed final states are checked by the sealed plan.
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
