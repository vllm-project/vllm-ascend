# SPDX-License-Identifier: Apache-2.0
"""Prepared Conv/SSM page copies for the dense MegaKDA ABI.

Each direction uses one launch. Index validation and NULL masking happen in
the copy itself; strided block-table columns need no intermediate tensors.
"""

from types import MappingProxyType

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.kda_state_copy_plan import cache_signature
from vllm_ascend.ops.triton.kda_state_copy_kernel import (
    DEFAULT_KDA_BLOCK_SIZE,
    _configuration,
    _validate_cache_layout,
)

_COPY_ALIGNMENT_BYTES = 16


@triton.jit(do_not_specialize=["conv_count", "state_count", "conv_index_stride", "state_index_stride"])
def _mega_kda_cache_copy_kernel(
    conv_cache,
    state_cache,
    conv_packed,
    state_packed,
    conv_indices,
    state_indices,
    conv_rows,
    state_rows,
    conv_page_stride,
    state_page_stride,
    conv_payload,
    state_payload,
    conv_count,
    state_count,
    conv_index_stride,
    state_index_stride,
    TO_CACHE: tl.constexpr,
    NULL_SLOT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    conv_tiles = tl.cdiv(conv_payload, BLOCK_SIZE)
    state_tiles = tl.cdiv(state_payload, BLOCK_SIZE)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    if program < conv_count * conv_tiles:
        selected = program // conv_tiles
        element = program % conv_tiles * BLOCK_SIZE + lane
        row = tl.load(conv_indices + selected * conv_index_stride).to(tl.int64)
        valid = (row >= 0) & (row < conv_rows) & (row != NULL_SLOT)
        safe_row = tl.where(valid, row, 0)
        cache_offset = safe_row * conv_page_stride + element
        packed_offset = selected * conv_payload + element
        if TO_CACHE:
            value = tl.load(conv_packed + packed_offset, mask=(element < conv_payload) & valid, other=0)
            tl.store(conv_cache + cache_offset, value, mask=(element < conv_payload) & valid)
        else:
            value = tl.load(conv_cache + cache_offset, mask=(element < conv_payload) & valid, other=0)
            tl.store(conv_packed + packed_offset, value, mask=element < conv_payload)
    else:
        program -= conv_count * conv_tiles
        selected = program // state_tiles
        element = program % state_tiles * BLOCK_SIZE + lane
        row = tl.load(state_indices + selected * state_index_stride).to(tl.int64)
        valid = (selected < state_count) & (row >= 0) & (row < state_rows) & (row != NULL_SLOT)
        safe_row = tl.where(valid, row, 0)
        cache_offset = safe_row * state_page_stride + element
        packed_offset = selected * state_payload + element
        if TO_CACHE:
            value = tl.load(state_packed + packed_offset, mask=(element < state_payload) & valid, other=0)
            tl.store(state_cache + cache_offset, value, mask=(element < state_payload) & valid)
        else:
            value = tl.load(state_cache + cache_offset, mask=(element < state_payload) & valid, other=0)
            tl.store(state_packed + packed_offset, value, mask=element < state_payload)


class MegaKDAPagedCopyPlan:
    """Four startup variants, no serving-time JIT or metadata device kernels.

    Only layout metadata and compiled launches are retained. Warmup touches
    disposable tensors, never a scheduler-owned page or a live staging buffer.
    """

    @classmethod
    def prepare(cls, cache, max_batch, spec_len):
        if type(max_batch) is not int or max_batch <= 0 or type(spec_len) is not int or spec_len <= 0:
            raise ValueError("MegaKDA copy requires positive batch and speculative width")
        conv, state = cache
        conv_payload, conv_rows, conv_stride = _validate_cache_layout(conv.unsqueeze(1))
        state_payload, state_rows, state_stride = _validate_cache_layout(state)
        if conv.dtype != torch.bfloat16 or state.dtype != torch.float32:
            raise ValueError("MegaKDA page copies require BF16 Conv and FP32 SSM")
        if conv.device != state.device or any(t.data_ptr() % _COPY_ALIGNMENT_BYTES for t in cache):
            raise ValueError("MegaKDA cache pointers must share a device and be 16-byte aligned")
        plan = cls()
        plan._layouts = tuple(cache_signature(t) for t in cache)
        plan._limits = (max_batch, max_batch * spec_len)
        plan._scalars = (conv_rows, state_rows, conv_stride, state_stride, conv_payload, state_payload)
        plan._tiles = tuple(
            (size + DEFAULT_KDA_BLOCK_SIZE - 1) // DEFAULT_KDA_BLOCK_SIZE for size in (conv_payload, state_payload)
        )
        plan._configuration = _configuration()
        plan._sealed = False
        compiled = {}
        with torch.npu.device(conv.device):
            scratch_conv = torch.zeros(conv_payload, dtype=conv.dtype, device=conv.device)
            scratch_state = torch.zeros(state_payload, dtype=state.dtype, device=state.device)
            packed_conv = torch.empty_like(scratch_conv)
            packed_state = torch.empty_like(scratch_state)
            if _mega_kda_cache_copy_kernel.pre_run_hooks:
                raise RuntimeError("MegaKDA copy JIT hooks are unsupported")
            for dtype in (torch.int32, torch.int64):
                # -1 masks all cache accesses, including with NULL_BLOCK_ID != 0.
                # Compile with only element alignment: any block-table slice
                # can then be passed directly, including non-16-byte offsets.
                indices = torch.full((2,), -1, dtype=dtype, device=conv.device)[1:]
                for to_cache in (False, True):
                    compiled[dtype, to_cache] = _mega_kda_cache_copy_kernel[(sum(plan._tiles),)](
                        scratch_conv,
                        scratch_state,
                        packed_conv,
                        packed_state,
                        indices,
                        indices,
                        *plan._scalars,
                        1,
                        1,
                        1,
                        1,
                        TO_CACHE=to_cache,
                        NULL_SLOT=NULL_BLOCK_ID,
                        BLOCK_SIZE=DEFAULT_KDA_BLOCK_SIZE,
                    )
            torch.npu.synchronize()
        plan._compiled = MappingProxyType(compiled)
        return plan

    def seal(self):
        if self._configuration != _configuration() or _mega_kda_cache_copy_kernel.pre_run_hooks:
            raise RuntimeError("MegaKDA copy configuration/hooks changed during preparation")
        self._sealed = True

    def copy(self, cache, conv, state, conv_indices, state_indices, *, to_cache):
        if not self._sealed or tuple(cache_signature(t) for t in cache) != self._layouts:
            raise RuntimeError("unprepared MegaKDA cache layout")
        if _mega_kda_cache_copy_kernel.pre_run_hooks:
            raise RuntimeError("MegaKDA copy JIT hooks changed after seal")
        indices = (conv_indices, state_indices)
        if any(
            t.ndim != 1 or t.dtype not in (torch.int32, torch.int64) or t.device != cache[0].device for t in indices
        ):
            raise ValueError("MegaKDA copy indices must be same-device INT32/INT64 vectors")
        if conv_indices.dtype != state_indices.dtype:
            raise ValueError("MegaKDA Conv/SSM indices must have the same dtype")
        counts = tuple(t.numel() for t in indices)
        if any(count > limit for count, limit in zip(counts, self._limits)):
            raise ValueError("MegaKDA copy exceeds prepared batch capacity")
        for source, packed, index in zip(cache, (conv, state), indices):
            if (
                packed.shape != (index.numel(), *source.shape[1:])
                or packed.device != source.device
                or packed.dtype != source.dtype
                or not packed.is_contiguous()
                or packed.data_ptr() % _COPY_ALIGNMENT_BYTES
            ):
                raise ValueError("MegaKDA copy requires aligned packed payloads with matching shape, dtype and device")
        grid = sum(count * tiles for count, tiles in zip(counts, self._tiles))
        if not grid:
            return
        with torch.npu.device(cache[0].device):
            self._compiled[conv_indices.dtype, to_cache][(grid, 1, 1)](
                *cache,
                conv,
                state,
                *indices,
                *self._scalars,
                *counts,
                conv_indices.stride(0),
                state_indices.stride(0),
                to_cache,
                NULL_BLOCK_ID,
                DEFAULT_KDA_BLOCK_SIZE,
            )
