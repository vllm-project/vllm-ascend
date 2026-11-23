# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse native INT8 indexer K quantization and paged K/FP16-scale writes."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

HEAD_DIM = 128
DMA_BLOCK_BYTES = 32
SCALES_PER_DMA_BLOCK = DMA_BLOCK_BYTES // 2
# Each destination block leader searches the small decode batch for its rows.
# Leave large prefill batches on the existing path rather than scaling that scan.
MAX_FUSED_TOKENS = 128


def supports_indexer_quant_cache_int8(key: torch.Tensor, scale: torch.Tensor, num_tokens: int) -> bool:
    """Metadata-only eligibility check; never read device tensor contents."""
    return (
        0 < num_tokens <= MAX_FUSED_TOKENS
        and key.dtype == torch.int8
        and scale.dtype == torch.float16
        and key.ndim == scale.ndim == 4
        and key.shape[:3] == scale.shape[:3]
        and key.shape[2:] == (1, HEAD_DIM)
        and scale.shape[2:] == (1, 1)
        and key.shape[0] > 0
        and key.shape[1] > 0
        and key.shape[1] % SCALES_PER_DMA_BLOCK == 0
        and key.stride(-1) == scale.stride(-1) == 1
        and key.stride(1) == HEAD_DIM
        and scale.stride(1) == 1
        and key.stride(0) >= key.shape[1] * HEAD_DIM
        and scale.stride(0) >= scale.shape[1]
        and scale.stride(0) % SCALES_PER_DMA_BLOCK == 0
        and key.stride(0) % DMA_BLOCK_BYTES == 0
        and key.data_ptr() % DMA_BLOCK_BYTES == 0
        and scale.data_ptr() % DMA_BLOCK_BYTES == 0
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _write_indexer_cache_block(
    key_ptr,
    scale_ptr,
    x_ptr,
    source_slots,
    source_rows,
    first_slot,
    num_tokens,
    PAGE_SIZE: tl.constexpr,
    KEY_PAGE_STRIDE: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    X_COL_STRIDE: tl.constexpr,
    INPUT_ROWS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    SCALES_PER_BLOCK: tl.constexpr,
    QUANT_MAX: tl.constexpr,
):
    lanes = tl.arange(0, SCALES_PER_BLOCK)
    columns = tl.arange(0, HEAD_SIZE)
    matches = (source_slots[None, :] == first_slot + lanes[:, None]) & (source_rows[None, :] < num_tokens)
    rows = tl.min(tl.where(matches, source_rows[None, :], INPUT_ROWS), axis=1)
    active = rows < num_tokens
    safe_rows = tl.minimum(rows, num_tokens - 1)
    x = tl.load(
        x_ptr + safe_rows[:, None] * X_ROW_STRIDE + columns[None, :] * X_COL_STRIDE,
        active[:, None],
        other=0,
    ).to(tl.float32)
    abs_max = tl.max(tl.abs(x), axis=1)
    scale = tl.div_rn(abs_max, QUANT_MAX)
    # Quantize using the FP32 scale. Rounding it to FP16 before division
    # changes INT8 values and does not match the split DynamicQuant path.
    divisor = tl.where(scale > 0, scale, 1.0)
    normalized = tl.div_rn(x, divisor[:, None])
    quantized = tl.extra.cann.libdevice.nearbyint(normalized)
    quantized = tl.minimum(tl.maximum(quantized, -QUANT_MAX), QUANT_MAX).to(tl.int8)
    page = first_slot // PAGE_SIZE
    position = first_slot % PAGE_SIZE
    key_start = tl.multiple_of(page * KEY_PAGE_STRIDE + position * HEAD_SIZE, 32)
    scale_start = tl.multiple_of(page * SCALE_PAGE_STRIDE + position, SCALES_PER_BLOCK)
    key_addresses = key_ptr + key_start + lanes[:, None] * HEAD_SIZE + columns[None, :]
    scale_addresses = scale_ptr + scale_start + lanes
    # Read/modify/write complete aligned blocks. Arbitrary masked scatter
    # stores can lower to sync_block_lock, whose launcher initialization
    # uses synchronous H2D copies on affected Triton-Ascend versions and
    # fails during GLOBAL graph capture. Preserve untouched lanes in UB;
    # both GM stores below are contiguous and deliberately unmasked.
    old_key = tl.load(key_addresses)
    old_scale = tl.load(scale_addresses)
    tl.store(key_addresses, tl.where(active[:, None], quantized, old_key))
    tl.store(scale_addresses, tl.where(active, scale.to(tl.float16), old_scale))


@triton.jit(do_not_specialize=["num_tokens"])
def _indexer_quant_cache_int8_kernel(
    key_ptr,
    scale_ptr,
    x_ptr,
    slots_ptr,
    num_tokens,
    NUM_SLOTS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KEY_PAGE_STRIDE: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    X_COL_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    INPUT_ROWS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    SCALES_PER_BLOCK: tl.constexpr,
    QUANT_MAX: tl.constexpr,
):
    program = tl.program_id(0)
    slot = tl.load(slots_ptr + program * SLOT_STRIDE).to(tl.int64)
    if (slot >= 0) & (slot < NUM_SLOTS):
        source_rows = tl.arange(0, INPUT_ROWS)
        source_slots = tl.load(slots_ptr + source_rows * SLOT_STRIDE, source_rows < num_tokens, other=-1).to(tl.int64)
        # Pages and scale page strides are multiples of a 32-byte scale block.
        # Thus each logical group of 16 slots maps to one complete physical block.
        first_slot = (slot // SCALES_PER_BLOCK) * SCALES_PER_BLOCK
        in_block = (
            (source_rows < num_tokens) & (source_slots >= first_slot) & (source_slots < first_slot + SCALES_PER_BLOCK)
        )
        leader = tl.min(tl.where(in_block, source_rows, INPUT_ROWS), axis=0)
        if program == leader:
            _write_indexer_cache_block(
                key_ptr,
                scale_ptr,
                x_ptr,
                source_slots,
                source_rows,
                first_slot,
                num_tokens,
                PAGE_SIZE,
                KEY_PAGE_STRIDE,
                SCALE_PAGE_STRIDE,
                X_ROW_STRIDE,
                X_COL_STRIDE,
                INPUT_ROWS,
                HEAD_SIZE,
                SCALES_PER_BLOCK,
                QUANT_MAX,
            )


class IndexerQuantCacheInt8Kernel(AscendTritonJitKernel["IndexerQuantCacheInt8Kernel.CompileKey"]):
    kernel = _indexer_quant_cache_int8_kernel
    bind_launch_inputs = False

    @dataclass(frozen=True)
    class CompileKey:
        num_slots: int
        page_size: int
        key_page_stride: int
        scale_page_stride: int
        x_row_stride: int
        x_col_stride: int
        slot_stride: int
        slot_dtype: torch.dtype
        input_rows: int
        x_aligned: bool
        slots_aligned: bool

    def get_warmup_keys(
        self,
        *,
        num_slots: int,
        page_size: int,
        key_page_stride: int,
        scale_page_stride: int,
        x_row_stride: int = HEAD_DIM,
        x_col_stride: int = 1,
        slot_stride: int = 1,
        x_aligned: bool = True,
    ) -> list[CompileKey]:
        # Hadamard outputs are contiguous/aligned in the model. Slot mappings
        # can start at an offset in a per-group buffer; cover both pointer keys.
        return [
            self.CompileKey(
                num_slots,
                page_size,
                key_page_stride,
                scale_page_stride,
                x_row_stride,
                x_col_stride,
                slot_stride,
                slot_dtype,
                1 << power,
                x_aligned,
                slots_aligned,
            )
            for power in range(MAX_FUSED_TOKENS.bit_length())
            for slot_dtype in (torch.int32, torch.int64)
            for slots_aligned in (True, False)
        ]

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        key = compile_key
        pages = key.num_slots // key.page_size
        return dict(
            key_cache=TritonWarmupTensor(
                torch.int8,
                shape=(pages, key.page_size, 1, HEAD_DIM),
                strides=(key.key_page_stride, HEAD_DIM, HEAD_DIM, 1),
            ),
            scale_cache=TritonWarmupTensor(
                torch.float16,
                shape=(pages, key.page_size, 1, 1),
                strides=(key.scale_page_stride, 1, 1, 1),
            ),
            x=TritonWarmupTensor(
                torch.bfloat16,
                aligned=key.x_aligned,
                shape=(key.input_rows, HEAD_DIM),
                strides=(key.x_row_stride, key.x_col_stride),
            ),
            slots=TritonWarmupTensor(
                key.slot_dtype,
                aligned=key.slots_aligned,
                shape=(key.input_rows,),
                strides=(key.slot_stride,),
            ),
        )

    @kernel_launcher
    def __call__(
        self, key_cache: torch.Tensor, scale_cache: torch.Tensor, x: torch.Tensor, slots: torch.Tensor
    ) -> LaunchSpec:
        return (x.shape[0],), dict(
            key_ptr=key_cache,
            scale_ptr=scale_cache,
            x_ptr=x,
            slots_ptr=slots,
            num_tokens=x.shape[0],
            NUM_SLOTS=key_cache.shape[0] * key_cache.shape[1],
            PAGE_SIZE=key_cache.shape[1],
            KEY_PAGE_STRIDE=key_cache.stride(0),
            SCALE_PAGE_STRIDE=scale_cache.stride(0),
            X_ROW_STRIDE=x.stride(0),
            X_COL_STRIDE=x.stride(1),
            SLOT_STRIDE=slots.stride(0),
            INPUT_ROWS=triton.next_power_of_2(x.shape[0]),
            HEAD_SIZE=HEAD_DIM,
            SCALES_PER_BLOCK=SCALES_PER_DMA_BLOCK,
            QUANT_MAX=127.0,
        )


_INDEXER_QUANT_CACHE_INT8_KERNEL = IndexerQuantCacheInt8Kernel()


def register_indexer_quant_cache_int8_warmup(key_cache: torch.Tensor, scale_cache: torch.Tensor) -> None:
    """Register final cache geometry without allocating or writing device data."""
    if not supports_indexer_quant_cache_int8(key_cache, scale_cache, 1):
        return
    _INDEXER_QUANT_CACHE_INT8_KERNEL.register_warmup(
        num_slots=key_cache.shape[0] * key_cache.shape[1],
        page_size=key_cache.shape[1],
        key_page_stride=key_cache.stride(0),
        scale_page_stride=scale_cache.stride(0),
    )


def indexer_quant_cache_int8(
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    x: torch.Tensor,
    slots: torch.Tensor,
    *,
    quant_mode: str = "int8",
    round_scale: bool = False,
) -> None:
    """Write BF16 [tokens, 128] K to INT8/FP16 paged cache in one launch.

    Cache views may share a padded hybrid allocation. All strides are in the
    corresponding tensor's elements; pointers already include storage offsets.
    Slots are flat INT32/INT64 device indices, with negative padding ignored.
    Valid slots must be unique, as with the original scatter-cache operation.
    Whole aligned cache blocks are updated in place, preserving untouched lanes.
    No intermediate GM tensors or host/device sync are needed.
    """
    assert quant_mode == "int8" and not round_scale
    assert x.ndim == 2 and x.shape[1] == HEAD_DIM and x.dtype == torch.bfloat16
    assert slots.ndim == 1 and slots.numel() == x.shape[0]
    assert slots.dtype in (torch.int32, torch.int64)
    assert x.device == slots.device == key_cache.device == scale_cache.device
    if x.shape[0] == 0:
        return
    assert supports_indexer_quant_cache_int8(key_cache, scale_cache, x.shape[0])
    _INDEXER_QUANT_CACHE_INT8_KERNEL(key_cache, scale_cache, x, slots)
