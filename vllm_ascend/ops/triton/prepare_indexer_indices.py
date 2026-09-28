# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert score-ordered QLI indices into visible, chronological positions."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_rows", "blocks_per_core"])
def _prepare_indexer_indices_kernel(
    selected_ptr,
    positions_ptr,
    output_ptr,
    num_rows,
    TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    blocks_per_core,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    SENTINEL: tl.constexpr,
    SORT_KEY_SHIFT: tl.constexpr,
    NEGATIVE_KEY_BASE: tl.constexpr,
):
    first_block = tl.program_id(0) * blocks_per_core
    last_block = tl.minimum(first_block + blocks_per_core, tl.cdiv(num_rows, BLOCK_ROWS))
    columns = tl.arange(0, BLOCK_COLS)
    for block in range(first_block, last_block):
        rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * TOPK + columns[None, :]
        mask = (rows[:, None] < num_rows) & (columns[None, :] < TOPK)
        selected = tl.load(selected_ptr + offsets, mask, other=SENTINEL)
        positions = tl.load(positions_ptr + rows, rows < num_rows, other=-1)
        visible = (positions + 1) // COMPRESS_RATIO
        valid = (selected >= 0) & (selected < visible[:, None])
        selected = tl.where(valid, selected, SENTINEL)
        # Use native FP32 sort even when the INT32 implementation is absent.
        # Encode ordered, normal FP32 bit patterns without numeric conversion:
        # [0, 2**24) uses negative keys; larger indices use positive keys.
        key_bits = tl.where(selected < 2 * SORT_KEY_SHIFT, NEGATIVE_KEY_BASE - selected, selected - SORT_KEY_SHIFT)
        sort_keys = key_bits.to(tl.float32, bitcast=True)
        sort_keys = tl.extra.cann.extension.sort(sort_keys, dim=1, descending=False)
        key_bits = sort_keys.to(tl.int32, bitcast=True)
        selected = tl.where(key_bits < 0, NEGATIVE_KEY_BASE - key_bits, key_bits + SORT_KEY_SHIFT)
        selected = tl.where(selected == SENTINEL, -1, selected)
        tl.store(output_ptr + offsets, selected, mask)


class PrepareIndexerIndicesKernel(VllmTritonJitKernel["PrepareIndexerIndicesKernel.CompileKey"]):
    kernel = _prepare_indexer_indices_kernel

    @dataclass(frozen=True)
    class CompileKey:
        topk: int
        compress_ratio: int
        block_rows: int
        positions_dtype: torch.dtype

    def dispatch(self, *, topk: int, compress_ratio: int, block_rows: int, positions_dtype: torch.dtype) -> CompileKey:
        return self.CompileKey(
            topk=topk, compress_ratio=compress_ratio, block_rows=block_rows, positions_dtype=positions_dtype
        )

    def get_warmup_keys(self, context: Any) -> list[CompileKey]:
        padded_topk = 1 << (context.topk - 1).bit_length()
        max_block_rows = 128 * 1024 // (padded_topk * 4 * 8)
        rows = []
        for tokens in context.token_counts:
            raw_rows = (tokens + context.num_cores - 1) // context.num_cores
            block_rows = min(max_block_rows, 1 << (max(raw_rows, 1) - 1).bit_length())
            for ratio in context.compress_ratios:
                for positions_dtype in (torch.int32, torch.int64):
                    rows.append(
                        dict(
                            topk=context.topk,
                            compress_ratio=ratio,
                            block_rows=block_rows,
                            positions_dtype=positions_dtype,
                        )
                    )
        from vllm.model_executor.warmup.jit_warmup import zip_inputs

        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        shape = (1, compile_key.topk)
        strides = (compile_key.topk, 1)
        return dict(
            selected=TritonWarmupTensor(torch.int32, shape=shape, strides=strides),
            positions=TritonWarmupTensor(compile_key.positions_dtype, shape=(1,)),
            output=TritonWarmupTensor(torch.int32, shape=shape, strides=strides),
            num_rows=1,
            TOPK=compile_key.topk,
            COMPRESS_RATIO=compile_key.compress_ratio,
            blocks_per_core=1,
            BLOCK_ROWS=compile_key.block_rows,
            BLOCK_COLS=1 << (compile_key.topk - 1).bit_length(),
            SENTINEL=torch.iinfo(torch.int32).max,
            SORT_KEY_SHIFT=1 << 23,
            NEGATIVE_KEY_BASE=0x81800000 - (1 << 32),
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        selected,
        positions,
        output,
        *,
        num_rows,
        TOPK,
        COMPRESS_RATIO,
        blocks_per_core,
        BLOCK_ROWS,
        BLOCK_COLS,
        SENTINEL,
        SORT_KEY_SHIFT,
        NEGATIVE_KEY_BASE,
        grid_size,
    ) -> LaunchSpec:
        return (grid_size,), dict(multibuffer=False, unit_flag=False)


_PREPARE_INDEXER_INDICES_KERNEL = PrepareIndexerIndicesKernel()


def prepare_indexer_indices(
    selected: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Filter and sort [tokens, topk] INT32 indices, with invalid slots last.

    ``output`` lets callers with a stable destination buffer avoid an
    intermediate device-to-device copy after the Triton kernel completes.
    """
    assert selected.ndim == 2 and selected.dtype == torch.int32
    assert positions.ndim == 1 and positions.shape[0] == selected.shape[0]
    assert positions.dtype in (torch.int32, torch.int64)
    assert compress_ratio in (1, 2)
    num_rows, topk = selected.shape
    assert 1 <= topk <= 2048
    selected = selected.contiguous()
    positions = positions.contiguous()
    if output is None:
        output = torch.empty_like(selected)
    else:
        assert output.shape == selected.shape
        assert output.dtype == torch.int32
        assert output.device == selected.device
        assert output.is_contiguous()
    if num_rows == 0:
        return output

    num_cores = get_vectorcore_num()
    block_cols = triton.next_power_of_2(topk)
    # Budget 128 KiB for INT32 sort data and scratch (eight buffers).
    max_block_rows = 128 * 1024 // (block_cols * 4 * 8)
    # Core boundaries must also align to 32 bytes for arbitrary TopK widths.
    aligned_rows = 8 // math.gcd(topk, 8)
    block_rows = min(max_block_rows, triton.next_power_of_2(triton.cdiv(num_rows, num_cores)))
    num_blocks = triton.cdiv(num_rows, block_rows)
    aligned_blocks = triton.cdiv(aligned_rows, block_rows)
    grid = min(triton.cdiv(num_blocks, aligned_blocks), num_cores)
    blocks_per_core = triton.cdiv(triton.cdiv(num_blocks, grid), aligned_blocks) * aligned_blocks
    _PREPARE_INDEXER_INDICES_KERNEL(
        selected=selected,
        positions=positions,
        output=output,
        num_rows=num_rows,
        TOPK=topk,
        COMPRESS_RATIO=compress_ratio,
        blocks_per_core=blocks_per_core,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        SENTINEL=torch.iinfo(torch.int32).max,
        SORT_KEY_SHIFT=1 << 23,
        NEGATIVE_KEY_BASE=0x81800000 - (1 << 32),
        grid_size=grid,
    )
    return output
