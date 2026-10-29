# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert score-ordered QLI indices into visible, chronological positions."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
)
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_FUSED_LENGTHS_MAX_ELEMENTS = 1024 * 512


@triton.jit(do_not_specialize=["num_rows", "blocks_per_core"])
def _prepare_indexer_indices_kernel(
    selected_ptr,
    positions_ptr,
    output_ptr,
    lengths_ptr,
    num_rows,
    TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    blocks_per_core,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    SENTINEL: tl.constexpr,
    SORT_KEY_SHIFT: tl.constexpr,
    NEGATIVE_KEY_BASE: tl.constexpr,
    WRITE_LENGTHS: tl.constexpr,
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
        if WRITE_LENGTHS:
            valid_lengths = tl.sum(valid.to(tl.int32), axis=1)
            tl.store(lengths_ptr + rows, valid_lengths, rows < num_rows)


def _indexer_max_block_rows(block_cols: int) -> int:
    """Budget 128 KiB for INT32 sort data and scratch (eight buffers)."""
    return 128 * 1024 // (block_cols * 4 * 8)


def collect_indexer_warmup_block_rows(topk: int, num_cores: int, max_tokens: int) -> list[int]:
    """Every ``BLOCK_ROWS`` reachable for token counts up to ``max_tokens``.

    ``BLOCK_ROWS = min(max_block_rows, pow2floor(cdiv(num_rows, cores)))``
    and ``num_rows`` spans ``[1, max_tokens]``, so the reachable set is every
    power of two up to ``min(max_block_rows, pow2floor(cdiv(max_tokens,
    cores)))``.
    """
    block_cols = triton.next_power_of_2(topk)
    max_rows = min(
        _indexer_max_block_rows(block_cols),
        triton.next_power_of_2(triton.cdiv(max_tokens, num_cores)),
    )
    values = []
    block_rows = 1
    while block_rows <= max_rows:
        values.append(block_rows)
        block_rows *= 2
    return values


class PrepareIndexerIndicesKernel(AscendTritonJitKernel["PrepareIndexerIndicesKernel.CompileKey"]):
    kernel = _prepare_indexer_indices_kernel

    @dataclass(frozen=True)
    class CompileKey:
        topk: int
        compress_ratio: int
        block_rows: int
        block_cols: int
        write_lengths: bool
        positions_dtype: torch.dtype

    def dispatch(
        self,
        *,
        topk: int,
        compress_ratio: int,
        block_rows: int,
        block_cols: int,
        write_lengths: bool,
        positions_dtype: torch.dtype,
    ) -> "PrepareIndexerIndicesKernel.CompileKey":
        return self.CompileKey(
            topk=topk,
            compress_ratio=compress_ratio,
            block_rows=block_rows,
            block_cols=block_cols,
            write_lengths=write_lengths,
            positions_dtype=positions_dtype,
        )

    def get_warmup_keys(self, context: Any) -> list["PrepareIndexerIndicesKernel.CompileKey"]:
        # ``context`` is the DeepseekV41Indexer registration tuple built by
        # ``register_indexer_warmup``; the runtime positions tensor is int64.
        topk, compress_ratio, vllm_config = context
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # WRITE_LENGTHS bumps BLOCK_COLS to >= 32, and the runtime then caps
        # BLOCK_ROWS with the wider column budget; mirror that pairing so the
        # wide-column variants never claim a row tile the device cannot hold.
        base_rows = collect_indexer_warmup_block_rows(topk, get_vectorcore_num(), max_tokens)
        wl_rows_cap = _indexer_max_block_rows(32)
        rows = []
        for block_rows in base_rows:
            for write_lengths in (False, True):
                # write_lengths bumps BLOCK_COLS to the reduction width.
                block_cols = triton.next_power_of_2(topk)
                if write_lengths:
                    block_cols = max(block_cols, 32)
                    if block_rows > wl_rows_cap:
                        continue
                rows.append(
                    dict(
                        topk=topk,
                        compress_ratio=compress_ratio,
                        block_rows=block_rows,
                        block_cols=block_cols,
                        write_lengths=write_lengths,
                        positions_dtype=torch.int64,
                    )
                )
        return self._trace_dispatch(self.dispatch)(zip_inputs(*rows)) if rows else []

    def warmup_inputs(self, compile_key: "PrepareIndexerIndicesKernel.CompileKey") -> dict[str, Any]:
        shape = (1, compile_key.topk)
        strides = (compile_key.topk, 1)
        return dict(
            selected=TritonWarmupTensor(torch.int32, shape=shape, strides=strides),
            positions=TritonWarmupTensor(compile_key.positions_dtype, shape=(1,)),
            output=TritonWarmupTensor(torch.int32, shape=shape, strides=strides),
            lengths=TritonWarmupTensor(torch.int32, shape=(1, 1)),
            num_rows=1,
            TOPK=compile_key.topk,
            COMPRESS_RATIO=compile_key.compress_ratio,
            blocks_per_core=1,
            BLOCK_ROWS=compile_key.block_rows,
            BLOCK_COLS=compile_key.block_cols,
            SENTINEL=torch.iinfo(torch.int32).max,
            SORT_KEY_SHIFT=1 << 23,
            NEGATIVE_KEY_BASE=0x81800000 - (1 << 32),
            WRITE_LENGTHS=compile_key.write_lengths,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        selected: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        lengths: torch.Tensor,
        *,
        num_rows: int,
        TOPK: int,
        COMPRESS_RATIO: int,
        blocks_per_core: int,
        BLOCK_ROWS: int,
        BLOCK_COLS: int,
        SENTINEL: int,
        SORT_KEY_SHIFT: int,
        NEGATIVE_KEY_BASE: int,
        WRITE_LENGTHS: bool,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            multibuffer=False,
            unit_flag=False,
        )


_PREPARE_INDEXER_INDICES_KERNEL = PrepareIndexerIndicesKernel()


def register_indexer_warmup(topk: int, compress_ratio: int, vllm_config: Any) -> None:
    """Register the indexer index-postprocessing kernel for one ratio.

    No-op outside a ``JitWarmupRegistry.activate()`` window.
    """
    _PREPARE_INDEXER_INDICES_KERNEL.register_warmup((topk, compress_ratio, vllm_config))


def prepare_indexer_indices(
    selected: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    output: torch.Tensor | None = None,
    *,
    indices_output: torch.Tensor | None = None,
    lengths_output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Filter/sort indices into optional graph-stable output buffers."""
    assert selected.ndim == 2 and selected.dtype == torch.int32
    assert positions.ndim == 1 and positions.shape[0] == selected.shape[0]
    assert positions.dtype in (torch.int32, torch.int64)
    assert compress_ratio in (1, 2)
    num_rows, topk = selected.shape
    assert 1 <= topk <= 2048
    selected = selected.contiguous()
    positions = positions.contiguous()
    if output is not None and indices_output is not None:
        raise ValueError("Specify only one of output and indices_output")
    indices_output = output if indices_output is None else indices_output
    expected_indices_shape = (num_rows, topk)
    if indices_output is None:
        output = torch.empty_like(selected)
    else:
        if (
            tuple(indices_output.shape) != expected_indices_shape
            or indices_output.dtype != torch.int32
            or indices_output.device != selected.device
            or not indices_output.is_contiguous()
        ):
            raise ValueError(
                f"indices_output must be contiguous INT32 on the selected device with shape {expected_indices_shape}"
            )
        output = indices_output
    if lengths_output is not None:
        expected_shape = (num_rows, 1)
        if (
            tuple(lengths_output.shape) != expected_shape
            or lengths_output.dtype != torch.int32
            or lengths_output.device != selected.device
            or not lengths_output.is_contiguous()
        ):
            raise ValueError(
                f"lengths_output must be contiguous INT32 on the selected device with shape {expected_shape}"
            )
    if num_rows == 0:
        return output

    num_cores = get_vectorcore_num()
    block_cols = triton.next_power_of_2(topk)
    reduction_cols = max(block_cols, 32)
    write_lengths = lengths_output is not None and num_rows * reduction_cols <= _FUSED_LENGTHS_MAX_ELEMENTS
    if write_lengths:
        block_cols = max(block_cols, 32)
    # Budget 128 KiB for INT32 sort data and scratch (eight buffers).
    max_block_rows = _indexer_max_block_rows(block_cols)
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
        lengths=(output if lengths_output is None else lengths_output),
        num_rows=num_rows,
        TOPK=topk,
        COMPRESS_RATIO=compress_ratio,
        blocks_per_core=blocks_per_core,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        SENTINEL=torch.iinfo(torch.int32).max,
        SORT_KEY_SHIFT=1 << 23,
        NEGATIVE_KEY_BASE=0x81800000 - (1 << 32),
        WRITE_LENGTHS=write_lengths,
        grid_size=grid,
    )
    if lengths_output is not None and not write_lengths:
        torch.sum(
            output >= 0,
            dim=-1,
            keepdim=True,
            dtype=torch.int32,
            out=lengths_output,
        )
    return output
