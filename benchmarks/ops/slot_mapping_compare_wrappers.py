"""Launch adapters for upstream and Ascend slot-mapping kernel benchmarks."""

from __future__ import annotations

import torch
from vllm.v1.worker.gpu.block_table import (
    _compute_slot_mappings_kernel as upstream_compute_slot_mappings_kernel,
)

from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel as ascend_compute_slot_mappings_kernel,
)


def _grid(block_table_ptrs: torch.Tensor, idx_mapping: torch.Tensor) -> tuple[int, int]:
    return block_table_ptrs.numel(), idx_mapping.numel() + 1


def launch_ascend(
    max_num_tokens: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    pos: torch.Tensor,
    block_table_ptrs: torch.Tensor,
    block_table_strides: torch.Tensor,
    block_sizes: torch.Tensor,
    kernel_block_sizes: torch.Tensor,
    slot_mapping_enabled: torch.Tensor,
    slot_mappings: torch.Tensor,
    slot_mappings_stride: int,
    cp_rank: int,
    CP_SIZE: int,
    CP_INTERLEAVE: int,
    PAD_ID: int,
    TRITON_BLOCK_SIZE: int,
    BLOCK_TABLE_WINDOW_SIZE: int,
) -> None:
    """Launch the Ascend implementation; upstream-only tensors are ignored."""
    del kernel_block_sizes, slot_mapping_enabled
    ascend_compute_slot_mappings_kernel[_grid(block_table_ptrs, idx_mapping)](
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        pos,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
        slot_mappings,
        slot_mappings_stride,
        cp_rank,
        CP_SIZE=CP_SIZE,
        CP_INTERLEAVE=CP_INTERLEAVE,
        PAD_ID=PAD_ID,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        BLOCK_TABLE_WINDOW_SIZE=BLOCK_TABLE_WINDOW_SIZE,
    )


def launch_upstream(
    max_num_tokens: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    pos: torch.Tensor,
    block_table_ptrs: torch.Tensor,
    block_table_strides: torch.Tensor,
    block_sizes: torch.Tensor,
    kernel_block_sizes: torch.Tensor,
    slot_mapping_enabled: torch.Tensor,
    slot_mappings: torch.Tensor,
    slot_mappings_stride: int,
    cp_rank: int,
    CP_SIZE: int,
    CP_INTERLEAVE: int,
    PAD_ID: int,
    TRITON_BLOCK_SIZE: int,
    BLOCK_TABLE_WINDOW_SIZE: int,
) -> None:
    """Launch the upstream reference across its supported signatures."""
    del BLOCK_TABLE_WINDOW_SIZE
    kernel_args = (
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        pos,
        block_table_ptrs,
        block_table_strides,
        block_sizes,
    )
    if "kernel_block_sizes" in upstream_compute_slot_mappings_kernel.arg_names:
        kernel_args += (kernel_block_sizes, slot_mapping_enabled)
    upstream_compute_slot_mappings_kernel[_grid(block_table_ptrs, idx_mapping)](
        *kernel_args,
        slot_mappings,
        slot_mappings_stride,
        cp_rank,
        CP_SIZE=CP_SIZE,
        CP_INTERLEAVE=CP_INTERLEAVE,
        PAD_ID=PAD_ID,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
    )
