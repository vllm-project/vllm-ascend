# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the DSpark SWA slot-indices Triton kernel before ACL graph capture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.dspark_swa_indices import (
    build_dspark_swa_slot_indices_triton,
    dspark_swa_indices_supported,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


@torch.inference_mode()
def dspark_swa_indices_warmup(worker: NPUWorker) -> None:
    """JIT-warm the fused kernel for the exact (B, W) variant seen at runtime.

    The kernel's JIT key is the constexpr set (WINDOW_SIZE, BLOCK_SIZE,
    INDEX_W, BLOCK_W, ROW_POW2, Q_POW2). All of them are fixed by the startup
    configuration: ROW_POW2 derives from the block-table width the draft
    builder sees at runtime -- the full per-group table, or the
    SlidingWindowAdapter's cropped clone when ``draft_window_size`` is set --
    and Q_POW2 from the spec token count. A post-capture JIT would block the
    replay path, so this runs from ``kernel_warmup`` -- after
    ``initialize_kv_cache`` (the proposer has registered the exact per-request
    query count and the persistent buffers exist) and before ``capture_model``.
    """
    if not HAS_TRITON:
        return
    runner = worker.model_runner
    drafter = getattr(runner, "drafter", None)
    if drafter is None or not hasattr(drafter, "draft_attn_groups"):
        return

    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

    sliding_window_adapter = getattr(drafter, "sliding_window", None)
    for attn_group in drafter.draft_attn_groups:
        builder = attn_group.get_metadata_builder()
        if not isinstance(builder, AscendDSAMetadataBuilder):
            continue
        # Only the device-metadata (graph-capture) path needs pre-capture JIT;
        # the eager path compiles on first call outside any graph.
        if not getattr(builder, "_device_metadata_enabled", False):
            continue
        if build_dspark_swa_slot_indices_triton is None or builder._dspark_index_width is None:
            continue
        # builder.block_table is assigned per step from the runtime metadata
        # and is still None here. Compute the width the builder will see: the
        # sliding-window adapter crops the table to its window clone, the
        # plain path reads the full per-group table from the input batch.
        if sliding_window_adapter is not None:
            table_width = sliding_window_adapter.max_window_blocks
        else:
            gid = attn_group.kv_cache_group_id
            table_width = runner.input_batch.block_table[gid].get_device_tensor().shape[1]
        if not dspark_swa_indices_supported(builder.storage_block_size, table_width):
            continue
        max_num_reqs = builder.max_num_reqs_for_dspark()
        num_query_per_req = builder.dspark_num_query_per_req()
        # Dummies mirror the runtime buffers: query_start_loc is the runner's
        # int32 buffer, seq_lens int32, block_table int32, so the warmup
        # compiles the exact pointer specializations the real call launches.
        # All-zero metadata means the launch exercises compilation and the
        # capacity grid only; no gather or store executes.
        build_dspark_swa_slot_indices_triton(
            torch.zeros(max_num_reqs, table_width, dtype=torch.int32, device=builder.device),
            builder.speculative_config.num_speculative_tokens,
            builder.model_config.hf_config.sliding_window,
            builder.storage_block_size,
            torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=builder.device),
            torch.zeros(max_num_reqs, dtype=torch.int32, device=builder.device),
            num_decode_tokens=max_num_reqs * num_query_per_req,
            index_width=builder._dspark_index_width,
            indices_output=builder.dspark_swa_indices_buffer,
            lens_output=builder.dspark_lens_scratch,
            max_num_reqs=max_num_reqs,
            num_query_per_req=num_query_per_req,
        )
