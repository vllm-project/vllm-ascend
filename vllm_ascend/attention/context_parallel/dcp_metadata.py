#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Backend-agnostic decode-context-parallel (DCP) metadata math.

This module owns the pure DCP metadata calculations that used to live inside
``vllm_ascend/worker/dcp_utils.py::DCPManager``: interleave-aware rank-local KV
lengths, the documented interleaved slot layout as a CPU reference
implementation, and the ``AscendDCPMetadata`` builders. The worker manager now
keeps only the orchestration and delegates here.

vLLM and ``AscendDCPMetadata`` are imported lazily inside the functions so the
slot-layout math can be unit tested without a vLLM installation.
"""

from __future__ import annotations

import torch


def dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    dcp_rank: int | None = None,
) -> torch.Tensor:
    """Return interleave-aware KV lengths, for all DCP ranks unless dcp_rank is given."""
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    if dcp_rank is None:
        # Keep the existing all-rank matrix semantics used by the legacy GQA
        # metadata and the DCP metadata generation paths.
        return get_dcp_local_seq_lens(
            seq_lens,
            dcp_size=dcp_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
    return get_dcp_local_seq_lens(
        seq_lens,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )


def local_context_lens(
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (device, cpu) rank-local DCP lengths from the already-sliced mirrors."""
    device_lens = dcp_local_seq_lens(
        seq_lens,
        dcp_size=dcp_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        dcp_rank=dcp_rank,
    )
    cpu_lens = dcp_local_seq_lens(
        seq_lens_cpu,
        dcp_size=dcp_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        dcp_rank=dcp_rank,
    )
    return device_lens, cpu_lens


def reference_dcp_slot_offsets(
    positions: torch.Tensor,
    *,
    block_size: int,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    dcp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU reference for the documented interleaved DCP slot layout.

    Returns ``(owned_mask, logical_block_idx, block_offset)`` per token, using
    the design-doc formula and the same arithmetic as
    ``worker/block_table.py::_compute_dcp_slot_mapping`` for the
    ``physical_block_size == block_size`` / ``blocks_per_phys_block == 1``
    layout. The layout is only meaningful when
    ``block_size % cp_kv_cache_interleave_size == 0`` (see the design doc).
    Pure int64 tensor math; no vLLM/torch_npu needed.
    """
    virtual_block_size = block_size * dcp_size
    virtual_block_index = positions // virtual_block_size
    virtual_block_offsets = positions % virtual_block_size
    # Design doc: interleave_index = offset // interleave;
    # target_rank = interleave_index % dcp_size.
    interleave_index = virtual_block_offsets // cp_kv_cache_interleave_size
    target_rank = interleave_index % dcp_size
    owned_mask = target_rank == dcp_rank
    # Pack the owned tokens of this rank contiguously inside the virtual block.
    local_physical_offsets = (
        virtual_block_offsets // (dcp_size * cp_kv_cache_interleave_size) * cp_kv_cache_interleave_size
        + virtual_block_offsets % cp_kv_cache_interleave_size
    )
    logical_block_idx = virtual_block_index + local_physical_offsets // block_size
    block_offset = local_physical_offsets % block_size
    return owned_mask, logical_block_idx, block_offset


def build_dcp_metadata(
    local_seq_lens: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    max_query_len: int,
    dcp_mtp_attn_mask=None,
):
    """Build AscendDCPMetadata from rank-local lengths (lazy import AscendDCPMetadata)."""
    from vllm_ascend.attention.utils import AscendDCPMetadata

    return AscendDCPMetadata(
        num_computed_tokens_of_dcp=local_seq_lens.numpy(),
        query_lens_cpu=query_lens_cpu,
        max_query_len=max_query_len,
        dcp_mtp_attn_mask=dcp_mtp_attn_mask,
    )


def build_legacy_gqa_dcp_metadata(
    seq_lens_cpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
):
    """Build the legacy GQA AscendDCPMetadata (all-rank local lengths) from CPU mirrors."""
    local_seq_lens = dcp_local_seq_lens(
        seq_lens_cpu,
        dcp_size=dcp_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    return build_dcp_metadata(local_seq_lens, query_lens_cpu, max_query_len, None)
