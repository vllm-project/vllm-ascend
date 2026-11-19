# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CANN query-fragment metadata for upstream ShardQ exchanges."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.logger import logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.ops.shardq import ShardQMetadata, build_shardq_split

from vllm_ascend.attention.indexer import INDEXER_K_CACHE_SLOT, INDEXER_SCALE_CACHE_SLOT
from vllm_ascend.device.device_op import DeviceOperator

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadata
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


@dataclass
class SFAShardQMetadata(ShardQMetadata):
    # CANN consumes contiguous query fragments instead of SparseMLA metadata.
    attn_metadata: "AscendSFAMetadata" = field(init=False)  # type: ignore[assignment]
    # Original batch rows for indexer projection: decode prefix, then local prefill.
    query_indices: torch.Tensor
    # KPool uses the same request fragments with its own block table.
    prefill_request_ids: torch.Tensor
    decode: "AscendSFAMetadata | None"


def build_sfa_shardq_metadata(
    common: "AscendCommonAttentionMetadata", metadata: "AscendSFAMetadata", config: "VllmConfig"
) -> SFAShardQMetadata | None:
    if common.num_actual_tokens == 0:
        return None
    split = build_shardq_split(common, config)
    if split is None:
        return None

    device = metadata.seq_lens.device
    rows = split.local_token_ids + split.num_decode_tokens
    indexer_rows = np.r_[np.arange(split.num_decode_tokens), rows] if split.num_decode_tokens else rows
    query_indices = async_tensor_h2d(indexer_rows, device=device)
    query_starts = common.query_start_loc_cpu[: common.num_reqs + 1].numpy()
    requests = np.searchsorted(query_starts[1:], rows, side="right")
    # Zigzag gives a rank up to two disjoint pieces of a request. Each piece
    # needs its own causal end; joining them would expose future keys.
    boundaries = np.r_[0, np.flatnonzero((np.diff(rows) != 1) | (np.diff(requests) != 0)) + 1, len(rows)]
    request_ids = async_tensor_h2d(requests[boundaries[:-1]], device=device)
    query_start_loc = async_tensor_h2d(boundaries, dtype=common.query_start_loc.dtype, device=device)
    trailing_tokens = query_starts[requests[boundaries[:-1]] + 1] - rows[boundaries[1:] - 1] - 1
    assert metadata.positions is not None
    seq_lens = metadata.seq_lens[request_ids] - async_tensor_h2d(
        trailing_tokens, dtype=metadata.seq_lens.dtype, device=device
    )
    prefill = replace(
        metadata,
        num_input_tokens=len(rows),
        num_actual_tokens=len(rows),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=len(request_ids),
        query_start_loc=query_start_loc,
        cum_query_lens=query_start_loc[1:],
        seq_lens=seq_lens,
        seq_lens_cpu=None,
        block_table=metadata.block_table[request_ids],
        positions=metadata.positions[query_indices[split.num_decode_tokens :]],
        max_query_len=int(np.diff(boundaries).max()),
        sin=None,
        cos=None,
    )
    decode = None
    if split.num_decode_tokens:
        num_decodes = int(np.searchsorted(query_starts, split.num_decode_tokens))
        decode = replace(
            metadata,
            num_input_tokens=split.num_decode_tokens,
            num_actual_tokens=split.num_decode_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=split.num_decode_tokens,
            num_prefills=0,
            query_start_loc=metadata.query_start_loc[: num_decodes + 1],
            cum_query_lens=metadata.cum_query_lens[:num_decodes],
            seq_lens=metadata.seq_lens[:num_decodes],
            block_table=metadata.block_table[:num_decodes],
            positions=metadata.positions[: split.num_decode_tokens],
            max_query_len=int(np.diff(query_starts[: num_decodes + 1]).max()),
        )
    shardq = SFAShardQMetadata(
        num_decode_tokens=split.num_decode_tokens,
        num_tokens_per_rank=split.num_tokens_per_rank.tolist(),
        token_ids=async_tensor_h2d(split.token_ids, device=device),
        rank_offsets=async_tensor_h2d(np.r_[0, np.cumsum(split.num_tokens_per_rank)], device=device),
        query_indices=query_indices,
        prefill_request_ids=request_ids,
        decode=decode,
    )
    shardq.attn_metadata = prefill
    logger.info_once("Ascend SFA is sharding sparse MLA prefill queries across TP ranks (sparse_mla_shard_q).")
    return shardq


def select_shardq_topk(
    q_li: torch.Tensor,
    weights: torch.Tensor,
    indexer: "AscendSFAIndexerBackend",
    metadata: "AscendSFAIndexerMetadata",
    q_li_scale: torch.Tensor | None,
    *,
    shardq: SFAShardQMetadata,
) -> torch.Tensor:
    # Q is already compact and the full K cache has been written.
    num_tokens = weights.shape[0]
    query = q_li.view(num_tokens, indexer.n_head, -1)
    outputs = []
    for view, rows in (
        (shardq.decode, slice(0, shardq.num_decode_tokens)),
        (shardq.attn_metadata, slice(shardq.num_decode_tokens, None)),
    ):
        if view is None:
            continue
        local_query = query[rows]
        local_scale = None
        if q_li_scale is not None:
            local_scale = q_li_scale.view(num_tokens, indexer.n_head, *q_li_scale.shape[2:])[rows]
        local_metadata = replace(
            metadata,
            block_table=view.block_table,
            actual_seq_lengths_query=view.cum_query_lens,
            actual_seq_lengths_key=view.seq_lens,
        )
        outputs.append(
            DeviceOperator.indexer_select_post_process(
                local_query,
                local_scale,
                (local_query.shape[0], indexer.n_head, indexer.head_dim) if indexer.enable_sparse_li_quant else None,
                weights[rows],
                indexer.k_cache.kv_cache,
                INDEXER_K_CACHE_SLOT,
                INDEXER_SCALE_CACHE_SLOT,
                local_metadata,
                view.cum_query_lens,
                view.seq_lens,
                indexer.enable_sparse_li_c8,
                indexer.enable_sparse_li_c4,
                indexer.use_torch_npu_lightning_indexer,
            )
        )
    return torch.cat(outputs, dim=0) if shardq.decode is not None else outputs[0]
