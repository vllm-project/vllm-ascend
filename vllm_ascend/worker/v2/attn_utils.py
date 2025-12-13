# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any

import torch
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         AscendPrefillContextParallelMetadata)


def build_attn_metadata(
    attn_metadata_builders: list[AttentionMetadataBuilder],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_computed_tokens_cpu: torch.Tensor,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    decode_token_per_req: int,
    actual_seq_lengths_q: list[int],
    positions: torch.Tensor | None = None,
    attn_mask: torch.Tensor
    | None = None,
    spec_attn_mask: torch.Tensor | None = None,
    attn_state: Any | None = None,
    is_only_prefill: bool = False,
    graph_pad_size: int = -1,
    num_input_tokens: int = 0,
    prefill_context_parallel_metadata: AscendPrefillContextParallelMetadata
    | None = None,
) -> dict[str, Any]:
    max_query_len = int(query_start_loc_cpu.max())

    attn_metadata: dict[str, Any] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for i, kv_cache_spec in enumerate(kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu[:num_reqs],
            seq_lens=seq_lens[:num_reqs],
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            decode_token_per_req=decode_token_per_req,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            actual_seq_lengths_q=actual_seq_lengths_q,
            positions=positions,
            attn_mask=attn_mask,
            spec_attn_mask=spec_attn_mask,
            attn_state=attn_state,
            is_only_prefill=is_only_prefill,
            graph_pad_size=graph_pad_size,
            num_input_tokens=num_input_tokens,
            prefill_context_parallel_metadata=prefill_context_parallel_metadata,
        )

        attn_metadata_builder = attn_metadata_builders[i]
        metadata = attn_metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        for layer_name in kv_cache_spec.layer_names:
            attn_metadata[layer_name] = metadata
    return attn_metadata
