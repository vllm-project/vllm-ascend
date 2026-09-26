# SPDX-License-Identifier: Apache-2.0
"""Canonical reduction groups for short queries and long Prefill."""
from dataclasses import dataclass

import torch

from .kv_boundary_metadata import plan_boundary_queries


@dataclass
class QueryInvariantMetadata:
    block_table: torch.Tensor
    query_ends: list[int]
    kv_lengths: list[int]


def expand_query_metadata(query_ends, kv_lengths, block_table, decode_threshold):
    if block_table.ndim != 2 or block_table.shape[0] < len(query_ends):
        raise ValueError('Unaligned block table')
    # This candidate's caller passes the entire target token count, including
    # Prefill. Keep the API contract explicit instead of silently skipping it.
    if not query_ends or decode_threshold < query_ends[-1]:
        raise ValueError('All target queries must be covered')
    plan = plan_boundary_queries(query_ends, kv_lengths)
    indices = torch.tensor(plan.owners, dtype=torch.long, device=block_table.device)
    return QueryInvariantMetadata(block_table.index_select(0, indices), plan.query_ends, plan.kv_lengths)


def query_invariant_attention(kernel, query, key, value, metadata, *, block_size,
                              num_heads, num_kv_heads, scale, mask):
    output, _ = kernel(query=query, key=key, value=value,
                       block_table=metadata.block_table, block_size=block_size,
                       input_layout='TND', actual_seq_lengths=metadata.query_ends,
                       actual_seq_lengths_kv=metadata.kv_lengths, num_heads=num_heads,
                       num_key_value_heads=num_kv_heads, scale=scale,
                       atten_mask=mask, sparse_mode=3)
    return output
