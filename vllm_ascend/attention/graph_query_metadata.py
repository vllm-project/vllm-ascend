# SPDX-License-Identifier: Apache-2.0
"""Stable Graph metadata for short-query canonical reduction groups."""
from dataclasses import dataclass

import torch

from .kv_boundary_metadata import MAX_GROUP_WIDTH, plan_boundary_queries


@dataclass(frozen=True)
class GraphQueryMetadata:
    block_table: torch.Tensor
    query_ends: list[int]
    kv_lengths: list[int]
    num_actual_tokens: int


def plan_graph_queries(query_ends, kv_lengths, num_actual_tokens, decode_threshold):
    if not 1 <= decode_threshold <= MAX_GROUP_WIDTH:
        raise ValueError('Unsupported Verify width')
    previous = 0
    for end in query_ends:
        if previous < num_actual_tokens and not 1 <= end - previous <= decode_threshold:
            raise ValueError('Invalid real Decode/Verify request')
        previous = end
    return plan_boundary_queries(query_ends, kv_lengths, num_actual_tokens)


class GraphQueryMetadataBuilder:
    """Own one fixed-capacity table per captured query shape.

    Group counts vary with causal lengths. Capacity stays at the captured
    token count; unused table rows have valid owner zero and are never read
    by FIA because the actual sequence lists contain only live groups.
    Build is called before forward/capture on the model stream.
    """

    def __init__(self):
        self._tables = {}

    def build(self, query_ends, kv_lengths, block_table, num_actual_tokens, decode_threshold):
        plan = plan_graph_queries(query_ends, kv_lengths, num_actual_tokens, decode_threshold)
        if block_table.ndim != 2 or block_table.shape[0] < len(query_ends):
            raise ValueError('Unaligned block table')
        capacity = query_ends[-1]
        key = (capacity, block_table.shape[1], block_table.dtype, block_table.device)
        if key not in self._tables:
            self._tables[key] = torch.empty(
                (capacity, block_table.shape[1]), dtype=block_table.dtype, device=block_table.device)
        table = self._tables[key]
        indices = plan.owners + [0] * (capacity - len(plan.owners))
        owners = torch.tensor(indices, dtype=torch.long, device=block_table.device)
        torch.index_select(block_table, 0, owners, out=table)
        return GraphQueryMetadata(table, plan.query_ends, plan.kv_lengths, plan.num_actual_tokens)
