# SPDX-License-Identifier: Apache-2.0
"""Experimental canonical KV512 grouping for the pinned CANN FAI kernel.

The kernel processes KV in MAX_KV_STACK_LEN=512 chunks. Keep queries together
only when every row ends in the same KV chunk. This avoids an extra all-masked
online-softmax update on earlier rows. Validity still requires NPU regression;
this module alone does not change an installed runtime.
"""
from dataclasses import dataclass


KV_CHUNK = 512
SPECIAL_REDUCTION_STRIDE = 256
MAX_GROUP_WIDTH = 16
MAX_PREFILL_GROUP_WIDTH = 128


@dataclass(frozen=True)
class BoundaryPlan:
    owners: list[int]
    query_ends: list[int]
    kv_lengths: list[int]
    num_actual_tokens: int


def plan_boundary_queries(query_ends, kv_lengths, num_actual_tokens=None):
    if not query_ends or len(query_ends) != len(kv_lengths):
        raise ValueError('Unaligned or empty query/KV metadata')
    if num_actual_tokens is None:
        num_actual_tokens = query_ends[-1]
    if not 0 <= num_actual_tokens <= query_ends[-1] or query_ends[-1] <= 0:
        raise ValueError('Invalid real-token boundary')
    owners, ends, lengths = [], [], []
    previous = 0
    for owner, (end, kv_length) in enumerate(zip(query_ends, kv_lengths)):
        width = end - previous
        if width < 0:
            raise ValueError('Query ends must be ordered')
        if previous >= num_actual_tokens:
            if kv_length != 0:
                raise ValueError('Virtual padding must not read real KV')
            if width:
                owners.append(owner)
                ends.append(end)
                lengths.append(0)
        else:
            if end > num_actual_tokens or not 1 <= width <= kv_length:
                raise ValueError('Invalid causal length or split real/padded request')
            cursor = previous
            prefix = kv_length - width
            # Experimental Prefill extension: retain short Q tiles and the
            # same 256/512 reduction boundaries for every original request.
            group_limit = MAX_GROUP_WIDTH if width <= MAX_GROUP_WIDTH else MAX_PREFILL_GROUP_WIDTH
            while cursor < end:
                # The pinned softmax has special reduction trees for an
                # exact 256- or 512-column tail. Isolate those final rows;
                # the other rows use the sequential TAILTILE reduction.
                tail_room = SPECIAL_REDUCTION_STRIDE - 1 - prefix % SPECIAL_REDUCTION_STRIDE
                take = min(end - cursor, max(1, tail_room), group_limit)
                cursor += take
                prefix += take
                owners.append(owner)
                ends.append(cursor)
                lengths.append(prefix)
        previous = end
    if not ends or ends[-1] != query_ends[-1]:
        raise ValueError('Incomplete query coverage')
    return BoundaryPlan(owners, ends, lengths, num_actual_tokens)
