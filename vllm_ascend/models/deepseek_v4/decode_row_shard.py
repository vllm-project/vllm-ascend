# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Host-only plans for whole-request, round-robin decode indexer sharding."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeIndexerRowPlan:
    num_requests: int
    rows_per_request: int
    tp_size: int
    tp_rank: int
    request_ids: tuple[int, ...]
    query_rows: tuple[int, ...]
    restore_rows: tuple[int, ...]

    @property
    def max_local_rows(self) -> int:
        return ((self.num_requests + self.tp_size - 1) // self.tp_size) * self.rows_per_request


def make_decode_indexer_row_plan(
    num_requests: int,
    rows_per_request: int,
    tp_size: int,
    tp_rank: int,
) -> DecodeIndexerRowPlan:
    """Assign requests r, r+TP, ... and invert padded rank-order AllGather.

    Eager passes the actual request count. Graphs pass the padded request
    bucket and zero the lengths of inactive requests in separate static
    buffers. Additional AllGather padding never becomes a cache write.
    """
    if tp_size < 2 or num_requests < tp_size:
        raise ValueError("Require TP > 1 and at least one request on every TP rank")
    if rows_per_request < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid rows_per_request or TP rank")
    request_ids = tuple(range(tp_rank, num_requests, tp_size))
    query_rows = tuple(req * rows_per_request + row for req in request_ids for row in range(rows_per_request))
    max_local_rows = ((num_requests + tp_size - 1) // tp_size) * rows_per_request
    restore_rows = tuple(
        (req % tp_size) * max_local_rows + (req // tp_size) * rows_per_request + row
        for req in range(num_requests)
        for row in range(rows_per_request)
    )
    return DecodeIndexerRowPlan(
        num_requests=num_requests,
        rows_per_request=rows_per_request,
        tp_size=tp_size,
        tp_rank=tp_rank,
        request_ids=request_ids,
        query_rows=query_rows,
        restore_rows=restore_rows,
    )


def make_graph_decode_indexer_row_plan(
    num_tokens: int,
    rows_per_request: int,
    tp_size: int,
    tp_rank: int,
    min_requests: int,
) -> DecodeIndexerRowPlan | None:
    """Choose solely from the capture bucket, never the live request count."""
    if num_tokens < 1 or rows_per_request < 1 or num_tokens % rows_per_request:
        raise ValueError("Graph token bucket must contain complete uniform requests")
    num_requests = num_tokens // rows_per_request
    if tp_size < 2 or num_requests < max(min_requests, tp_size):
        return None
    return make_decode_indexer_row_plan(num_requests, rows_per_request, tp_size, tp_rank)
