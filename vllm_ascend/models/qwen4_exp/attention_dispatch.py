# SPDX-License-Identifier: Apache-2.0
"""Request dispatch for Qwen4Exp: bounded FIA prefill and GBSA fallback."""

from dataclasses import dataclass

import torch

from .fia import prepare_qsa_fia_paged_kv, qsa_fia_attention
from .gbsa import qsa_gbsa_attention

QSA_FIA_MAX_LOGICAL_KV_LENGTH = 24 * 1024


@dataclass(frozen=True)
class QSARequest:
    request_index: int
    query_start: int
    query_end: int
    kv_length: int
    is_prefill: bool

    @property
    def use_paged_kv(self) -> bool:
        return self.kv_length != self.query_end - self.query_start

    @property
    def use_fia(self) -> bool:
        """Whether this Prefill chunk starts below the FIA KV limit."""
        return self.is_prefill and self.fia_query_end > self.query_start

    @property
    def fia_query_end(self) -> int:
        """Route the whole chunk to FIA when its cached prefix is below 24K."""
        if not self.is_prefill:
            return self.query_start
        query_length = self.query_end - self.query_start
        kv_before_query = self.kv_length - query_length
        if kv_before_query < QSA_FIA_MAX_LOGICAL_KV_LENGTH:
            return self.query_end
        return self.query_start

    @property
    def gbsa_query_start(self) -> int:
        """First row routed to GBSA; Decode starts at the request boundary."""
        return self.fia_query_end if self.is_prefill else self.query_start


def build_qsa_request_plan(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_actual_tokens: int,
    is_prefilling_cpu: torch.Tensor,
) -> tuple[QSARequest, ...]:
    """Use scheduler phase flags, including one-token prefills and MTP decodes."""
    if any(t.device.type != "cpu" for t in (query_start_loc_cpu, seq_lens_cpu, is_prefilling_cpu)):
        raise ValueError("QSA dispatch requires scheduler CPU metadata")
    starts = query_start_loc_cpu.tolist()
    lengths = seq_lens_cpu.tolist()
    phases = is_prefilling_cpu.tolist()
    if len(lengths) < len(starts) - 1 or len(phases) < len(starts) - 1:
        raise ValueError("QSA phase and KV metadata must cover every request")
    requests = []
    for request_index, (start, end) in enumerate(zip(starts, starts[1:])):
        start = min(max(start, 0), num_actual_tokens)
        end = min(max(end, start), num_actual_tokens)
        if end > start:
            requests.append(
                QSARequest(request_index, start, end, int(lengths[request_index]), bool(phases[request_index]))
            )
    return tuple(requests)


def dispatch_qsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    request_plan: tuple[QSARequest, ...] | None,
) -> torch.Tensor:
    # None is the decode-only metadata sentinel, including graph capture.
    # This branch contains no host length decisions or FIA launches; device KV
    # lengths, positions and page tables may change between graph replays.
    if request_plan is None or not any(request.is_prefill for request in request_plan):
        return qsa_gbsa_attention(
            query,
            key_cache,
            value_cache,
            compact_indices,
            block_table,
            token_to_req,
            logical_positions,
            seq_lens,
            output,
        )

    for request in request_plan:
        if not request.use_fia:
            continue
        rows = slice(request.query_start, request.fia_query_end)
        fia_kv_length = request.kv_length
        visible_lens = torch.minimum(
            logical_positions[rows].add(1).clamp_min(0),
            seq_lens[request.request_index].clamp_max(fia_kv_length),
        )
        if request.use_paged_kv:
            keys, values, table, _ = prepare_qsa_fia_paged_kv(
                key_cache,
                value_cache,
                block_table[request.request_index : request.request_index + 1],
                fia_kv_length,
            )
        else:
            keys, values, table = key[rows], value[rows], None
        qsa_fia_attention(
            query[rows],
            keys,
            values,
            compact_indices[rows],
            visible_lens,
            fia_kv_length,
            output[rows],
            block_table=table,
        )

    # Every Decode row and each Prefill chunk starting at/above the logical-KV
    # limit uses GBSA. Coalesce consecutive rows without gathering/copying K/V or
    # changing the original request IDs used to address block tables and lengths.
    gbsa_spans: list[tuple[int, int]] = []
    for request in request_plan:
        start = request.gbsa_query_start
        if start >= request.query_end:
            continue
        if gbsa_spans and gbsa_spans[-1][1] == start:
            gbsa_spans[-1] = (gbsa_spans[-1][0], request.query_end)
        else:
            gbsa_spans.append((start, request.query_end))
    for start, end in gbsa_spans:
        rows = slice(start, end)
        qsa_gbsa_attention(
            query[rows],
            key_cache,
            value_cache,
            compact_indices[rows],
            block_table,
            token_to_req[rows],
            logical_positions[rows],
            seq_lens,
            output[rows],
        )
    return output
