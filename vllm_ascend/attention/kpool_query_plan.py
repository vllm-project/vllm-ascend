# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local full-visible prefixes for pooled indexer queries."""

from dataclasses import dataclass

import torch

MIN_PARTIAL_BYPASS_ROWS = 1024


@dataclass(frozen=True)
class KPoolQueryPlan:
    num_tokens: int
    full_visible_ranges: tuple[tuple[int, int], ...]
    selective_rows: torch.Tensor
    selection_index: torch.Tensor
    cum_query_lens: torch.Tensor


def build_kpool_query_plan(
    common, topk: int | None, device: torch.device, *, min_partial_rows: int = MIN_PARTIAL_BYPASS_ROWS
) -> KPoolQueryPlan | None:
    seq_lens = getattr(common, "seq_lens_cpu", None)
    if topk is None or topk <= 0 or seq_lens is None or not getattr(common, "causal", True):
        return None
    query_starts = common.query_start_loc_cpu
    # A host upper bound only shortens the bypass prefix. Selection still
    # reads the authoritative device lengths; never fetch them to the host.
    if query_starts.device.type != "cpu" or seq_lens.device.type != "cpu":
        return None
    num_reqs, num_tokens = common.num_reqs, common.num_actual_tokens
    starts = query_starts[: num_reqs + 1].tolist()
    lengths = seq_lens[:num_reqs].tolist()
    if len(starts) != num_reqs + 1 or len(lengths) != num_reqs or starts[0] != 0:
        return None
    if any(a < 0 or b < a for a, b in zip(starts, starts[1:])) or starts[-1] < num_tokens:
        return None
    full_ranges, selective_ranges, ends = [], [], []
    selective_count = 0
    for begin, end, length in zip(starts, starts[1:], lengths):
        begin, end = min(begin, num_tokens), min(end, num_tokens)
        query_len = end - begin
        if length < query_len:
            return None
        full_rows = min(query_len, max(0, topk - (length - query_len)))
        if full_rows:
            full_ranges.append((begin, begin + full_rows))
        selective_ranges.append((begin + full_rows, end))
        selective_count += query_len - full_rows
        ends.append(selective_count)
    # Small partial prefixes do not amortize packing. Whole-visible batches
    # avoid both projections and PKI setup, including for short requests.
    if not full_ranges or (selective_count and num_tokens - selective_count < min_partial_rows):
        return None
    selection_index = torch.full((num_tokens,), -1, dtype=torch.int32, device="cpu")
    rows = torch.cat([torch.arange(begin, end, device="cpu") for begin, end in selective_ranges])
    selection_index[rows] = torch.arange(rows.numel(), dtype=torch.int32, device="cpu")
    return KPoolQueryPlan(
        num_tokens,
        tuple(full_ranges),
        rows.to(device, non_blocking=True),
        selection_index.to(device, non_blocking=True),
        torch.tensor(ends, dtype=torch.int64, device=device),
    )
