# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local full-visible prefixes for pooled indexer queries."""

from dataclasses import dataclass

import torch

MIN_PARTIAL_BYPASS_ROWS = 1024
MIN_DENSE_PREFIX_ROWS = 1024
MAX_HYBRID_QUERY_ROWS = 8192
DENSE_KERNEL_BLOCK_SIZE = 128
DENSE_SELECTION_BUDGET = 2048
MAX_MHA_KEY_ROWS = 32768
MAX_MHA_KEY_LENGTH = 16384
MIN_MHA_QUERY_ROWS = 1024
MAX_MHA_KV_QUERY_RATIO = 2
MAX_MHA_MASK_ELEMENTS = 128 * 1024 * 1024


@dataclass(frozen=True)
class KPoolMHAPlan:
    num_tokens: int
    query_lengths: tuple[int, ...]
    query_ends: tuple[int, ...]
    key_lengths: tuple[int, ...]
    key_ends: tuple[int, ...]
    max_query_len: int
    max_key_len: int
    visible_key_lengths: torch.Tensor


def build_kpool_mha_plan(common, device: torch.device) -> KPoolMHAPlan | None:
    lengths = getattr(common, "seq_lens_cpu_upper_bound", None)
    prefilling = getattr(common, "is_prefilling", None)
    starts = common.query_start_loc_cpu
    if (
        lengths is None
        or prefilling is None
        or starts.device.type != "cpu"
        or lengths.device.type != "cpu"
        or prefilling.device.type != "cpu"
        or getattr(common, "causal", True) is not True
        or getattr(common, "mm_req_doc_ranges", None)
        or prefilling.numel() < common.num_reqs
        or not prefilling[: common.num_reqs].all()
    ):
        return None
    starts = starts[: common.num_reqs + 1].tolist()
    lengths = lengths[: common.num_reqs].tolist()
    if (
        len(starts) != common.num_reqs + 1
        or len(lengths) != common.num_reqs
        or not lengths
        or starts[0] != 0
        or starts[-1] != common.num_actual_tokens
        or common.num_actual_tokens > MAX_HYBRID_QUERY_ROWS
    ):
        return None
    queries = [end - begin for begin, end in zip(starts, starts[1:])]
    max_query, max_key = max(queries), max(lengths)
    if (
        min(queries) <= 0
        or max_query < MIN_MHA_QUERY_ROWS
        or any(length < query for query, length in zip(queries, lengths))
        or sum(lengths) > MAX_MHA_KV_QUERY_RATIO * common.num_actual_tokens
        or max_key > MAX_MHA_KEY_LENGTH
        or sum(lengths) > MAX_MHA_KEY_ROWS
        or common.num_reqs * max_query * max_key > MAX_MHA_MASK_ELEMENTS
    ):
        return None
    visible: list[torch.Tensor] = []
    key_ends: list[int] = []
    for request, (query, length) in enumerate(zip(queries, lengths)):
        visible.append(torch.arange(length - query + 1, length + 1, dtype=torch.int32, device="cpu"))
        key_ends.append((key_ends[-1] if key_ends else 0) + length)
    return KPoolMHAPlan(
        common.num_actual_tokens,
        tuple(queries),
        tuple(starts[1:]),
        tuple(lengths),
        tuple(key_ends),
        max_query,
        max_key,
        _copy_plan_tensor(torch.cat(visible), device),
    )


@dataclass(frozen=True)
class KPoolDensePlan:
    request_index: int
    query_rows: torch.Tensor
    sparse_rows: torch.Tensor
    sparse_cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    seq_len: int


@dataclass(frozen=True)
class KPoolQueryPlan:
    num_tokens: int
    full_visible_ranges: tuple[tuple[int, int], ...]
    selective_rows: torch.Tensor
    selection_index: torch.Tensor
    cum_query_lens: torch.Tensor
    dense: KPoolDensePlan | None = None


def _copy_plan_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Pageable host storage synchronizes NPU copies even with non_blocking=True.
    if device.type != "cpu" and tensor.numel():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=True)


def build_kpool_query_plan(
    common,
    topk: int | None,
    device: torch.device,
    *,
    min_partial_rows: int = MIN_PARTIAL_BYPASS_ROWS,
    plan_dense_attention: bool = False,
) -> KPoolQueryPlan | None:
    seq_lens = getattr(common, "seq_lens_cpu_upper_bound", None)
    if seq_lens is None:
        # Do not invoke upstream's deprecated property, which can copy device lengths.
        seq_lens = getattr(common, "__dict__", {}).get("seq_lens_cpu")
    causal = getattr(common, "causal", True)
    if (
        topk is None
        or topk <= 0
        or seq_lens is None
        or not isinstance(causal, bool)
        or not causal
        or getattr(common, "mm_req_doc_ranges", None)
    ):
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
        _copy_plan_tensor(rows, device),
        _copy_plan_tensor(selection_index, device),
        _copy_plan_tensor(torch.tensor(ends, dtype=torch.int64, device="cpu"), device),
        _build_dense_plan(common, starts, lengths, topk, device) if plan_dense_attention else None,
    )


def _build_dense_plan(common, starts, lengths, topk: int, device: torch.device) -> KPoolDensePlan | None:
    prefilling = getattr(common, "is_prefilling", None)
    if (
        topk != DENSE_SELECTION_BUDGET
        or common.num_actual_tokens > MAX_HYBRID_QUERY_ROWS
        or prefilling is None
        or prefilling.device.type != "cpu"
        or prefilling.numel() < common.num_reqs
    ):
        return None
    candidates = []
    for request, (begin, end, length, is_prefill) in enumerate(zip(starts, starts[1:], lengths, prefilling.tolist())):
        end = min(end, common.num_actual_tokens)
        context = length - (end - begin)
        count = min(end - begin, max(0, topk - context))
        # Scheduler bounds are exact for prefill requests; async-spec decode
        # bounds cannot supply FIA's host-side causal lengths.
        # C8 admission is measured with a full selection-budget KV prefix.
        if is_prefill and count >= MIN_DENSE_PREFIX_ROWS and context + count == topk:
            candidates.append((request, begin, count, context + count))
    # Admission is limited to the single-prefix shapes measured on Ascend.
    if len(candidates) != 1:
        return None
    request, begin, count, seq_len = candidates[0]
    sparse_rows = torch.cat(
        [torch.arange(begin, device="cpu"), torch.arange(begin + count, common.num_actual_tokens, device="cpu")]
    )
    sparse_ends = [
        min(end, common.num_actual_tokens) - (count if i >= request else 0) for i, end in enumerate(starts[1:])
    ]
    pages = (seq_len + DENSE_KERNEL_BLOCK_SIZE - 1) // DENSE_KERNEL_BLOCK_SIZE
    return KPoolDensePlan(
        request,
        _copy_plan_tensor(torch.arange(begin, begin + count, device="cpu"), device),
        _copy_plan_tensor(sparse_rows, device),
        _copy_plan_tensor(torch.tensor(sparse_ends, dtype=torch.int32, device="cpu"), device),
        torch.arange(pages, dtype=torch.int32, device=device).unsqueeze(0),
        seq_len,
    )
