from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CompressorSPPlan:
    enabled: bool
    reason: str
    ratio: int = 0
    path: str = ""
    coff: int = 0
    cache_mode: int = 1
    is_chunked_prefill: bool = False
    state_block_table_rows: int = 0
    seq_lens: tuple[int, ...] = ()
    query_lens: tuple[int, ...] = ()
    request_start_positions: tuple[int, ...] = ()
    history_start_positions: tuple[int, ...] = ()
    start_pos_zero: bool = True
    seq_len_aligned: bool = True
    requires_history_state: bool = False
    requires_tail_state_update: bool = False
    requires_boundary_state_sync: bool = False
    global_compressed_row_count: int = 0
    boundary_req_indices: tuple[int, ...] = ()
    boundary_positions: tuple[int, ...] = ()
    boundary_owner_mask: tuple[bool, ...] = ()
    supports_boundary_state_replay: bool = False
    boundary_replay_token_ranges: tuple[tuple[int, int], ...] = ()
    boundary_replay_token_indices: tuple[int, ...] = ()
    boundary_replay_token_slice: tuple[int, int] | None = None
    boundary_replay_req_indices: tuple[int, ...] = ()
    boundary_replay_req_slice: tuple[int, int] | None = None
    boundary_replay_cu_seqlens: tuple[int, ...] = ()
    boundary_replay_start_pos: tuple[int, ...] = ()
    boundary_replay_compressed_row_indices: tuple[int, ...] = ()
    boundary_replay_compressed_row_slice: tuple[int, int] | None = None
    boundary_replay_rope_row_indices: tuple[int, ...] = ()
    boundary_replay_rope_row_slice: tuple[int, int] | None = None
    token_indices: tuple[int, ...] = ()
    token_slice: tuple[int, int] | None = None
    req_indices: tuple[int, ...] = ()
    req_slice: tuple[int, int] | None = None
    cu_seqlens: tuple[int, ...] = ()
    start_pos: tuple[int, ...] = ()
    compressed_row_indices: tuple[int, ...] = ()
    compressed_row_slice: tuple[int, int] | None = None
    rope_row_indices: tuple[int, ...] = ()
    rope_row_slice: tuple[int, int] | None = None
    valid_row_indices: tuple[int, ...] = ()
    valid_row_slice: tuple[int, int] | None = None
    output_keep_indices: tuple[int, ...] = ()
    output_keep_slice: tuple[int, int] | None = None
    slot_mapping_indices: tuple[int, ...] = ()
    slot_mapping_slice: tuple[int, int] | None = None
    local_keep_to_full_row_indices: tuple[int, ...] = ()
    local_keep_to_slot_row_indices: tuple[int, ...] = ()
    tail_token_ranges: tuple[tuple[int, int], ...] = ()
    padding_row_indices: tuple[int, ...] = ()
    padding_row_slice: tuple[int, int] | None = None
    # Selects valid rows from the rank-major padded all-gather buffer. The
    # flattened source row for rank r/local row i is r * max_rows + i.
    gather_compact_indices: tuple[int, ...] = ()
    gather_compact_slice: tuple[int, int] | None = None
    # Number of rank-owned compressed output rows for every rank in the TP
    # group, in rank order.  Because each rank owns a contiguous slice of the
    # flattened token stream, these counts also describe how the global
    # compressed rows partition across ranks, so an all-gather that
    # concatenates rank outputs in rank order reproduces the global row order.
    sp_row_counts_per_rank: tuple[int, ...] = ()
    tp_rank: int = 0
    tp_size: int = 1


def all_ranks_have_compressor_sp_rows(
    row_counts: Sequence[int],
) -> bool:
    """Whether every TP rank can contribute a non-empty SP output row.

    Non-replay paths conservatively require this before entering a compressed-row
    collective. C4 boundary replay is the sole caller allowed to contribute an
    empty padded tensor.
    """
    return bool(row_counts) and all(count > 0 for count in row_counts)


def _contiguous_slice(values: Sequence[int]) -> tuple[int, int] | None:
    if not values:
        return (0, 0)
    start = values[0]
    for offset, value in enumerate(values):
        if value != start + offset:
            return None
    return (start, len(values))


def _rope_row_selector(compressed_rows: Sequence[int], target_rows: int) -> tuple[int, ...]:
    """Map Compressor RoPE rows to the full RoPE table.

    Full-path metadata fills non-output RoPE rows with position 0. Encoding
    those rows as repeated zero indices lets runtime use one indexed selection
    instead of select + expand + concat.
    """
    if target_rows < len(compressed_rows):
        raise ValueError(
            f"CompressorSP RoPE target has fewer rows than compressed rows: {target_rows} < {len(compressed_rows)}"
        )
    return tuple(compressed_rows) + (0,) * (target_rows - len(compressed_rows))


def _in_ranges(idx: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= idx < end for start, end in ranges)


def _build_c4_boundary_replay(
    *,
    input_positions: Sequence[int],
    query_start_loc: Sequence[int],
    query_lens: Sequence[int],
    start_positions: Sequence[int],
) -> dict[str, Any]:
    flat_to_compressed_row: dict[int, int] = {}
    compressed_row = 0
    for flat_idx, position in enumerate(input_positions):
        if (position + 1) % 4 == 0:
            flat_to_compressed_row[flat_idx] = compressed_row
            compressed_row += 1

    token_ranges: list[tuple[int, int]] = []
    token_indices: list[int] = []
    req_indices: list[int] = []
    cu_seqlens = [0]
    replay_start_pos: list[int] = []
    compressed_row_indices: list[int] = []

    for req_idx, query_len in enumerate(query_lens):
        if query_len <= 0:
            continue

        req_start = query_start_loc[req_idx]
        req_end = query_start_loc[req_idx + 1]
        replay_tokens = 8 + (4 if start_positions[req_idx] > 0 else 0)
        replay_start = max(req_start, req_end - replay_tokens)
        while replay_start > req_start and input_positions[replay_start] % 8 != 0:
            replay_start -= 1

        token_ranges.append((replay_start, req_end))
        token_indices.extend(range(replay_start, req_end))
        req_indices.append(req_idx)
        cu_seqlens.append(cu_seqlens[-1] + req_end - replay_start)
        replay_start_pos.append(start_positions[req_idx] + replay_start - req_start)
        compressed_row_indices.extend(
            flat_to_compressed_row[flat_idx]
            for flat_idx in range(replay_start, req_end)
            if flat_idx in flat_to_compressed_row
        )

    target_rope_rows = min(len(token_indices), len(token_indices) // 4 + len(req_indices))
    rope_row_indices = _rope_row_selector(compressed_row_indices, target_rope_rows)

    return {
        "supports_boundary_state_replay": bool(token_indices),
        "boundary_replay_token_ranges": tuple(token_ranges),
        "boundary_replay_token_indices": tuple(token_indices),
        "boundary_replay_token_slice": _contiguous_slice(token_indices),
        "boundary_replay_req_indices": tuple(req_indices),
        "boundary_replay_req_slice": _contiguous_slice(req_indices),
        "boundary_replay_cu_seqlens": tuple(cu_seqlens),
        "boundary_replay_start_pos": tuple(replay_start_pos),
        "boundary_replay_compressed_row_indices": tuple(compressed_row_indices),
        "boundary_replay_compressed_row_slice": _contiguous_slice(compressed_row_indices),
        "boundary_replay_rope_row_indices": rope_row_indices,
        "boundary_replay_rope_row_slice": _contiguous_slice(rope_row_indices),
    }


def collect_state_row_indices(
    *,
    token_positions: Sequence[int] | Any,
    req_block_table: Sequence[Sequence[int]] | Any,
    cu_seqlens: Sequence[int] | Any,
    state_block_size: int,
) -> tuple[int, ...]:
    if state_block_size <= 0:
        return ()

    if hasattr(token_positions, "tolist"):
        token_positions = token_positions.tolist()
    else:
        token_positions = list(token_positions)
    if hasattr(req_block_table, "tolist"):
        req_block_table = req_block_table.tolist()
    else:
        req_block_table = list(req_block_table)
    if hasattr(cu_seqlens, "tolist"):
        cu_seqlens = cu_seqlens.tolist()
    else:
        cu_seqlens = list(cu_seqlens)

    if not token_positions or not req_block_table or len(cu_seqlens) < 2:
        return ()

    rows: set[int] = set()
    req_count = min(len(req_block_table), len(cu_seqlens) - 1)
    for req_idx in range(req_count):
        start = int(cu_seqlens[req_idx])
        end = int(cu_seqlens[req_idx + 1])
        if start >= end:
            continue

        req_rows = req_block_table[req_idx]
        if hasattr(req_rows, "tolist"):
            req_rows = req_rows.tolist()
        else:
            req_rows = list(req_rows)
        if not req_rows:
            continue

        for pos in token_positions[start:end]:
            if pos < 0:
                continue
            block_idx = int(pos) // state_block_size
            if 0 <= block_idx < len(req_rows):
                row_id = int(req_rows[block_idx])
                # Physical block IDs are zero-based. Negative entries are
                # reserved for invalid/padded block-table slots.
                if row_id >= 0:
                    rows.add(row_id)

    return tuple(sorted(rows))


def collect_boundary_state_row_indices(
    *,
    boundary_positions: Sequence[int] | Any,
    req_block_table: Sequence[Sequence[int]] | Any,
    state_block_size: int,
) -> tuple[int, ...]:
    if state_block_size <= 0:
        return ()

    if hasattr(boundary_positions, "tolist"):
        boundary_positions = boundary_positions.tolist()
    else:
        boundary_positions = list(boundary_positions)
    if hasattr(req_block_table, "tolist"):
        req_block_table = req_block_table.tolist()
    else:
        req_block_table = list(req_block_table)

    if not boundary_positions or not req_block_table:
        return ()

    rows: set[int] = set()
    req_count = min(len(boundary_positions), len(req_block_table))
    for req_idx in range(req_count):
        boundary_pos = int(boundary_positions[req_idx])
        if boundary_pos < 0:
            continue

        req_rows = req_block_table[req_idx]
        if hasattr(req_rows, "tolist"):
            req_rows = req_rows.tolist()
        else:
            req_rows = list(req_rows)
        if not req_rows:
            continue

        block_idx = boundary_pos // state_block_size
        if 0 <= block_idx < len(req_rows):
            row_id = int(req_rows[block_idx])
            if row_id >= 0:
                rows.add(row_id)

    return tuple(sorted(rows))


def sync_boundary_state_blocks(
    *,
    state_cache: Any,
    state_block_table: Any,
    boundary_req_indices: Any,
    boundary_positions: Any,
    boundary_owner_mask: Any,
    all_reduce: Any,
) -> Any:
    import torch

    boundary_req_indices = boundary_req_indices.long()
    boundary_positions = boundary_positions.long()
    boundary_owner_mask = boundary_owner_mask.bool()
    if boundary_req_indices.numel() == 0:
        return boundary_req_indices
    if not (boundary_req_indices.shape == boundary_positions.shape == boundary_owner_mask.shape):
        raise RuntimeError("Chunked Compressor SP boundary-state metadata shape mismatch")

    state_block_size = int(state_cache.shape[1])
    boundary_block_indices = torch.div(
        boundary_positions,
        state_block_size,
        rounding_mode="floor",
    )
    request_block_tables = state_block_table.index_select(0, boundary_req_indices)
    boundary_state_rows = request_block_tables.gather(1, boundary_block_indices.unsqueeze(1)).squeeze(1).long()

    boundary_state = state_cache.index_select(0, boundary_state_rows)
    owner_shape = (boundary_owner_mask.shape[0],) + (1,) * (boundary_state.dim() - 1)
    owner_state = boundary_state * boundary_owner_mask.view(owner_shape).to(boundary_state.dtype)
    synchronized_state = all_reduce(owner_state)
    state_cache.index_copy_(0, boundary_state_rows, synchronized_state)
    return boundary_state_rows


def build_compressor_sp_plan(
    *,
    enabled: bool,
    has_prefill: bool,
    need_gather_q_kv: bool,
    tp_size: int,
    compress_ratio: int,
    path: str = "",
    coff: int = 0,
    cache_mode: int = 1,
    is_chunked_prefill: bool = False,
    state_block_table_rows: int = 0,
    allow_c4_non_aligned: bool = False,
    allow_c128_non_aligned: bool = False,
    input_positions: Sequence[int],
    query_start_loc: Sequence[int],
    seq_lens: Sequence[int],
    local_start: int,
    local_end: int,
    tp_rank: int = 0,
) -> CompressorSPPlan:
    def disabled(reason: str, **kwargs: Any) -> CompressorSPPlan:
        return CompressorSPPlan(
            enabled=False,
            reason=reason,
            ratio=compress_ratio,
            path=path,
            coff=coff,
            cache_mode=cache_mode,
            is_chunked_prefill=is_chunked_prefill,
            state_block_table_rows=state_block_table_rows,
            **kwargs,
        )

    if not enabled:
        return disabled("env_disabled")
    if not has_prefill:
        return disabled("not_prefill")
    if not need_gather_q_kv or tp_size <= 1:
        return disabled("no_need_gather")
    if compress_ratio not in (4, 128):
        return disabled("unsupported_ratio")
    if len(query_start_loc) < 2:
        return disabled("empty_query_start_loc")

    num_input_tokens = len(input_positions)
    if any(query_start_loc[i] > query_start_loc[i + 1] for i in range(len(query_start_loc) - 1)):
        return disabled("invalid_query_start_loc")
    if query_start_loc[0] < 0 or query_start_loc[-1] > num_input_tokens:
        return disabled("query_start_loc_out_of_bounds")

    owned_start = max(0, local_start)
    owned_end = min(max(owned_start, local_end), num_input_tokens)

    num_reqs = len(query_start_loc) - 1
    if len(seq_lens) < num_reqs:
        return disabled("seq_lens_too_short")

    query_lens = [query_start_loc[i + 1] - query_start_loc[i] for i in range(num_reqs)]
    start_positions = [seq_lens[i] - query_lens[i] for i in range(num_reqs)]
    boundary_req_indices = [req_idx for req_idx, query_len in enumerate(query_lens) if query_len > 0]
    boundary_positions = [seq_lens[req_idx] - 1 for req_idx in boundary_req_indices]
    boundary_owner_mask = [
        owned_start <= query_start_loc[req_idx + 1] - 1 < owned_end for req_idx in boundary_req_indices
    ]
    global_compressed_row_count = sum((position + 1) % compress_ratio == 0 for position in input_positions)
    seq_len_aligned = all(seq_len % compress_ratio == 0 for seq_len in seq_lens[:num_reqs])
    boundary_replay_fields: dict[str, Any] = {}
    if is_chunked_prefill and compress_ratio == 4 and boundary_req_indices:
        boundary_replay_fields = _build_c4_boundary_replay(
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            start_positions=start_positions,
        )
    base_plan_fields = dict(
        seq_lens=tuple(seq_lens[:num_reqs]),
        query_lens=tuple(query_lens),
        request_start_positions=tuple(start_positions),
        history_start_positions=tuple(start_positions),
        start_pos_zero=all(pos == 0 for pos in start_positions),
        seq_len_aligned=seq_len_aligned,
        requires_history_state=any(pos > 0 for pos in start_positions),
        requires_tail_state_update=not seq_len_aligned,
        requires_boundary_state_sync=(
            is_chunked_prefill and bool(boundary_req_indices) and global_compressed_row_count > 0
        ),
        global_compressed_row_count=global_compressed_row_count,
        boundary_req_indices=tuple(boundary_req_indices),
        boundary_positions=tuple(boundary_positions),
        boundary_owner_mask=tuple(boundary_owner_mask),
        **boundary_replay_fields,
    )
    history_pad = compress_ratio if compress_ratio == 4 and any(pos > 0 for pos in start_positions) else 0
    can_replay_boundary_without_local_rows = bool(
        compress_ratio == 4
        and base_plan_fields["requires_boundary_state_sync"]
        and base_plan_fields.get("supports_boundary_state_replay", False)
    )
    if owned_start >= owned_end and not can_replay_boundary_without_local_rows:
        return disabled("no_local_tokens", **base_plan_fields)

    for req_idx, query_len in enumerate(query_lens):
        if query_len < 0 or start_positions[req_idx] < 0:
            return disabled("invalid_lengths", **base_plan_fields)
        if query_len == 0:
            continue
        if seq_lens[req_idx] % compress_ratio != 0:
            reason = "seq_len_not_aligned_c4" if compress_ratio == 4 else "seq_len_not_aligned_c128"
            allow_non_aligned = allow_c4_non_aligned if compress_ratio == 4 else allow_c128_non_aligned
            if not allow_non_aligned:
                return disabled(reason, **base_plan_fields)
        if compress_ratio != 4 and start_positions[req_idx] % compress_ratio != 0 and not allow_c128_non_aligned:
            return disabled("unaligned_start_pos", **base_plan_fields)

    true_ranges: list[tuple[int, int]] = []
    expanded_ranges: list[tuple[int, int]] = []
    token_indices: list[int] = []
    req_indices: list[int] = []
    cu_seqlens = [0]
    start_pos: list[int] = []
    tail_token_ranges: list[tuple[int, int]] = []

    for req_idx in range(num_reqs):
        req_start = query_start_loc[req_idx]
        req_end = query_start_loc[req_idx + 1]
        true_start = max(req_start, owned_start)
        true_end = min(req_end, owned_end)
        if true_start >= true_end:
            continue

        first_output_idx = None
        for flat_idx in range(true_start, true_end):
            if (input_positions[flat_idx] + 1) % compress_ratio == 0:
                first_output_idx = flat_idx
                break
        if first_output_idx is None:
            continue

        output_position = input_positions[first_output_idx]
        dependency_tokens = 2 * compress_ratio if compress_ratio == 4 else compress_ratio
        expanded_start = max(req_start, first_output_idx - dependency_tokens + 1 - history_pad)
        expected_position = max(0, output_position - dependency_tokens + 1 - history_pad)
        while expanded_start < first_output_idx and input_positions[expanded_start] < expected_position:
            expanded_start += 1
        if compress_ratio == 4:
            while expanded_start > req_start and input_positions[expanded_start] % 8 != 0:
                expanded_start -= 1
        expanded_end = true_end
        if expanded_end < req_end:
            # A rank-local non-aligned right edge would be interpreted by
            # Compressor as a request tail and would write a false partial
            # state. C4 state rows span 8 tokens, while every C128 group is
            # already aligned to its 32-token state rows.
            right_alignment = 8 if compress_ratio == 4 else compress_ratio
            while expanded_end < req_end and (input_positions[expanded_end - 1] + 1) % right_alignment != 0:
                expanded_end += 1
        if expanded_start >= expanded_end:
            continue

        true_ranges.append((true_start, true_end))
        expanded_ranges.append((expanded_start, expanded_end))
        req_indices.append(req_idx)
        token_indices.extend(range(expanded_start, expanded_end))
        cu_seqlens.append(cu_seqlens[-1] + expanded_end - expanded_start)
        start_pos.append(start_positions[req_idx] + expanded_start - req_start)
        if seq_lens[req_idx] % compress_ratio != 0:
            tail_len = seq_lens[req_idx] % compress_ratio
            tail_start = max(req_start, req_end - tail_len)
            tail_range = (tail_start, req_end)
            tail_token_ranges.append(tail_range)

            state_tail_start = tail_start
            if compress_ratio == 4:
                req_history_pad = compress_ratio if start_positions[req_idx] > 0 else 0
                state_tail_start = max(
                    req_start,
                    tail_start - 2 * compress_ratio - req_history_pad,
                )
                while state_tail_start > req_start and input_positions[state_tail_start] % 8 != 0:
                    state_tail_start -= 1
            state_tail_range = (state_tail_start, req_end)

            tail_is_covered = any(
                range_start <= state_tail_start and req_end <= range_end for range_start, range_end in expanded_ranges
            )
            if not tail_is_covered:
                # Every rank may own the first group of the next chunk. Add
                # the real request tail as a state-only synthetic request so
                # each rank carries identical partial history state. Its
                # padding output is never included in output_keep_indices.
                expanded_ranges.append(state_tail_range)
                req_indices.append(req_idx)
                token_indices.extend(range(state_tail_start, req_end))
                cu_seqlens.append(cu_seqlens[-1] + req_end - state_tail_start)
                start_pos.append(start_positions[req_idx] + state_tail_start - req_start)

    if not token_indices and not can_replay_boundary_without_local_rows:
        return disabled("no_local_compressed_rows", **base_plan_fields)

    compressed_row_indices: list[int] = []
    output_keep_indices: list[int] = []
    slot_mapping_indices: list[int] = []

    global_compressed_row = 0
    for flat_idx, pos in enumerate(input_positions):
        if (pos + 1) % compress_ratio != 0:
            continue

        in_expanded = _in_ranges(flat_idx, expanded_ranges)
        in_true = _in_ranges(flat_idx, true_ranges)
        if in_expanded:
            local_row = len(compressed_row_indices)
            compressed_row_indices.append(global_compressed_row)
            if in_true:
                output_keep_indices.append(local_row)
        if in_true:
            slot_mapping_indices.append(global_compressed_row)
        global_compressed_row += 1

    if not output_keep_indices and not can_replay_boundary_without_local_rows:
        reason = "tail_state_update_unverified" if tail_token_ranges else "no_local_compressed_rows"
        return disabled(reason, **base_plan_fields, tail_token_ranges=tuple(tail_token_ranges))
    if len(output_keep_indices) != len(slot_mapping_indices):
        return disabled(
            "slot_mapping_mismatch",
            **base_plan_fields,
            token_indices=tuple(token_indices),
            token_slice=_contiguous_slice(token_indices),
            req_indices=tuple(req_indices),
            req_slice=_contiguous_slice(req_indices),
            cu_seqlens=tuple(cu_seqlens),
            start_pos=tuple(start_pos),
            compressed_row_indices=tuple(compressed_row_indices),
            compressed_row_slice=_contiguous_slice(compressed_row_indices),
            output_keep_indices=tuple(output_keep_indices),
            output_keep_slice=_contiguous_slice(output_keep_indices),
            slot_mapping_indices=tuple(slot_mapping_indices),
            slot_mapping_slice=_contiguous_slice(slot_mapping_indices),
            local_keep_to_full_row_indices=tuple(slot_mapping_indices),
            local_keep_to_slot_row_indices=tuple(slot_mapping_indices),
            tail_token_ranges=tuple(tail_token_ranges),
        )

    local_output_rows = min(len(token_indices), len(token_indices) // compress_ratio + len(req_indices))
    padding_row_indices = list(range(len(compressed_row_indices), local_output_rows))
    rope_row_indices = _rope_row_selector(compressed_row_indices, local_output_rows)

    # Compute how many compressed output rows each rank owns.  Each rank's
    # owned token range is [rank*tokens_per_rank, min((rank+1)*tokens_per_rank,
    # num_input_tokens)), and a compressed row is owned by the rank whose
    # true-owned range contains the flat token index that produced it.
    num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    sp_row_counts: list[int] = [0] * tp_size
    for flat_idx, pos in enumerate(input_positions):
        if (pos + 1) % compress_ratio != 0:
            continue
        owning_rank = min(flat_idx // tokens_per_rank, tp_size - 1)
        sp_row_counts[owning_rank] += 1

    max_rows_per_rank = max(sp_row_counts, default=0)
    gather_compact_indices = tuple(
        rank_idx * max_rows_per_rank + row_idx
        for rank_idx, count in enumerate(sp_row_counts)
        for row_idx in range(count)
    )

    return CompressorSPPlan(
        enabled=True,
        reason="enabled",
        ratio=compress_ratio,
        path=path,
        coff=coff,
        cache_mode=cache_mode,
        is_chunked_prefill=is_chunked_prefill,
        state_block_table_rows=state_block_table_rows,
        seq_lens=base_plan_fields["seq_lens"],
        query_lens=base_plan_fields["query_lens"],
        request_start_positions=base_plan_fields["request_start_positions"],
        history_start_positions=base_plan_fields["history_start_positions"],
        start_pos_zero=base_plan_fields["start_pos_zero"],
        seq_len_aligned=seq_len_aligned,
        requires_history_state=base_plan_fields["requires_history_state"],
        # The tail is request-global.  It may belong to another rank, so a
        # rank-local tail range must not clear the global state-update guard.
        requires_tail_state_update=base_plan_fields["requires_tail_state_update"],
        requires_boundary_state_sync=base_plan_fields["requires_boundary_state_sync"],
        global_compressed_row_count=base_plan_fields["global_compressed_row_count"],
        boundary_req_indices=base_plan_fields["boundary_req_indices"],
        boundary_positions=base_plan_fields["boundary_positions"],
        boundary_owner_mask=base_plan_fields["boundary_owner_mask"],
        supports_boundary_state_replay=base_plan_fields.get("supports_boundary_state_replay", False),
        boundary_replay_token_ranges=base_plan_fields.get("boundary_replay_token_ranges", ()),
        boundary_replay_token_indices=base_plan_fields.get("boundary_replay_token_indices", ()),
        boundary_replay_token_slice=base_plan_fields.get("boundary_replay_token_slice"),
        boundary_replay_req_indices=base_plan_fields.get("boundary_replay_req_indices", ()),
        boundary_replay_req_slice=base_plan_fields.get("boundary_replay_req_slice"),
        boundary_replay_cu_seqlens=base_plan_fields.get("boundary_replay_cu_seqlens", ()),
        boundary_replay_start_pos=base_plan_fields.get("boundary_replay_start_pos", ()),
        boundary_replay_compressed_row_indices=base_plan_fields.get("boundary_replay_compressed_row_indices", ()),
        boundary_replay_compressed_row_slice=base_plan_fields.get("boundary_replay_compressed_row_slice"),
        boundary_replay_rope_row_indices=base_plan_fields.get("boundary_replay_rope_row_indices", ()),
        boundary_replay_rope_row_slice=base_plan_fields.get("boundary_replay_rope_row_slice"),
        token_indices=tuple(token_indices),
        token_slice=_contiguous_slice(token_indices),
        req_indices=tuple(req_indices),
        req_slice=_contiguous_slice(req_indices),
        cu_seqlens=tuple(cu_seqlens),
        start_pos=tuple(start_pos),
        compressed_row_indices=tuple(compressed_row_indices),
        compressed_row_slice=_contiguous_slice(compressed_row_indices),
        rope_row_indices=rope_row_indices,
        rope_row_slice=_contiguous_slice(rope_row_indices),
        valid_row_indices=tuple(output_keep_indices),
        valid_row_slice=_contiguous_slice(output_keep_indices),
        output_keep_indices=tuple(output_keep_indices),
        output_keep_slice=_contiguous_slice(output_keep_indices),
        slot_mapping_indices=tuple(slot_mapping_indices),
        slot_mapping_slice=_contiguous_slice(slot_mapping_indices),
        local_keep_to_full_row_indices=tuple(slot_mapping_indices),
        local_keep_to_slot_row_indices=tuple(slot_mapping_indices),
        tail_token_ranges=tuple(tail_token_ranges),
        padding_row_indices=tuple(padding_row_indices),
        padding_row_slice=_contiguous_slice(padding_row_indices),
        gather_compact_indices=gather_compact_indices,
        gather_compact_slice=_contiguous_slice(gather_compact_indices),
        sp_row_counts_per_rank=tuple(sp_row_counts),
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def run_compressor_op(
    x: Any,
    wkv: Any,
    wgate: Any,
    state_cache: Any,
    ape: Any,
    norm_weight: Any,
    rope_sin: Any,
    rope_cos: Any,
    *,
    state_block_table: Any,
    cu_seqlens: Any,
    seqused: Any,
    start_pos: Any,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    norm_eps: float,
    rotary_mode: int,
    cache_mode: int,
) -> Any:
    return torch.ops._C_ascend.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        coff=coff,
        norm_eps=norm_eps,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )


def build_padded_destination_for_scatter(
    full_slot_mapping: torch.Tensor,
    gather_compact_indices,  # Tensor[int] | None
    gather_compact_slice,  # (start, length) | None
    max_rows: int,
    tp_size: int,
    block_size: int,
    buffer: "torch.Tensor | None" = None,
):
    """Build a padded [P, 2] block-offset slot_mapping for direct scatter of a
    padded rank-major all-gather buffer (compact=False path).

    ``full_slot_mapping[k]`` is the destination slot of global compressed row
    ``k``; ``gather_compact_indices[k]`` is the physical row of global row ``k``
    in the padded rank-major buffer. So the valid destination for padded row
    ``gather_compact_indices[k]`` is ``full_slot_mapping[k]``.

    Padding rows are encoded as ``[-1, block_size-1]`` (the same invalid-slot
    encoding used by the compressor_metadata kernel), which linearizes to a
    negative offset and is filtered by the NoSort scatter kernel
    (``linearIndex >= start >= 0``).

    The caller may pass ``buffer`` (a pre-allocated [>=P, >=2] int32 tensor) for
    reuse; it is fully re-initialized every call so no stale destination
    survives across calls.

    Returns (padded_slot_mapping, buffer) where buffer is the one to reuse next.
    """
    if full_slot_mapping.ndim != 2 or full_slot_mapping.shape[1] != 2:
        raise ValueError("padded Compressor SP scatter requires [rows, 2] block-offset slot mapping")

    padded_rows = tp_size * max_rows
    slot_cols = full_slot_mapping.shape[1]

    if buffer is None or buffer.shape[0] < padded_rows or buffer.shape[1] < slot_cols:
        buffer = torch.empty(
            (padded_rows, slot_cols),
            dtype=full_slot_mapping.dtype,
            device=full_slot_mapping.device,
        )
    padded_slot_mapping = buffer[:padded_rows]

    # Initialize ALL rows to the invalid slot encoding [-1, block_size-1].
    padded_slot_mapping[:, 0].fill_(-1)
    padded_slot_mapping[:, 1].fill_(block_size - 1)

    # Scatter valid destinations: padded_slot_mapping[gather_compact_indices[k]]
    # = full_slot_mapping[k] for every global compressed row k.
    if gather_compact_indices is not None:
        padded_slot_mapping.index_copy_(0, gather_compact_indices.to(torch.long), full_slot_mapping)
    else:
        # Contiguous selector: every padded row is valid, so the destination is
        # a straight copy of the global-order slot_mapping.
        start, length = int(gather_compact_slice[0]), int(gather_compact_slice[1])
        padded_slot_mapping.narrow(0, 0, length).copy_(full_slot_mapping.narrow(0, start, length))

    return padded_slot_mapping, buffer
