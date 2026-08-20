from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressorSPPlan:
    enabled: bool
    reason: str
    num_input_tokens: int = 0
    token_indices: tuple[int, ...] = ()
    token_slice: tuple[int, int] | None = None
    req_indices: tuple[int, ...] = ()
    req_slice: tuple[int, int] | None = None
    cu_seqlens: tuple[int, ...] = ()
    start_pos: tuple[int, ...] = ()
    rope_row_indices: tuple[int, ...] = ()
    rope_row_slice: tuple[int, int] | None = None
    output_keep_indices: tuple[int, ...] = ()
    output_keep_slice: tuple[int, int] | None = None
    gather_compact_indices: tuple[int, ...] = ()
    gather_compact_slice: tuple[int, int] | None = None
    sp_row_counts_per_rank: tuple[int, ...] = ()
    state_replay_token_indices: tuple[int, ...] = ()
    state_replay_token_slice: tuple[int, int] | None = None
    state_replay_req_indices: tuple[int, ...] = ()
    state_replay_req_slice: tuple[int, int] | None = None
    state_replay_cu_seqlens: tuple[int, ...] = ()
    state_replay_start_pos: tuple[int, ...] = ()
    state_replay_rope_row_indices: tuple[int, ...] = ()
    state_replay_rope_row_slice: tuple[int, int] | None = None
    requires_state_sync: bool = False
    state_sync_token_indices: tuple[int, ...] = ()
    state_sync_global_token_indices: tuple[int, ...] = ()
    state_sync_gather_compact_indices: tuple[int, ...] = ()
    state_sync_gather_compact_slice: tuple[int, int] | None = None
    state_sync_row_counts_per_rank: tuple[int, ...] = ()
    tp_rank: int = 0
    tp_size: int = 1


def _contiguous_slice(values: Sequence[int]) -> tuple[int, int] | None:
    if not values:
        return (0, 0)
    start = values[0]
    if all(value == start + offset for offset, value in enumerate(values)):
        return (start, len(values))
    return None


def _in_ranges(index: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _rope_row_selector(
    compressed_rows: Sequence[int],
    target_rows: int,
    source_rows: int,
) -> tuple[int, ...]:
    if target_rows < len(compressed_rows):
        raise ValueError(
            f"Compressor SP RoPE target has fewer rows than compressed rows: {target_rows} < {len(compressed_rows)}"
        )
    padding_rows = target_rows - len(compressed_rows)
    if padding_rows == 0:
        return tuple(compressed_rows)
    if compressed_rows:
        padding_start = compressed_rows[-1] + 1
        padding_end = padding_start + padding_rows
        if padding_end <= source_rows:
            return tuple(compressed_rows) + tuple(range(padding_start, padding_end))
    return tuple(compressed_rows) + (0,) * padding_rows


def _gather_compact_indices(row_counts: Sequence[int]) -> tuple[int, ...]:
    max_rows = max(row_counts, default=0)
    return tuple(rank * max_rows + row for rank, count in enumerate(row_counts) for row in range(count))


def _build_c4_state_replay(
    input_positions: Sequence[int],
    query_start_loc: Sequence[int],
    request_start_positions: Sequence[int],
    rope_source_rows: int,
) -> dict[str, tuple[int, ...] | tuple[int, int] | None]:
    flat_to_compressed_row: dict[int, int] = {}
    compressed_row = 0
    for flat_index, position in enumerate(input_positions):
        if (position + 1) % 4 == 0:
            flat_to_compressed_row[flat_index] = compressed_row
            compressed_row += 1

    token_indices: list[int] = []
    req_indices: list[int] = []
    cu_seqlens = [0]
    start_pos: list[int] = []
    compressed_rows: list[int] = []

    for req_index, request_start in enumerate(request_start_positions):
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        replay_tokens = 8 + (4 if request_start > 0 else 0)
        replay_start = max(req_start, req_end - replay_tokens)
        while replay_start > req_start and input_positions[replay_start] % 8 != 0:
            replay_start -= 1

        req_token_indices = tuple(range(replay_start, req_end))
        token_indices.extend(req_token_indices)
        req_indices.append(req_index)
        cu_seqlens.append(cu_seqlens[-1] + len(req_token_indices))
        start_pos.append(input_positions[replay_start])

        compressed_rows.extend(
            flat_to_compressed_row[flat_index]
            for flat_index in req_token_indices
            if flat_index in flat_to_compressed_row
        )

    target_rows = min(len(token_indices), len(token_indices) // 4 + len(req_indices))
    rope_rows = _rope_row_selector(compressed_rows, target_rows, rope_source_rows)

    return {
        "state_replay_token_indices": tuple(token_indices),
        "state_replay_token_slice": _contiguous_slice(token_indices),
        "state_replay_req_indices": tuple(req_indices),
        "state_replay_req_slice": _contiguous_slice(req_indices),
        "state_replay_cu_seqlens": tuple(cu_seqlens),
        "state_replay_start_pos": tuple(start_pos),
        "state_replay_rope_row_indices": tuple(rope_rows),
        "state_replay_rope_row_slice": _contiguous_slice(rope_rows),
    }


def build_compressor_sp_plan(
    *,
    enabled: bool,
    has_prefill: bool,
    need_gather_q_kv: bool,
    tp_size: int,
    compress_ratio: int,
    input_positions: Sequence[int],
    query_start_loc: Sequence[int],
    seq_lens: Sequence[int],
    local_start: int,
    local_end: int,
    tp_rank: int = 0,
    min_input_tokens: int = 0,
) -> CompressorSPPlan:
    """Plan Compressor sequence-parallel execution for prefill batches.

    Unsupported layouts return a disabled plan before the local Compressor can
    mutate its state cache. Each enabled rank computes its owned token ranges,
    expanding them only where a compressed output needs neighboring tokens.
    """

    def disabled(reason: str) -> CompressorSPPlan:
        return CompressorSPPlan(
            enabled=False,
            reason=reason,
            num_input_tokens=len(input_positions),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    if not enabled:
        return disabled("disabled")
    if not has_prefill:
        return disabled("unsupported_attention_state")
    if not need_gather_q_kv or tp_size <= 1:
        return disabled("no_sequence_parallelism")
    if compress_ratio not in (4, 128):
        return disabled("unsupported_ratio")
    if not 0 <= tp_rank < tp_size:
        return disabled("invalid_tp_rank")
    if len(query_start_loc) < 2:
        return disabled("empty_query_start_loc")

    num_input_tokens = len(input_positions)
    if num_input_tokens < min_input_tokens:
        return disabled("adaptive_small_chunk")
    if query_start_loc[0] != 0 or query_start_loc[-1] != num_input_tokens:
        return disabled("query_start_loc_out_of_bounds")
    if any(query_start_loc[i] > query_start_loc[i + 1] for i in range(len(query_start_loc) - 1)):
        return disabled("invalid_query_start_loc")

    num_reqs = len(query_start_loc) - 1
    if len(seq_lens) < num_reqs:
        return disabled("seq_lens_too_short")
    query_lens = [query_start_loc[i + 1] - query_start_loc[i] for i in range(num_reqs)]
    if any(length <= 0 for length in query_lens):
        return disabled("empty_request")
    request_start_positions = [seq_lens[i] - query_lens[i] for i in range(num_reqs)]
    if any(start < 0 for start in request_start_positions):
        return disabled("negative_start_pos")
    for req_index, request_start_pos in enumerate(request_start_positions):
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        if any(
            input_positions[flat_index] != request_start_pos + flat_index - req_start
            for flat_index in range(req_start, req_end)
        ):
            return disabled("noncontiguous_positions")

    num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    if (tp_size - 1) * tokens_per_rank >= num_input_tokens:
        return disabled("zero_token_rank")

    sp_row_counts = [0] * tp_size
    for flat_index, position in enumerate(input_positions):
        if (position + 1) % compress_ratio == 0:
            owner = min(flat_index // tokens_per_rank, tp_size - 1)
            sp_row_counts[owner] += 1

    expected_owned_start = tp_rank * tokens_per_rank
    expected_owned_end = min(expected_owned_start + tokens_per_rank, num_input_tokens)
    if local_start != expected_owned_start or local_end != expected_owned_end:
        return disabled("invalid_local_range")

    true_ranges: list[tuple[int, int]] = []
    expanded_ranges: list[tuple[int, int]] = []
    token_indices: list[int] = []
    req_indices: list[int] = []
    cu_seqlens = [0]
    local_start_positions: list[int] = []

    for req_index in range(num_reqs):
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        true_start = max(req_start, local_start)
        true_end = min(req_end, local_end)
        if true_start >= true_end:
            continue

        first_output = next(
            (
                flat_index
                for flat_index in range(true_start, true_end)
                if (input_positions[flat_index] + 1) % compress_ratio == 0
            ),
            None,
        )
        dependency_tokens = 2 * compress_ratio if compress_ratio == 4 else compress_ratio
        expanded_start = true_start
        expanded_end = true_end
        if first_output is not None:
            expanded_start = max(req_start, first_output - dependency_tokens + 1)
            expected_position = max(
                request_start_positions[req_index],
                input_positions[first_output] - dependency_tokens + 1,
            )
            while expanded_start < first_output and input_positions[expanded_start] < expected_position:
                expanded_start += 1
            if compress_ratio == 4:
                while expanded_start > req_start and input_positions[expanded_start] % dependency_tokens != 0:
                    expanded_start -= 1

            if expanded_end < req_end:
                while expanded_end < req_end and (input_positions[expanded_end - 1] + 1) % dependency_tokens != 0:
                    expanded_end += 1
        true_ranges.append((true_start, true_end))
        expanded_ranges.append((expanded_start, expanded_end))
        req_indices.append(req_index)
        token_indices.extend(range(expanded_start, expanded_end))
        cu_seqlens.append(cu_seqlens[-1] + expanded_end - expanded_start)
        local_start_positions.append(request_start_positions[req_index] + expanded_start - req_start)

    compressed_row_indices: list[int] = []
    output_keep_indices: list[int] = []
    global_compressed_row = 0
    for flat_index, position in enumerate(input_positions):
        if (position + 1) % compress_ratio != 0:
            continue
        if _in_ranges(flat_index, expanded_ranges):
            local_row = len(compressed_row_indices)
            compressed_row_indices.append(global_compressed_row)
            if _in_ranges(flat_index, true_ranges):
                output_keep_indices.append(local_row)
        global_compressed_row += 1

    local_output_rows = min(
        len(token_indices),
        len(token_indices) // compress_ratio + len(req_indices),
    )
    rope_source_rows = min(num_input_tokens, global_compressed_row + num_reqs)
    rope_row_indices = _rope_row_selector(
        compressed_row_indices,
        local_output_rows,
        rope_source_rows,
    )

    gather_compact_indices = _gather_compact_indices(sp_row_counts)

    state_replay = (
        _build_c4_state_replay(
            input_positions,
            query_start_loc,
            request_start_positions,
            rope_source_rows,
        )
        if compress_ratio == 4
        else {}
    )

    # C4 restores its bounded state with replay. C128 only needs the unfinished
    # compression block; aligned ends therefore need no synchronization data.
    state_sync_ranges = (
        [
            (max(req_start, req_end - seq_lens[req_index] % compress_ratio), req_end)
            for req_index, (req_start, req_end) in enumerate(zip(query_start_loc[:-1], query_start_loc[1:]))
        ]
        if compress_ratio == 128
        else []
    )
    requires_state_sync = any(start < end for start, end in state_sync_ranges)
    state_sync_indices_per_rank: list[tuple[int, ...]] = [()] * tp_size
    if requires_state_sync:
        state_sync_indices_per_rank = []
        for rank in range(tp_size):
            rank_start = rank * tokens_per_rank
            rank_end = min(rank_start + tokens_per_rank, num_input_tokens)
            state_sync_indices_per_rank.append(
                tuple(
                    flat_index
                    for sync_start, sync_end in state_sync_ranges
                    for flat_index in range(max(sync_start, rank_start), min(sync_end, rank_end))
                )
            )
    state_sync_row_counts = [len(indices) for indices in state_sync_indices_per_rank]
    state_sync_global_token_indices = tuple(
        flat_index for rank_indices in state_sync_indices_per_rank for flat_index in rank_indices
    )
    state_sync_gather_compact_indices = _gather_compact_indices(state_sync_row_counts)

    return CompressorSPPlan(
        enabled=True,
        reason="enabled",
        num_input_tokens=num_input_tokens,
        token_indices=tuple(token_indices),
        token_slice=_contiguous_slice(token_indices),
        req_indices=tuple(req_indices),
        req_slice=_contiguous_slice(req_indices),
        cu_seqlens=tuple(cu_seqlens),
        start_pos=tuple(local_start_positions),
        rope_row_indices=rope_row_indices,
        rope_row_slice=_contiguous_slice(rope_row_indices),
        output_keep_indices=tuple(output_keep_indices),
        output_keep_slice=_contiguous_slice(output_keep_indices),
        gather_compact_indices=gather_compact_indices,
        gather_compact_slice=_contiguous_slice(gather_compact_indices),
        sp_row_counts_per_rank=tuple(sp_row_counts),
        **state_replay,
        requires_state_sync=requires_state_sync,
        state_sync_token_indices=state_sync_indices_per_rank[tp_rank],
        state_sync_global_token_indices=state_sync_global_token_indices,
        state_sync_gather_compact_indices=state_sync_gather_compact_indices,
        state_sync_gather_compact_slice=_contiguous_slice(state_sync_gather_compact_indices),
        state_sync_row_counts_per_rank=tuple(state_sync_row_counts),
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def is_block_offset_slot_mapping(slot_mapping: torch.Tensor) -> bool:
    return slot_mapping.ndim == 2 and slot_mapping.shape[1] == 2


def flatten_slot_mapping(slot_mapping: torch.Tensor, block_size: int) -> torch.Tensor:
    """Return flat cache slots for either supported slot-mapping format."""
    if is_block_offset_slot_mapping(slot_mapping):
        return slot_mapping[:, 0].to(torch.long) * block_size + slot_mapping[:, 1].to(torch.long)
    if slot_mapping.ndim == 1:
        return slot_mapping.to(torch.long)
    raise ValueError("slot mapping must be flat [rows] or block-offset [rows, 2]")


def _block_cache_coordinates(flat_slots: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat_slots = flat_slots.to(torch.long)
    return (
        torch.div(flat_slots, block_size, rounding_mode="floor"),
        torch.remainder(flat_slots, block_size),
    )


def select_block_cache_rows(
    cache_rows: torch.Tensor,
    flat_slots: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Select rows without flattening a potentially non-contiguous cache."""
    block_indices, block_offsets = _block_cache_coordinates(flat_slots, block_size)
    return cache_rows[block_indices, block_offsets]


def update_block_cache_rows_(
    cache_rows: torch.Tensor,
    flat_slots: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
) -> None:
    """Update rows without flattening a potentially non-contiguous cache."""
    block_indices, block_offsets = _block_cache_coordinates(flat_slots, block_size)
    cache_rows.index_put_((block_indices, block_offsets), values)


def build_padded_destination_for_scatter(
    full_slot_mapping: torch.Tensor,
    gather_compact_indices: torch.Tensor,
    padded_rows: int,
    block_size: int,
    buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a ragged rank-major AllGather buffer to block-offset cache slots."""
    required_shape = (padded_rows, 2)
    if (
        buffer is None
        or buffer.shape[0] < padded_rows
        or buffer.dtype != full_slot_mapping.dtype
        or buffer.device != full_slot_mapping.device
    ):
        buffer = torch.empty(required_shape, dtype=full_slot_mapping.dtype, device=full_slot_mapping.device)

    padded_slot_mapping = buffer[:padded_rows, :2]
    padded_slot_mapping[:, 0].fill_(-1)
    padded_slot_mapping[:, 1].fill_(block_size - 1)
    padded_slot_mapping.index_copy_(0, gather_compact_indices.to(torch.long), full_slot_mapping)
    return padded_slot_mapping, buffer
