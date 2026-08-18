from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressorSPPlan:
    enabled: bool
    reason: str
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


def _rope_row_selector(compressed_rows: Sequence[int], target_rows: int) -> tuple[int, ...]:
    if target_rows < len(compressed_rows):
        raise ValueError(
            f"Compressor SP RoPE target has fewer rows than compressed rows: {target_rows} < {len(compressed_rows)}"
        )
    return tuple(compressed_rows) + (0,) * (target_rows - len(compressed_rows))


def build_compressor_sp_plan(
    *,
    enabled: bool,
    is_full_prefill: bool,
    need_gather_q_kv: bool,
    tp_size: int,
    compress_ratio: int,
    input_positions: Sequence[int],
    query_start_loc: Sequence[int],
    seq_lens: Sequence[int],
    local_start: int,
    local_end: int,
    tp_rank: int = 0,
) -> CompressorSPPlan:
    """Plan Compressor sequence-parallel execution for aligned full prefills.

    Unsupported layouts return a disabled plan before the local Compressor can
    mutate its state cache. Each enabled rank computes only the expanded token
    ranges needed for its owned compressed rows.
    """

    def disabled(reason: str) -> CompressorSPPlan:
        return CompressorSPPlan(enabled=False, reason=reason, tp_rank=tp_rank, tp_size=tp_size)

    if not enabled:
        return disabled("disabled")
    if not is_full_prefill:
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
    if any(seq_lens[i] != query_lens[i] for i in range(num_reqs)):
        return disabled("nonzero_start_pos")
    if any(seq_lens[i] % compress_ratio != 0 for i in range(num_reqs)):
        return disabled("seq_len_not_aligned")

    num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    sp_row_counts = [0] * tp_size
    for flat_index, position in enumerate(input_positions):
        if (position + 1) % compress_ratio == 0:
            owner = min(flat_index // tokens_per_rank, tp_size - 1)
            sp_row_counts[owner] += 1
    if any(count == 0 for count in sp_row_counts):
        return disabled("zero_row_rank")

    owned_start = max(0, local_start)
    owned_end = min(max(owned_start, local_end), num_input_tokens)
    if owned_start >= owned_end:
        return disabled("no_local_tokens")

    true_ranges: list[tuple[int, int]] = []
    expanded_ranges: list[tuple[int, int]] = []
    token_indices: list[int] = []
    req_indices: list[int] = []
    cu_seqlens = [0]
    local_start_positions: list[int] = []

    for req_index in range(num_reqs):
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        true_start = max(req_start, owned_start)
        true_end = min(req_end, owned_end)
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
        if first_output is None:
            continue

        dependency_tokens = 2 * compress_ratio if compress_ratio == 4 else compress_ratio
        expanded_start = max(req_start, first_output - dependency_tokens + 1)
        expected_position = max(0, input_positions[first_output] - dependency_tokens + 1)
        while expanded_start < first_output and input_positions[expanded_start] < expected_position:
            expanded_start += 1
        if compress_ratio == 4:
            while expanded_start > req_start and input_positions[expanded_start] % 8 != 0:
                expanded_start -= 1

        expanded_end = true_end
        if expanded_end < req_end:
            right_alignment = 8 if compress_ratio == 4 else compress_ratio
            while expanded_end < req_end and (input_positions[expanded_end - 1] + 1) % right_alignment != 0:
                expanded_end += 1
        if expanded_start >= expanded_end:
            continue

        true_ranges.append((true_start, true_end))
        expanded_ranges.append((expanded_start, expanded_end))
        req_indices.append(req_index)
        token_indices.extend(range(expanded_start, expanded_end))
        cu_seqlens.append(cu_seqlens[-1] + expanded_end - expanded_start)
        local_start_positions.append(expanded_start - req_start)

    if not token_indices:
        return disabled("no_local_compressed_rows")

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

    expected_local_rows = sp_row_counts[tp_rank]
    if len(output_keep_indices) != expected_local_rows:
        return disabled("local_row_count_mismatch")

    local_output_rows = min(
        len(token_indices),
        len(token_indices) // compress_ratio + len(req_indices),
    )
    rope_row_indices = _rope_row_selector(compressed_row_indices, local_output_rows)

    max_rows = max(sp_row_counts)
    gather_compact_indices = tuple(
        rank * max_rows + row for rank, count in enumerate(sp_row_counts) for row in range(count)
    )

    return CompressorSPPlan(
        enabled=True,
        reason="enabled",
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
        gather_compact_slice=(_contiguous_slice(gather_compact_indices) if len(set(sp_row_counts)) == 1 else None),
        sp_row_counts_per_rank=tuple(sp_row_counts),
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def is_block_offset_slot_mapping(slot_mapping: torch.Tensor) -> bool:
    return slot_mapping.ndim == 2 and slot_mapping.shape[1] == 2


def build_padded_destination_for_scatter(
    full_slot_mapping: torch.Tensor,
    gather_compact_indices: torch.Tensor,
    padded_rows: int,
    block_size: int,
    buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a ragged rank-major AllGather buffer to block-offset cache slots."""
    if not is_block_offset_slot_mapping(full_slot_mapping):
        raise ValueError("padded Compressor SP scatter requires [rows, 2] block-offset slot mapping")
    if gather_compact_indices.ndim != 1:
        raise ValueError("gather_compact_indices must be one-dimensional")
    if gather_compact_indices.numel() != full_slot_mapping.shape[0]:
        raise ValueError("gather selector and full slot mapping row counts must match")
    if padded_rows < full_slot_mapping.shape[0]:
        raise ValueError("padded row count cannot be smaller than the valid row count")

    required_shape = (padded_rows, 2)
    if (
        buffer is None
        or buffer.shape[0] < padded_rows
        or buffer.shape[1] < 2
        or buffer.dtype != full_slot_mapping.dtype
        or buffer.device != full_slot_mapping.device
    ):
        buffer = torch.empty(required_shape, dtype=full_slot_mapping.dtype, device=full_slot_mapping.device)

    padded_slot_mapping = buffer[:padded_rows, :2]
    padded_slot_mapping[:, 0].fill_(-1)
    padded_slot_mapping[:, 1].fill_(block_size - 1)
    padded_slot_mapping.index_copy_(0, gather_compact_indices.to(torch.long), full_slot_mapping)
    return padded_slot_mapping, buffer
