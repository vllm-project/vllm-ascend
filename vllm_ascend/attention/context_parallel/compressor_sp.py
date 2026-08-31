import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain

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


def _selector_from_runs(runs: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], tuple[int, int] | None]:
    """Build an index selector only when ``narrow()`` cannot represent it.

    Every large selector in a plan is a concatenation of contiguous index runs,
    so whether ``narrow()`` can replace ``index_select()`` is decided by checking
    that consecutive runs join with unit stride. That is O(number of runs)
    instead of the O(number of indices) re-scan that inspecting the materialized
    selector requires. Contiguous selectors keep only ``(start, length)`` so the
    metadata path does not allocate a Python tuple or a device index tensor.

    Runs must be non-empty and given in selector order.
    """
    if not runs:
        return ((), (0, 0))
    start, total = runs[0]
    contiguous = True
    for run_start, run_length in runs[1:]:
        contiguous = contiguous and run_start == start + total
        total += run_length
    if contiguous:
        return ((), (start, total))
    return (
        tuple(chain.from_iterable(range(run_start, run_start + run_length) for run_start, run_length in runs)),
        None,
    )


def _contiguous_slice(values: Sequence[int]) -> tuple[int, int] | None:
    if not values:
        return (0, 0)
    length = len(values)
    start = values[0]
    # Endpoint span is a cheap necessary condition, so non-contiguous selectors
    # are rejected without touching the interior. The confirming comparison runs
    # in C against a materialized range, and matches the input container type so
    # that no copy of the selector itself is needed.
    if values[-1] - start != length - 1:
        return None
    expected = range(start, start + length)
    if isinstance(values, tuple):
        matches = values == tuple(expected)
    elif isinstance(values, list):
        matches = values == list(expected)
    else:
        matches = list(values) == list(expected)
    return (start, length) if matches else None


def _rope_row_runs(
    compressed_row_runs: Sequence[tuple[int, int]],
    num_compressed_rows: int,
    target_rows: int,
    source_rows: int,
) -> list[tuple[int, int]]:
    """Pad compressed-row runs up to the Compressor's RoPE row count.

    The operator wants exactly ``target_rows`` RoPE rows. Extra rows are never
    read, so they are filled by continuing past the last compressed row while
    that stays inside the source table, and otherwise by repeating row 0.
    """
    if target_rows < num_compressed_rows:
        raise ValueError(
            f"Compressor SP RoPE target has fewer rows than compressed rows: {target_rows} < {num_compressed_rows}"
        )
    padding_rows = target_rows - num_compressed_rows
    if padding_rows == 0:
        return list(compressed_row_runs)
    if compressed_row_runs:
        last_start, last_length = compressed_row_runs[-1]
        padding_start = last_start + last_length
        if padding_start + padding_rows <= source_rows:
            return [*compressed_row_runs, (padding_start, padding_rows)]
    return [*compressed_row_runs, *([(0, 1)] * padding_rows)]


def _build_c4_state_replay(
    query_start_loc: Sequence[int],
    request_start_positions: Sequence[int],
    rope_source_rows: int,
    replay_req_indices: Sequence[int] | None,
    output_rows_in_range: Callable[[int, int, int], tuple[int, int]],
) -> dict[str, tuple[int, ...] | tuple[int, int] | None]:
    token_runs: list[tuple[int, int]] = []
    req_indices: list[int] = []
    cu_seqlens = [0]
    start_pos: list[int] = []
    compressed_row_runs: list[tuple[int, int]] = []
    num_compressed_rows = 0
    num_replay_tokens = 0

    if replay_req_indices is None:
        replay_req_indices = range(len(request_start_positions))
    for req_index in replay_req_indices:
        request_start = request_start_positions[req_index]
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        replay_tokens = 8 + (4 if request_start > 0 else 0)
        replay_start = max(req_start, req_end - replay_tokens)
        replay_start_position = request_start + replay_start - req_start
        replay_start -= min(replay_start - req_start, replay_start_position % 8)

        run_length = req_end - replay_start
        token_runs.append((replay_start, run_length))
        num_replay_tokens += run_length
        req_indices.append(req_index)
        cu_seqlens.append(cu_seqlens[-1] + run_length)
        start_pos.append(request_start + replay_start - req_start)

        # Output tokens inside the replay window are consecutive outputs of the
        # request, so they own one contiguous global row run.
        row_lo, row_hi = output_rows_in_range(req_index, replay_start, req_end)
        if row_hi > row_lo:
            compressed_row_runs.append((row_lo, row_hi - row_lo))
            num_compressed_rows += row_hi - row_lo

    target_rows = min(num_replay_tokens, num_replay_tokens // 4 + len(req_indices))
    token_indices, token_slice = _selector_from_runs(token_runs)
    rope_row_indices, rope_row_slice = _selector_from_runs(
        _rope_row_runs(compressed_row_runs, num_compressed_rows, target_rows, rope_source_rows)
    )

    return {
        "state_replay_token_indices": token_indices,
        "state_replay_token_slice": token_slice,
        "state_replay_req_indices": tuple(req_indices),
        "state_replay_req_slice": _contiguous_slice(req_indices),
        "state_replay_cu_seqlens": tuple(cu_seqlens),
        "state_replay_start_pos": tuple(start_pos),
        "state_replay_rope_row_indices": rope_row_indices,
        "state_replay_rope_row_slice": rope_row_slice,
    }


def build_compressor_sp_plan(
    *,
    enabled: bool,
    has_prefill: bool,
    has_decode: bool,
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
        )

    if not enabled:
        return disabled("disabled")
    if not has_prefill:
        return disabled("unsupported_attention_state")
    if has_decode:
        return disabled("mixed_decode_prefill")
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
    request_start_positions = [seq_lens[i] - query_lens[i] for i in range(num_reqs)]
    if any(start < 0 for start in request_start_positions):
        return disabled("negative_start_pos")

    # Positions within a single request are contiguous by vLLM invariant, so a
    # per-request endpoint check is sufficient and avoids an O(num_tokens) scan
    # on the metadata-build hot path.
    for req_index, request_start_pos in enumerate(request_start_positions):
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        if (
            int(input_positions[req_start]) != request_start_pos
            or int(input_positions[req_end - 1]) != request_start_pos + (req_end - req_start) - 1
        ):
            return disabled("noncontiguous_positions")

    if num_input_tokens < min_input_tokens:
        return disabled("adaptive_small_chunk")

    num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    expected_owned_start = tp_rank * tokens_per_rank
    expected_owned_end_with_pad = expected_owned_start + tokens_per_rank
    expected_owned_end = min(expected_owned_start + tokens_per_rank, num_input_tokens)

    # A compressed row is emitted on every token where ``(position + 1) % ratio
    # == 0``. Positions are contiguous inside a request, so each request emits a
    # single arithmetic progression of output tokens, ``first + k * ratio``. Every
    # pass below walks those progressions per request instead of scanning all
    # tokens, which keeps planning off the O(num_input_tokens) path.
    req_output_first: list[int] = []
    req_output_count: list[int] = []
    req_output_row_base: list[int] = []
    total_global_rows = 0
    for req_index in range(num_reqs):
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        first_output = req_start + (-1 - request_start_positions[req_index]) % compress_ratio
        output_count = (req_end - first_output + compress_ratio - 1) // compress_ratio if first_output < req_end else 0
        req_output_first.append(first_output)
        req_output_count.append(output_count)
        req_output_row_base.append(total_global_rows)
        total_global_rows += output_count

    def output_rows_in_range(req_index: int, start: int, end: int) -> tuple[int, int]:
        """Global compressed rows ``[row_lo, row_hi)`` emitted inside ``[start, end)``.

        Consecutive output tokens of a request own consecutive global rows, so
        any flat range maps to one contiguous row run.
        """
        output_count = req_output_count[req_index]
        if output_count == 0:
            return (0, 0)
        first_output = req_output_first[req_index]
        last_output = first_output + (output_count - 1) * compress_ratio
        low = max(start, first_output)
        high = min(end, last_output + 1)
        if low >= high:
            return (0, 0)
        aligned_low = low + (first_output - low) % compress_ratio
        if aligned_low >= high:
            return (0, 0)
        row_base = req_output_row_base[req_index]
        return (
            row_base + (aligned_low - first_output) // compress_ratio,
            row_base + (high - 1 - first_output) // compress_ratio + 1,
        )

    # Rank ownership: token counts are a closed form over the shard bounds, and
    # each (request, shard) pair contributes exactly one contiguous row run.
    owned_token_counts = [0] * tp_size
    owned_row_runs: list[list[tuple[int, int]]] = [[] for _ in range(tp_size)]
    sp_row_counts = [0] * tp_size
    # owner = min(flat_index // tokens_per_rank, tp_size - 1).
    rank_bounds: list[tuple[int, int]] = []
    for rank in range(tp_size):
        shard_start = rank * tokens_per_rank
        shard_end = num_input_tokens if rank == tp_size - 1 else min(shard_start + tokens_per_rank, num_input_tokens)
        rank_bounds.append((shard_start, shard_end))
        owned_token_counts[rank] = max(0, shard_end - shard_start)
    for req_index in range(num_reqs):
        if req_output_count[req_index] == 0:
            continue
        for rank, (shard_start, shard_end) in enumerate(rank_bounds):
            row_lo, row_hi = output_rows_in_range(req_index, shard_start, shard_end)
            if row_hi > row_lo:
                owned_row_runs[rank].append((row_lo, row_hi - row_lo))
                sp_row_counts[rank] += row_hi - row_lo

    if any(count == 0 for count in owned_token_counts):
        return disabled("zero_token_rank")
    if local_start != expected_owned_start or local_end not in (
        expected_owned_end,
        expected_owned_end_with_pad,
    ):
        return disabled("invalid_local_range")
    local_end = expected_owned_end

    token_runs: list[tuple[int, int]] = []
    num_local_tokens = 0
    req_indices: list[int] = []
    cu_seqlens = [0]
    local_start_positions: list[int] = []
    compressed_row_runs: list[tuple[int, int]] = []
    num_compressed_rows = 0
    output_keep_runs: list[tuple[int, int]] = []

    for req_index in range(num_reqs):
        req_start = query_start_loc[req_index]
        req_end = query_start_loc[req_index + 1]
        true_start = max(req_start, local_start)
        true_end = min(req_end, local_end)
        if true_start >= true_end:
            continue

        first_true_output = None
        if req_output_count[req_index]:
            request_first_output = req_output_first[req_index]
            candidate = max(true_start, request_first_output)
            candidate += (request_first_output - candidate) % compress_ratio
            last_output = request_first_output + (req_output_count[req_index] - 1) * compress_ratio
            if candidate < true_end and candidate <= last_output:
                first_true_output = candidate
        dependency_tokens = 2 * compress_ratio if compress_ratio == 4 else compress_ratio
        expanded_start = true_start
        expanded_end = true_end
        if first_true_output is not None:
            expanded_start = max(req_start, first_true_output - dependency_tokens + 1)
            if compress_ratio == 4:
                expanded_start_position = request_start_positions[req_index] + expanded_start - req_start
                expanded_start -= min(
                    expanded_start - req_start,
                    expanded_start_position % dependency_tokens,
                )

            if expanded_end < req_end:
                expanded_end_position = request_start_positions[req_index] + expanded_end - req_start
                expanded_end = min(
                    req_end,
                    expanded_end + (-expanded_end_position) % dependency_tokens,
                )
        req_indices.append(req_index)
        token_runs.append((expanded_start, expanded_end - expanded_start))
        num_local_tokens += expanded_end - expanded_start
        cu_seqlens.append(cu_seqlens[-1] + expanded_end - expanded_start)
        local_start_positions.append(request_start_positions[req_index] + expanded_start - req_start)

        # The expanded range covers the true range, so both map to contiguous row
        # runs and the kept rows are a slice of the locally computed rows.
        expanded_row_lo, expanded_row_hi = output_rows_in_range(req_index, expanded_start, expanded_end)
        if expanded_row_hi > expanded_row_lo:
            local_row_base = num_compressed_rows
            compressed_row_runs.append((expanded_row_lo, expanded_row_hi - expanded_row_lo))
            num_compressed_rows += expanded_row_hi - expanded_row_lo
            true_row_lo, true_row_hi = output_rows_in_range(req_index, true_start, true_end)
            if true_row_hi > true_row_lo:
                keep_start = local_row_base + true_row_lo - expanded_row_lo
                output_keep_runs.append((keep_start, true_row_hi - true_row_lo))

    local_output_rows = min(
        num_local_tokens,
        num_local_tokens // compress_ratio + len(req_indices),
    )
    rope_source_rows = min(num_input_tokens, total_global_rows + num_reqs)
    token_indices, token_slice = _selector_from_runs(token_runs)
    output_keep_indices, output_keep_slice = _selector_from_runs(output_keep_runs)
    rope_row_indices, rope_row_slice = _selector_from_runs(
        _rope_row_runs(compressed_row_runs, num_compressed_rows, local_output_rows, rope_source_rows)
    )

    # Rebuild the rank-major AllGather layout. Rank shards partition the input
    # tokens, so the owned row runs partition the global rows: sorting them by
    # global row yields the selector directly, with no index-by-index write into
    # a full-length scratch list.
    max_rows_per_rank = max(sp_row_counts, default=0)
    gather_runs: list[tuple[int, int]] = []
    for rank, runs in enumerate(owned_row_runs):
        gather_position = rank * max_rows_per_rank
        for row_lo, run_length in runs:
            gather_runs.append((row_lo, gather_position, run_length))
            gather_position += run_length
    gather_runs.sort()
    gather_compact_indices, gather_compact_slice = _selector_from_runs(
        [(gather_position, run_length) for _, gather_position, run_length in gather_runs]
    )

    state_replay = (
        _build_c4_state_replay(
            query_start_loc,
            request_start_positions,
            rope_source_rows,
            tuple(range(num_reqs)),
            output_rows_in_range,
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
    state_sync_max_rows = max(state_sync_row_counts, default=0)
    state_sync_gather_compact_indices, state_sync_gather_compact_slice = _selector_from_runs(
        [(rank * state_sync_max_rows, count) for rank, count in enumerate(state_sync_row_counts) if count]
    )

    return CompressorSPPlan(
        enabled=True,
        reason="enabled",
        num_input_tokens=num_input_tokens,
        token_indices=token_indices,
        token_slice=token_slice,
        req_indices=tuple(req_indices),
        req_slice=_contiguous_slice(req_indices),
        cu_seqlens=tuple(cu_seqlens),
        start_pos=tuple(local_start_positions),
        rope_row_indices=rope_row_indices,
        rope_row_slice=rope_row_slice,
        output_keep_indices=output_keep_indices,
        output_keep_slice=output_keep_slice,
        gather_compact_indices=gather_compact_indices,
        gather_compact_slice=gather_compact_slice,
        sp_row_counts_per_rank=tuple(sp_row_counts),
        **state_replay,
        requires_state_sync=requires_state_sync,
        state_sync_token_indices=state_sync_indices_per_rank[tp_rank],
        state_sync_global_token_indices=state_sync_global_token_indices,
        state_sync_gather_compact_indices=state_sync_gather_compact_indices,
        state_sync_gather_compact_slice=state_sync_gather_compact_slice,
        state_sync_row_counts_per_rank=tuple(state_sync_row_counts),
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


@dataclass
class GatherWorkspace:
    """Persistent send/receive buffers for one Compressor-SP collective kind.

    Buffers are allocated once at the largest observed element count and only
    ever grown (never shrunk); per-layer calls consume zero-copy ``narrow``
    views of the same storage. Padding rows left in the send buffer keep stale
    content between calls: every consumer compacts through the plan's
    ``*_gather_compact`` selectors, which never select padded rows, so their
    content is transmitted but never read.
    """

    send: torch.Tensor | None = None
    gathered: torch.Tensor | None = None


def gather_workspace_view(
    workspace: GatherWorkspace,
    reference: torch.Tensor,
    elements: int,
    *,
    field: str,
) -> torch.Tensor:
    """Return a narrow view of a workspace buffer, growing it when needed.

    ``field`` is ``"send"`` (zero-initialized at allocation so a fresh buffer
    starts deterministic) or ``"gathered"`` (uninitialized receive side).
    """
    buffer = getattr(workspace, field)
    if (
        buffer is None
        or buffer.numel() < elements
        or buffer.dtype != reference.dtype
        or buffer.device != reference.device
    ):
        buffer = reference.new_zeros(elements) if field == "send" else reference.new_empty(elements)
        setattr(workspace, field, buffer)
    return buffer.narrow(0, 0, elements)


def fused_gather_rows(
    payloads: Sequence[tuple[torch.Tensor, Sequence[int]]],
    tp_size: int,
    all_gather_into_tensor: Callable[[torch.Tensor, torch.Tensor], None],
    workspace: GatherWorkspace | None = None,
) -> list[torch.Tensor] | None:
    """Replicate several rank-local row blocks with a single AllGather.

    Every block is zero-padded to its own per-rank maximum row count and the
    padded blocks are packed back to back into one flat rank-local buffer, so one
    collective rebuilds all blocks on all ranks. Each returned tensor uses the
    rank-major ``[tp_size * max_rows, *row_shape]`` layout that the plan's
    ``*_gather_compact`` selectors index.

    Returns ``None`` when the blocks cannot share a collective: mixed dtype or
    device, a block longer than its planned maximum, or nothing to send. Callers
    then fall back to one collective per block.

    Collective symmetry: the buffer sizes and the ``None`` decision depend only on
    the per-rank row counts carried by the plan (identical on every rank) and on
    model-constant row shapes and dtypes. Every rank therefore issues the same
    collective with the same shapes. Any future condition added here must keep
    that property.
    """
    if not payloads:
        return None
    reference = payloads[0][0]
    # (max_rows, row_width, row_shape, element_offset) per payload.
    layouts: list[tuple[int, int, tuple[int, ...], int]] = []
    elements_per_rank = 0
    for rows, row_counts in payloads:
        if rows.dtype != reference.dtype or rows.device != reference.device:
            return None
        max_rows = max(row_counts, default=0)
        if rows.shape[0] > max_rows:
            return None
        row_shape = tuple(rows.shape[1:])
        row_width = math.prod(row_shape)
        layouts.append((max_rows, row_width, row_shape, elements_per_rank))
        elements_per_rank += max_rows * row_width
    if elements_per_rank == 0:
        return None

    if workspace is None:
        workspace = GatherWorkspace()
    local_flat = gather_workspace_view(workspace, reference, elements_per_rank, field="send")
    for (rows, _), (_, row_width, _, offset) in zip(payloads, layouts):
        local_elements = rows.shape[0] * row_width
        if local_elements:
            local_flat[offset : offset + local_elements].copy_(rows.reshape(-1))
    gathered_flat = gather_workspace_view(
        workspace,
        reference,
        tp_size * elements_per_rank,
        field="gathered",
    )
    all_gather_into_tensor(gathered_flat, local_flat)

    # Each rank's blocks are interleaved in the flat buffer, so splitting them
    # back into rank-major order compacts a strided slice. That copy is far
    # cheaper than the collective it removes.
    gathered_by_rank = gathered_flat.view(tp_size, elements_per_rank)
    return [
        gathered_by_rank[:, offset : offset + max_rows * row_width].reshape(tp_size * max_rows, *row_shape)
        for max_rows, row_width, row_shape, offset in layouts
    ]


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
