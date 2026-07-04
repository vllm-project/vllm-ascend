# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from vllm.logger import logger

from vllm_ascend import envs

DSparkAttentionCustomOp = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        float,
    ],
    torch.Tensor,
]

DSPARK_SAS_MASK_MODE = 4
DSPARK_SAS_CMP_MASK_MODE = 3


def _dspark_sas_window(block_size: int, window_size: int) -> tuple[int, int, int]:
    return (
        DSPARK_SAS_MASK_MODE,
        window_size + block_size - 1,
        block_size - 1,
    )


def _dspark_sparse_sas_window(block_size: int, window_size: int) -> tuple[int, int, int]:
    # Compatibility scheduling bound for PA_ND + ori_sparse_indices. This is
    # not the DSpark visible-token definition; the slot list is authoritative.
    return (
        DSPARK_SAS_MASK_MODE,
        window_size + block_size - 1,
        block_size - 1,
    )


def _dspark_sas_lens_match_scheduling(
    dspark_swa_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor | None,
    block_size: int,
    window_size: int,
    num_query_tokens: int,
) -> bool:
    """Whether current fixed-bound SAS metadata covers the explicit slot list.

    Until the metadata op takes per-token DSpark lens, the AICore sparse path
    schedules up to `min(seq_len, window_size + block_size)` ori-side slots per
    request. Padding `-1` slots are trimmed in the ori-sparse kernel path, so
    shorter partial draft blocks are valid as long as the fixed bound covers
    the explicit visible slot length.
    """
    if num_query_tokens == 0 or dspark_swa_lens.numel() < num_query_tokens:
        return False
    if query_start_loc.numel() < 2 or seq_lens.numel() == 0:
        return False

    lens = dspark_swa_lens[:num_query_tokens].to(device="cpu", dtype=torch.long)
    q_starts = query_start_loc.to(device="cpu", dtype=torch.long)
    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.long)
    token_to_req = (
        token_to_req_indices[:num_query_tokens].to(device="cpu", dtype=torch.long)
        if token_to_req_indices is not None
        else None
    )

    covered = torch.zeros((num_query_tokens,), dtype=torch.bool)
    req_count = min(q_starts.numel() - 1, seq_lens_cpu.numel())
    scheduled_len_upper = int(window_size) + int(block_size)
    for req_idx in range(req_count):
        row_start = max(int(q_starts[req_idx].item()), 0)
        row_end = min(int(q_starts[req_idx + 1].item()), num_query_tokens)
        if row_end <= row_start:
            continue
        if token_to_req is None:
            row_indices = torch.arange(row_start, row_end, dtype=torch.long)
        else:
            row_indices = torch.nonzero(token_to_req == req_idx, as_tuple=False).flatten()
            row_indices = row_indices[row_indices < num_query_tokens]
            if row_indices.numel() == 0:
                continue
            expected_rows = torch.arange(row_start, row_end, dtype=torch.long)
            if row_indices.numel() != expected_rows.numel() or not bool(torch.all(row_indices == expected_rows).item()):
                return False

        scheduled_len = min(int(seq_lens_cpu[req_idx].item()), scheduled_len_upper)
        row_lens = lens.index_select(0, row_indices)
        if scheduled_len <= 0 or not bool(torch.all((row_lens > 0) & (row_lens <= scheduled_len)).item()):
            return False
        if not bool(torch.all(row_lens == row_lens[0]).item()):
            return False
        covered[row_indices] = True

    return bool(torch.all(covered).item())


def _validate_query_block_slots(request_slots: torch.Tensor, block_size: int) -> None:
    for block_offset in range(0, request_slots.numel(), block_size):
        block_slots = request_slots[block_offset : block_offset + block_size].to(torch.long)
        if block_slots.numel() == 0:
            continue
        if torch.any(block_slots != block_slots[0]):
            raise ValueError("DSpark query request_slots must be constant within each draft block")


def _gather_context_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    request_slot: int,
    context_start: int,
    context_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_end < context_start:
        empty = k_cache.new_empty((0,) + k_cache.shape[2:])
        return empty, v_cache.new_empty((0,) + v_cache.shape[2:])

    cache_capacity = k_cache.shape[1]
    ctx_positions = torch.arange(
        context_start,
        context_end + 1,
        dtype=torch.long,
        device=k_cache.device,
    )
    cache_indices = ctx_positions % cache_capacity
    cached_positions = cache_positions[request_slot, cache_indices].to(torch.long)
    valid = cache_valid[request_slot, cache_indices] & (cached_positions == ctx_positions)
    return k_cache[request_slot, cache_indices][valid], v_cache[request_slot, cache_indices][valid]


def _dspark_attention_reference(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if k_ctx.shape[1] == 1 and q.shape[1] != 1:
        k_ctx = k_ctx.expand(-1, q.shape[1], -1)
    if v_ctx.shape[1] == 1 and q.shape[1] != 1:
        v_ctx = v_ctx.expand(-1, q.shape[1], -1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k_ctx.float()) * softmax_scale
    sink = attn_sink[: q.shape[1]].float().view(1, q.shape[1], 1)
    scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max)
    probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
    return torch.einsum("qhk,khd->qhd", probs, v_ctx.float()).to(q.dtype)


def _unwrap_single_kv_cache(kv_cache: torch.Tensor | list | tuple) -> torch.Tensor:
    while isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 1:
        kv_cache = kv_cache[0]
    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError(f"Expected tensor KV cache, got {type(kv_cache)!r}")
    return kv_cache


def _gather_paged_swa_kv_positions(
    kv_cache: torch.Tensor | list | tuple,
    block_table: torch.Tensor,
    req_idx: int,
    positions: torch.Tensor,
    cache_block_size: int,
) -> torch.Tensor:
    cache = _unwrap_single_kv_cache(kv_cache)
    if positions.numel() == 0:
        return cache.new_empty((0,) + cache.shape[2:])
    positions = positions.to(device=cache.device, dtype=torch.long)
    block_nums = positions // cache_block_size
    block_offsets = positions % cache_block_size
    req_block_table = block_table[req_idx].to(device=cache.device, dtype=torch.long)
    block_ids = req_block_table.index_select(0, block_nums)
    return cache[block_ids, block_offsets]


def _gather_paged_swa_kv_slots(
    kv_cache: torch.Tensor | list | tuple,
    slot_ids: torch.Tensor,
    cache_block_size: int,
) -> torch.Tensor:
    cache = _unwrap_single_kv_cache(kv_cache)
    valid_slot_ids = slot_ids[slot_ids >= 0].to(device=cache.device, dtype=torch.long)
    if valid_slot_ids.numel() == 0:
        return cache.new_empty((0,) + cache.shape[2:])
    block_ids = valid_slot_ids // cache_block_size
    block_offsets = valid_slot_ids % cache_block_size
    return cache[block_ids, block_offsets]


def _valid_slot_rows(slot_mapping: torch.Tensor) -> torch.Tensor:
    if slot_mapping.ndim == 1:
        return slot_mapping >= 0
    return torch.all(slot_mapping >= 0, dim=-1)


def _aligned_dspark_index_width(window_size: int, block_size: int, alignment: int = 128) -> int:
    min_width = int(window_size) + int(block_size)
    return ((min_width + alignment - 1) // alignment) * alignment


def build_dspark_swa_indices(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    block_size: int,
    window_size: int,
    cache_block_size: int,
    *,
    index_width: int | None = None,
    query_start_loc: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DSpark non-causal visible slot ids for a paged SWA cache.

    The output is the Python equivalent of upstream DSpark's non-causal SWA
    metadata: each token in a draft block sees the trailing context window plus
    the whole current draft block. Invalid/padded rows get lens=0 and -1 slots.
    """
    if index_width is None:
        index_width = _aligned_dspark_index_width(window_size, block_size)
    min_width = int(window_size) + int(block_size)
    if index_width < min_width:
        raise ValueError(
            "DSpark SWA index_width must cover window_size + block_size: "
            f"index_width={index_width}, required={min_width}"
        )
    if positions.numel() == 0:
        return (
            torch.empty((0, 1, index_width), dtype=torch.int32, device=positions.device),
            torch.empty((0,), dtype=torch.int32, device=positions.device),
        )

    indices = torch.full(
        (positions.numel(), 1, index_width),
        -1,
        dtype=torch.int32,
        device=positions.device,
    )
    lens = torch.zeros((positions.numel(),), dtype=torch.int32, device=positions.device)
    pos_long = positions.to(device=positions.device, dtype=torch.long)

    if (query_start_loc is None) != (seq_lens is None):
        raise ValueError("DSpark SWA query_start_loc and seq_lens must be provided together")

    if query_start_loc is not None and seq_lens is not None:
        req_count = max(int(query_start_loc.numel()) - 1, 0)
        token_to_req = None
        if token_to_req_indices is not None:
            if token_to_req_indices.numel() < positions.numel():
                raise ValueError(
                    "DSpark SWA token_to_req_indices must cover query tokens: "
                    f"token_to_req_indices={token_to_req_indices.numel()}, positions={positions.numel()}"
                )
            token_to_req = token_to_req_indices[: positions.numel()].to(
                device=positions.device,
                dtype=torch.long,
            )
            req_count = min(req_count, block_table.shape[0], seq_lens.numel())
            if req_count <= 0:
                return indices, lens

            valid_rows = (token_to_req >= 0) & (token_to_req < req_count)
            if slot_mapping is not None:
                valid_rows &= _valid_slot_rows(slot_mapping[: positions.numel()].to(device=positions.device))
            if not bool(torch.any(valid_rows).item()):
                return indices, lens

            req_query_start = query_start_loc[: req_count + 1].to(device=positions.device, dtype=torch.long)
            query_lens = req_query_start[1:] - req_query_start[:-1]
            req_seq_lens = seq_lens[:req_count].to(device=positions.device, dtype=torch.long)
            start_pos = torch.clamp(req_seq_lens - query_lens - int(window_size), min=0)
            visible_lens = req_seq_lens - start_pos
            max_visible_len = int(visible_lens.max().item()) if visible_lens.numel() else 0
            if max_visible_len > index_width:
                raise ValueError(
                    "DSpark SWA visible length exceeds index_width: "
                    f"visible_len={max_visible_len}, index_width={index_width}"
                )
            if max_visible_len <= 0 or block_table.shape[1] <= 0:
                return indices, lens

            offsets = torch.arange(index_width, dtype=torch.long, device=positions.device)
            visible_positions = start_pos.unsqueeze(1) + offsets.unsqueeze(0)
            visible_mask = offsets.unsqueeze(0) < visible_lens.unsqueeze(1)
            block_nums = visible_positions // int(cache_block_size)
            if bool(torch.any(visible_mask).item()):
                max_block_num = int(block_nums[visible_mask].max().item())
                if max_block_num >= block_table.shape[1]:
                    raise ValueError(
                        "DSpark SWA block index exceeds block table width: "
                        f"block_num={max_block_num}, width={block_table.shape[1]}"
                    )

            req_block_table = block_table[:req_count].to(device=positions.device, dtype=torch.long)
            block_nums = block_nums.clamp(0, block_table.shape[1] - 1)
            block_ids = req_block_table.gather(1, block_nums)
            block_offsets = visible_positions % int(cache_block_size)
            slot_ids = (block_ids * int(cache_block_size) + block_offsets).to(torch.int32)
            slot_ids.masked_fill_(~visible_mask, -1)

            row_req = token_to_req.clamp(0, req_count - 1)
            indices[:, 0, :] = slot_ids.index_select(0, row_req)
            lens.copy_(visible_lens.index_select(0, row_req).to(torch.int32))
            indices[~valid_rows] = -1
            lens[~valid_rows] = 0
            return indices, lens

        for req_idx in range(req_count):
            if req_idx >= block_table.shape[0] or req_idx >= seq_lens.numel():
                continue

            query_start = int(query_start_loc[req_idx].item())
            query_end = int(query_start_loc[req_idx + 1].item())
            query_start = max(query_start, 0)
            query_end = min(query_end, positions.numel())
            if query_end <= query_start:
                continue

            if token_to_req is None:
                row_indices = torch.arange(query_start, query_end, dtype=torch.long, device=positions.device)
            else:
                row_indices = torch.nonzero(token_to_req == req_idx, as_tuple=False).flatten()
                row_indices = row_indices[row_indices < positions.numel()]
                if row_indices.numel() == 0:
                    continue

            valid_mask = torch.ones(row_indices.numel(), dtype=torch.bool, device=positions.device)
            if slot_mapping is not None:
                req_slots = slot_mapping.index_select(0, row_indices).to(device=positions.device)
                valid_mask = _valid_slot_rows(req_slots)
            if not bool(torch.any(valid_mask).item()):
                continue

            query_len = int(query_start_loc[req_idx + 1].item()) - int(query_start_loc[req_idx].item())
            seq_len = int(seq_lens[req_idx].item())
            prefix_len = seq_len - query_len
            start_pos = max(prefix_len - int(window_size), 0)
            visible_len = seq_len - start_pos
            if visible_len > index_width:
                raise ValueError(
                    "DSpark SWA visible length exceeds index_width: "
                    f"visible_len={visible_len}, index_width={index_width}"
                )

            visible_positions = torch.arange(
                start_pos,
                seq_len,
                dtype=torch.long,
                device=positions.device,
            )
            block_nums = visible_positions // cache_block_size
            block_offsets = visible_positions % cache_block_size
            req_block_table = block_table[req_idx].to(device=positions.device, dtype=torch.long)
            block_ids = req_block_table.index_select(0, block_nums)
            slot_ids = (block_ids * cache_block_size + block_offsets).to(torch.int32)

            valid_rows = row_indices[valid_mask]
            indices[valid_rows, 0, :visible_len] = slot_ids
            lens[valid_rows] = visible_len
        return indices, lens

    for block_offset in range(0, positions.numel(), block_size):
        block_end = min(block_offset + block_size, positions.numel())
        req_idx = block_offset // block_size
        if req_idx >= block_table.shape[0]:
            continue

        valid_mask = torch.ones(block_end - block_offset, dtype=torch.bool, device=positions.device)
        if slot_mapping is not None:
            block_slots = slot_mapping[block_offset:block_end].to(device=positions.device)
            valid_mask = _valid_slot_rows(block_slots)
        if not bool(torch.any(valid_mask).item()):
            continue

        valid_pos = pos_long[block_offset:block_end][valid_mask]
        query_len = int(valid_pos.numel())
        seq_len = int(valid_pos.max().item()) + 1
        prefix_len = seq_len - query_len
        start_pos = max(prefix_len - int(window_size), 0)
        visible_len = seq_len - start_pos
        if visible_len > index_width:
            raise ValueError(
                f"DSpark SWA visible length exceeds index_width: visible_len={visible_len}, index_width={index_width}"
            )

        visible_positions = torch.arange(
            start_pos,
            seq_len,
            dtype=torch.long,
            device=positions.device,
        )
        block_nums = visible_positions // cache_block_size
        block_offsets = visible_positions % cache_block_size
        req_block_table = block_table[req_idx].to(device=positions.device, dtype=torch.long)
        block_ids = req_block_table.index_select(0, block_nums)
        slot_ids = (block_ids * cache_block_size + block_offsets).to(torch.int32)

        valid_rows = torch.arange(block_end - block_offset, device=positions.device)[valid_mask] + block_offset
        indices[valid_rows, 0, :visible_len] = slot_ids
        lens[valid_rows] = visible_len

    return indices, lens


def dspark_attention_from_standard_cache(
    q: torch.Tensor,
    standard_kv_cache: torch.Tensor | list | tuple,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    draft_kv: torch.Tensor | None,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    cache_block_size: int,
    softmax_scale: float,
    *,
    request_slots: torch.Tensor | None = None,
    cache_positions: torch.Tensor | None = None,
    cache_valid: torch.Tensor | None = None,
    dspark_swa_indices: torch.Tensor | None = None,
    dspark_swa_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """PTA DSpark attention over vLLM-style paged SWA KV cache.

    This mirrors upstream DSpark SWA metadata: each query in a draft block
    attends to the trailing context window plus the full current draft block.
    """
    del draft_kv, request_slots, cache_positions, cache_valid
    if (dspark_swa_indices is None) != (dspark_swa_lens is None):
        raise ValueError("DSpark SWA indices and lens must be provided together")
    if dspark_swa_indices is None:
        dspark_swa_indices, dspark_swa_lens = build_dspark_swa_indices(
            block_table,
            positions,
            slot_mapping,
            block_size,
            window_size,
            cache_block_size,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req_indices,
        )
    assert dspark_swa_lens is not None
    out = torch.empty_like(q)
    out.zero_()

    if query_start_loc is None:
        query_groups = [
            torch.arange(
                block_offset,
                min(block_offset + block_size, positions.numel()),
                dtype=torch.long,
                device=q.device,
            )
            for block_offset in range(0, positions.numel(), block_size)
        ]
    elif token_to_req_indices is not None:
        token_to_req = token_to_req_indices[: positions.numel()].to(device=q.device, dtype=torch.long)
        query_groups = []
        for req_idx in range(max(int(query_start_loc.numel()) - 1, 0)):
            row_indices = torch.nonzero(token_to_req == req_idx, as_tuple=False).flatten()
            row_indices = row_indices[row_indices < positions.numel()]
            if row_indices.numel() > 0:
                query_groups.append(row_indices)
    else:
        query_groups = []
        for req_idx in range(max(int(query_start_loc.numel()) - 1, 0)):
            query_start = max(int(query_start_loc[req_idx].item()), 0)
            query_end = min(int(query_start_loc[req_idx + 1].item()), positions.numel())
            if query_end > query_start:
                query_groups.append(
                    torch.arange(
                        query_start,
                        query_end,
                        dtype=torch.long,
                        device=q.device,
                    )
                )

    for row_indices in query_groups:
        q_block = q.index_select(0, row_indices)
        block_lens = dspark_swa_lens.index_select(0, row_indices).to(device=q.device, dtype=torch.long)
        valid_mask = block_lens > 0

        if slot_mapping is not None:
            block_slots = slot_mapping.index_select(0, row_indices).to(device=q.device)
            valid_mask &= _valid_slot_rows(block_slots)
            if block_slots.numel() == 0 or torch.all(~valid_mask):
                continue

        valid_lens = block_lens[valid_mask]
        if valid_lens.numel() == 0:
            continue

        visible_len = int(valid_lens[0].item())
        if not bool(torch.all(valid_lens == visible_len).item()):
            raise ValueError("DSpark SWA lens must be shared within a draft block")
        visible_slots = dspark_swa_indices.index_select(0, row_indices)[:, 0, :visible_len].to(
            device=q.device,
        )
        visible_slots = visible_slots[valid_mask]
        if visible_slots.shape[0] > 1 and not bool(torch.all(visible_slots == visible_slots[:1]).item()):
            raise ValueError("DSpark SWA slot ids must be shared within a draft block")
        visible_slots = visible_slots[0]
        k_ctx = _gather_paged_swa_kv_slots(
            standard_kv_cache,
            visible_slots,
            cache_block_size,
        )
        if k_ctx.dim() != 3 or k_ctx.shape[-1] != q.shape[-1]:
            raise ValueError(
                "DSpark standard SWA cache PTA path expects cache rows shaped "
                f"[tokens, kv_heads, {q.shape[-1]}], got {tuple(k_ctx.shape)}"
            )
        out_block = _dspark_attention_reference(
            q_block[valid_mask],
            k_ctx,
            k_ctx,
            attn_sink,
            softmax_scale,
        )
        out[row_indices[valid_mask]] = out_block
    return out


def _get_dspark_attention_custom_op(q: torch.Tensor) -> DSparkAttentionCustomOp | None:
    if q.device.type == "cpu" or envs.VLLM_ASCEND_DSPARK_USE_PTA_REF:
        return None
    try:
        return torch.ops._C_ascend.dspark_attention
    except (AttributeError, RuntimeError):
        return None


def _maybe_call_dspark_attention_custom_op(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    draft_k: torch.Tensor,
    draft_v: torch.Tensor,
    request_slots: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    custom_op = _get_dspark_attention_custom_op(q)
    if custom_op is None:
        return None
    return custom_op(
        q,
        k_cache,
        v_cache,
        cache_positions,
        cache_valid,
        draft_k,
        draft_v,
        request_slots,
        positions,
        attn_sink,
        block_size,
        window_size,
        softmax_scale,
    )


def _get_dspark_sas_ops(q: torch.Tensor) -> tuple[Callable, Callable] | None:
    if q.device.type == "cpu" or envs.VLLM_ASCEND_DSPARK_USE_PTA_REF:
        return None
    try:
        return (
            torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
            torch.ops._C_ascend.npu_sparse_attn_sharedkv,
        )
    except (AttributeError, RuntimeError):
        return None


def dspark_attention_from_standard_cache_sas(
    q: torch.Tensor,
    standard_kv_cache: torch.Tensor | list | tuple,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    cache_block_size: int,
    softmax_scale: float,
    *,
    query_start_loc: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None = None,
    dspark_swa_indices: torch.Tensor | None = None,
    dspark_swa_lens: torch.Tensor | None = None,
    sas_metadata: torch.Tensor | None = None,
    skip_scheduling_guard: bool = False,
    raise_on_error: bool = False,
) -> torch.Tensor | None:
    """SAS fast path over standard paged SWA cache.

    The operator consumes `dspark_swa_indices` as the true visible slot list.
    The band window passed to metadata/op is only an upper-bound scheduling
    shape for this sparse path.
    """
    if q.device.type == "cpu" or query_start_loc is None or seq_lens is None:
        return None
    ops = _get_dspark_sas_ops(q)
    if ops is None:
        return None

    metadata_op, attn_op = ops
    try:
        standard_cache = _unwrap_single_kv_cache(standard_kv_cache)
        if (dspark_swa_indices is None) != (dspark_swa_lens is None):
            raise ValueError("DSpark SWA indices and lens must be provided together")
        if skip_scheduling_guard and (dspark_swa_indices is None or dspark_swa_lens is None or sas_metadata is None):
            return None
        if dspark_swa_indices is None:
            dspark_swa_indices, dspark_swa_lens = build_dspark_swa_indices(
                block_table,
                positions,
                slot_mapping,
                block_size,
                window_size,
                cache_block_size,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                token_to_req_indices=token_to_req_indices,
            )
        assert dspark_swa_lens is not None
        if skip_scheduling_guard:
            num_query_tokens = min(q.shape[0], dspark_swa_lens.shape[0])
        else:
            num_query_tokens = int(query_start_loc[-1].item()) if query_start_loc.numel() > 0 else 0
        if num_query_tokens <= 0 or num_query_tokens > q.shape[0]:
            return None

        if not skip_scheduling_guard and not _dspark_sas_lens_match_scheduling(
            dspark_swa_lens,
            query_start_loc,
            seq_lens,
            token_to_req_indices,
            block_size,
            window_size,
            num_query_tokens,
        ):
            return None

        q_active = q[:num_query_tokens]
        cu_seqlens_q = query_start_loc.to(device=q.device, dtype=torch.int32).contiguous()
        seqused_kv = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
        ori_sparse_indices = dspark_swa_indices[:num_query_tokens].to(device=q.device, dtype=torch.int32).contiguous()
        ori_block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
        sinks = attn_sink[: q_active.shape[1]].float().contiguous()
        _, ori_win_left, ori_win_right = _dspark_sparse_sas_window(block_size, window_size)

        metadata = sas_metadata
        if metadata is None:
            metadata = metadata_op(
                num_heads_q=q_active.shape[1],
                num_heads_kv=standard_cache.shape[2],
                head_dim=q_active.shape[2],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                batch_size=seq_lens.numel(),
                max_seqlen_q=block_size,
                max_seqlen_kv=int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0,
                cmp_ratio=1,
                ori_mask_mode=DSPARK_SAS_MASK_MODE,
                cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=False,
                device=str(q.device),
            )
        out_active = attn_op(
            q_active.contiguous(),
            ori_kv=standard_cache,
            ori_sparse_indices=ori_sparse_indices,
            ori_block_table=ori_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=1,
            ori_mask_mode=DSPARK_SAS_MASK_MODE,
            cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="TND",
            layout_kv="PA_ND",
        )[0]
        if num_query_tokens == q.shape[0]:
            out = out_active
        else:
            out = torch.zeros_like(q)
            out[:num_query_tokens] = out_active
        return out
    except (RuntimeError, ValueError) as err:
        if raise_on_error:
            raise
        logger.warning_once("DSpark standard-cache SAS attention failed; falling back to PTA: %s", err)
        return None


def _call_dspark_sas_block(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    block_size: int,
    window_size: int,
    metadata_op: Callable,
    attn_op: Callable,
) -> torch.Tensor:
    _, ori_win_left, ori_win_right = _dspark_sas_window(block_size, window_size)
    cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
    cu_seqlens_kv = torch.tensor([0, packed_kv.shape[0]], dtype=torch.int32, device=q.device)
    sinks = attn_sink[: q.shape[1]].float().contiguous()
    metadata = metadata_op(
        num_heads_q=q.shape[1],
        num_heads_kv=1,
        head_dim=q.shape[2],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_kv,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=None,
        batch_size=1,
        max_seqlen_q=q.shape[0],
        max_seqlen_kv=packed_kv.shape[0],
        cmp_ratio=1,
        ori_mask_mode=DSPARK_SAS_MASK_MODE,
        cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(q.device),
    )
    return attn_op(
        q,
        ori_kv=packed_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=1,
        ori_mask_mode=DSPARK_SAS_MASK_MODE,
        cmp_mask_mode=DSPARK_SAS_CMP_MASK_MODE,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q="TND",
        layout_kv="TND",
    )[0]


def _maybe_call_dspark_sas_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    draft_k: torch.Tensor,
    request_slots: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    ops = _get_dspark_sas_ops(q)
    if ops is None:
        return None
    metadata_op, attn_op = ops
    out = torch.empty_like(q)
    pos_long = positions.to(torch.long)
    request_slots_long = request_slots.to(torch.long)
    max_request_slots = k_cache.shape[0]

    try:
        for block_offset in range(0, positions.numel(), block_size):
            block_end = min(block_offset + block_size, positions.numel())
            block_pos = pos_long[block_offset:block_end]
            block_start = int(block_pos.min().item())
            context_end = block_start - 1
            context_start = max(0, context_end + 1 - window_size)
            request_slot = int(request_slots_long[block_offset].item())
            if request_slot >= max_request_slots:
                raise ValueError(
                    "DSpark request slot exceeds preallocated cache slots: "
                    f"slot={request_slot}, capacity={max_request_slots}"
                )
            k_ctx, _ = _gather_context_kv(
                k_cache,
                k_cache,
                cache_positions,
                cache_valid,
                request_slot,
                context_start,
                context_end,
            )
            packed_kv = torch.cat(
                [k_ctx[:, :1, :], draft_k[block_offset:block_end, :1, :]],
                dim=0,
            ).contiguous()
            out[block_offset:block_end] = _call_dspark_sas_block(
                q[block_offset:block_end].contiguous(),
                packed_kv,
                attn_sink,
                softmax_scale,
                block_size,
                window_size,
                metadata_op,
                attn_op,
            )
    except RuntimeError as err:
        logger.warning_once("DSpark SAS attention failed; falling back to torch attention: %s", err)
        return None
    return out


def dspark_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_positions: torch.Tensor,
    cache_valid: torch.Tensor,
    draft_k: torch.Tensor,
    draft_v: torch.Tensor,
    request_slots: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    block_size: int,
    window_size: int,
    softmax_scale: float,
    *,
    shared_kv: bool = False,
) -> torch.Tensor:
    """DSpark block attention semantic entry point.

    This is the Python reference for the future Ascend C kernel. It keeps the
    kernel-facing API explicit: request-local rolling context cache plus the
    current non-causal draft block.
    """
    if positions.numel() == 0:
        return torch.empty_like(q)
    custom_out = _maybe_call_dspark_attention_custom_op(
        q,
        k_cache,
        v_cache,
        cache_positions,
        cache_valid,
        draft_k,
        draft_v,
        request_slots,
        positions,
        attn_sink,
        block_size,
        window_size,
        softmax_scale,
    )
    if custom_out is not None:
        return custom_out
    if request_slots.numel() != positions.numel():
        raise ValueError(
            "DSpark request_slots length must match query positions: "
            f"request_slots={request_slots.numel()}, positions={positions.numel()}"
        )
    if request_slots.device.type == "cpu":
        _validate_query_block_slots(request_slots, block_size)

    if shared_kv:
        sas_out = _maybe_call_dspark_sas_attention(
            q,
            k_cache,
            cache_positions,
            cache_valid,
            draft_k,
            request_slots,
            positions,
            attn_sink,
            block_size,
            window_size,
            softmax_scale,
        )
        if sas_out is not None:
            return sas_out

    out = torch.empty_like(q)
    pos_long = positions.to(torch.long)
    request_slots_long = request_slots.to(torch.long)
    max_request_slots = k_cache.shape[0]

    for block_offset in range(0, positions.numel(), block_size):
        block_end = min(block_offset + block_size, positions.numel())
        block_pos = pos_long[block_offset:block_end]
        block_start = int(block_pos.min().item())
        context_end = block_start - 1
        context_start = max(0, context_end + 1 - window_size)
        request_slot = int(request_slots_long[block_offset].item())
        if request_slot >= max_request_slots:
            raise ValueError(
                "DSpark request slot exceeds preallocated cache slots: "
                f"slot={request_slot}, capacity={max_request_slots}"
            )

        k_ctx, v_ctx = _gather_context_kv(
            k_cache,
            v_cache,
            cache_positions,
            cache_valid,
            request_slot,
            context_start,
            context_end,
        )
        if shared_kv:
            v_ctx = k_ctx
            draft_v_block = draft_k[block_offset:block_end]
        else:
            draft_v_block = draft_v[block_offset:block_end]
        k_ctx = torch.cat([k_ctx, draft_k[block_offset:block_end]], dim=0)
        v_ctx = torch.cat([v_ctx, draft_v_block], dim=0)
        out[block_offset:block_end] = _dspark_attention_reference(
            q[block_offset:block_end],
            k_ctx,
            v_ctx,
            attn_sink,
            softmax_scale,
        )

    return out


__all__ = [
    "build_dspark_swa_indices",
    "dspark_attention",
    "dspark_attention_from_standard_cache",
    "dspark_attention_from_standard_cache_sas",
]
