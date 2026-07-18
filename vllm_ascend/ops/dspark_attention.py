# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


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


__all__ = ["build_dspark_swa_indices"]
