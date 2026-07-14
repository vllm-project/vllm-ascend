# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch


def _get_dspark_draft_hf_config(vllm_config: Any) -> Any | None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    return getattr(draft_model_config, "hf_config", None)


def get_dspark_query_block_size(vllm_config: Any) -> int:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = getattr(speculative_config, "num_speculative_tokens", None)
    if num_speculative_tokens:
        return int(num_speculative_tokens)

    draft_hf_config = _get_dspark_draft_hf_config(vllm_config)
    return int(getattr(draft_hf_config, "dspark_block_size", 0) or 0)


def is_dspark_noncausal_draft(vllm_config: Any, common_attn_metadata: Any) -> bool:
    if getattr(common_attn_metadata, "causal", True):
        return False

    speculative_config = getattr(vllm_config, "speculative_config", None)
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark):
        return bool(use_dspark())

    draft_hf_config = _get_dspark_draft_hf_config(vllm_config)
    return bool(getattr(draft_hf_config, "dspark_block_size", 0))


def get_draft_swa_window(vllm_config: Any) -> tuple[int, int]:
    window_size = int(vllm_config.model_config.hf_config.sliding_window)
    return window_size - 1, 0


def get_dspark_sparse_sas_window(vllm_config: Any) -> tuple[int, int]:
    window_size = int(vllm_config.model_config.hf_config.sliding_window)
    query_block_size = get_dspark_query_block_size(vllm_config)
    return window_size + query_block_size - 1, 0


def _valid_slot_rows(slot_mapping: torch.Tensor) -> torch.Tensor:
    if slot_mapping.ndim == 1:
        return slot_mapping >= 0
    return torch.all(slot_mapping >= 0, dim=-1)


def _aligned_index_width(window_size: int, query_block_size: int) -> int:
    visible_upper_bound = window_size + query_block_size
    return ((visible_upper_bound + 127) // 128) * 128


def build_dspark_swa_indices(
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_block_size: int,
    window_size: int,
    cache_block_size: int,
    num_query_tokens: int,
    token_to_req_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build physical slots for history-window plus full draft-block visibility."""
    index_width = _aligned_index_width(window_size, query_block_size)
    device = block_table.device
    indices = torch.full((num_query_tokens, 1, index_width), -1, dtype=torch.int32, device=device)
    lens = torch.zeros(num_query_tokens, dtype=torch.int32, device=device)
    if num_query_tokens == 0:
        return indices, lens
    if slot_mapping is not None and slot_mapping.shape[0] < num_query_tokens:
        raise ValueError("DSpark SWA slot_mapping must cover every query token")
    if token_to_req_indices is not None and token_to_req_indices.numel() < num_query_tokens:
        raise ValueError("DSpark SWA token_to_req_indices must cover every query token")

    num_reqs = min(max(query_start_loc.numel() - 1, 0), block_table.shape[0], seq_lens.numel())
    if num_reqs == 0 or block_table.shape[1] == 0:
        return indices, lens

    query_start_loc = query_start_loc[: num_reqs + 1].to(device=device, dtype=torch.long)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    seq_lens = seq_lens[:num_reqs].to(device=device, dtype=torch.long)
    visible_starts = torch.clamp(seq_lens - query_lens - window_size, min=0)
    visible_lens = torch.clamp(seq_lens - visible_starts, min=0, max=index_width)

    offsets = torch.arange(index_width, dtype=torch.long, device=device)
    visible_positions = visible_starts[:, None] + offsets[None, :]
    visible_mask = offsets[None, :] < visible_lens[:, None]
    logical_blocks = torch.div(visible_positions, cache_block_size, rounding_mode="floor")
    logical_blocks = logical_blocks.clamp(max=block_table.shape[1] - 1)
    physical_blocks = block_table[:num_reqs].to(device=device, dtype=torch.long).gather(1, logical_blocks)
    physical_slots = physical_blocks * cache_block_size + visible_positions % cache_block_size
    physical_slots.masked_fill_(~visible_mask, -1)

    if token_to_req_indices is None:
        token_rows = torch.arange(num_query_tokens, dtype=torch.long, device=device)
        token_to_req = torch.bucketize(token_rows, query_start_loc[1:], right=True)
    else:
        token_to_req = token_to_req_indices[:num_query_tokens].to(device=device, dtype=torch.long)

    valid_rows = (token_to_req >= 0) & (token_to_req < num_reqs)
    if slot_mapping is not None:
        valid_rows &= _valid_slot_rows(slot_mapping[:num_query_tokens].to(device=device))
    row_requests = token_to_req.clamp(0, num_reqs - 1)
    indices[:, 0] = physical_slots.index_select(0, row_requests).to(torch.int32)
    lens.copy_(visible_lens.index_select(0, row_requests).to(torch.int32))
    indices.masked_fill_(~valid_rows[:, None, None], -1)
    lens.masked_fill_(~valid_rows, 0)
    return indices, lens


def build_dspark_swa_metadata_for_drafting(
    vllm_config: Any,
    common_attn_metadata: Any,
    slot_mapping: torch.Tensor | None,
    cache_block_size: int,
) -> tuple[int, int, torch.Tensor | None, torch.Tensor | None]:
    if not is_dspark_noncausal_draft(vllm_config, common_attn_metadata):
        ori_win_left, ori_win_right = get_draft_swa_window(vllm_config)
        return ori_win_left, ori_win_right, None, None

    num_reqs = int(getattr(common_attn_metadata, "num_reqs", 0) or 0)
    num_query_tokens = int(getattr(common_attn_metadata, "num_input_tokens", 0) or 0)
    if num_query_tokens == 0:
        num_query_tokens = int(getattr(common_attn_metadata, "num_actual_tokens", 0) or 0)
    query_block_size = get_dspark_query_block_size(vllm_config)
    if query_block_size <= 0:
        raise ValueError("DSpark noncausal drafting requires a positive query block size")
    block_table = getattr(common_attn_metadata, "block_table_tensor", None)
    if block_table is None:
        raise ValueError("DSpark noncausal drafting requires a paged SWA block table")
    indices, lens = build_dspark_swa_indices(
        block_table[:num_reqs],
        slot_mapping,
        common_attn_metadata.query_start_loc[: num_reqs + 1],
        common_attn_metadata.seq_lens[:num_reqs],
        query_block_size,
        int(vllm_config.model_config.hf_config.sliding_window),
        int(cache_block_size),
        num_query_tokens,
        getattr(common_attn_metadata, "token_to_req_indices", None),
    )
    ori_win_left, ori_win_right = get_dspark_sparse_sas_window(vllm_config)
    return ori_win_left, ori_win_right, indices, lens
