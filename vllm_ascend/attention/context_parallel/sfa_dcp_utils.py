from typing import Any

import torch
from vllm.utils.math_utils import cdiv


def get_sfa_dcp_max_local_block_table_cols(
    max_model_len: int,
    physical_block_size: int,
    dcp_size: int,
    blocks_per_physical_block: int,
) -> int:
    """Return the local logical-block width of an SFA DCP block table."""
    return cdiv(max_model_len, physical_block_size * dcp_size) * blocks_per_physical_block


def get_sfa_dcp_local_block_table(
    block_table: torch.Tensor,
    num_reqs: int,
    max_local_block_table_cols: int,
) -> torch.Tensor:
    """Select the rank-local physical block-table view."""
    local_cols = min(block_table.shape[1], max_local_block_table_cols)
    return block_table[:num_reqs, :local_cols]


def build_sfa_dcp_replicated_block_table(
    dcp_block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table_output: torch.Tensor,
    replicated_col_idx: torch.Tensor,
    dcp_size: int,
    blocks_per_physical_block: int,
) -> torch.Tensor:
    """Expand a local SFA block table into replicated-indexer addresses."""
    local_col_idx = (
        replicated_col_idx // (dcp_size * blocks_per_physical_block) * blocks_per_physical_block
        + replicated_col_idx % blocks_per_physical_block
    )
    rank_in_replicated_view = (replicated_col_idx // blocks_per_physical_block) % dcp_size

    local_logical_blocks = torch.index_select(dcp_block_table, 1, local_col_idx)
    if blocks_per_physical_block == 1:
        replicated_blocks = local_logical_blocks * dcp_size + rank_in_replicated_view
    else:
        local_sub_blocks = local_logical_blocks % blocks_per_physical_block
        local_physical_blocks = local_logical_blocks // blocks_per_physical_block
        replicated_blocks = (
            local_physical_blocks * dcp_size + rank_in_replicated_view
        ) * blocks_per_physical_block + local_sub_blocks

    valid_req_mask = (
        (seq_lens[: dcp_block_table.shape[0]].to(device=dcp_block_table.device) > 0)
        .to(replicated_blocks.dtype)
        .view(-1, 1)
    )
    block_table_output.copy_(replicated_blocks * valid_req_mask)
    return block_table_output


def build_sfa_dcp_replicated_slot_mapping(
    common_attn_metadata: Any,
    block_table_replicated_view: torch.Tensor,
    slot_mapping_output: torch.Tensor,
    replicated_view_block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build slots that address the physically replicated indexer cache."""
    num_reqs = common_attn_metadata.num_reqs
    num_input_tokens = common_attn_metadata.num_input_tokens
    num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens)
    slot_mapping_output.fill_(-1)
    if num_actual_tokens == 0:
        return slot_mapping_output

    query_lens = (
        common_attn_metadata.query_start_loc[1 : num_reqs + 1] - common_attn_metadata.query_start_loc[:num_reqs]
    )
    req_indices = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        query_lens.to(device=device),
        output_size=num_input_tokens,
    )[:num_actual_tokens]
    if req_indices.numel() == 0:
        return slot_mapping_output

    num_actual_tokens = min(num_actual_tokens, req_indices.shape[0])
    req_indices = req_indices[:num_actual_tokens]
    positions = common_attn_metadata.positions[:num_actual_tokens].to(
        device=device,
        dtype=torch.int32,
    )
    logical_block_idx = positions // replicated_view_block_size
    block_offsets = positions % replicated_view_block_size
    block_table_indices = req_indices * block_table_replicated_view.shape[1] + logical_block_idx
    block_numbers = block_table_replicated_view.flatten()[block_table_indices]
    slot_mapping_output[:num_actual_tokens] = block_numbers * replicated_view_block_size + block_offsets
    return slot_mapping_output
