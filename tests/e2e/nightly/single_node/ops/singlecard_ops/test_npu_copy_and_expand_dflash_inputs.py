import numpy as np
import pytest
import torch
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

def golden_copy_and_expand_dflash_inputs_torch(
    # Inputs
    next_token_ids,  # [num_reqs]
    target_positions,  # [num_context]
    # Outputs
    out_input_ids,  # [num_query_total] (output)
    out_context_positions,  # [num_context] (output)
    out_query_positions,  # [num_query_total] (output)
    out_context_slot_mapping,  # [num_context] (output)
    out_query_slot_mapping,  # [num_query_total] (output)
    out_token_indices,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table,  # [max_reqs, max_blocks]
    # Metadata
    query_start_loc,  # [num_reqs + 1]
    num_rejected_tokens,  # [num_reqs] or None when not padded
    # Scalars
    parallel_drafting_token_id,  # int
    block_size,  # int
    num_query_per_req,  # int
    num_speculative_tokens,  # int
    total_input_tokens,  # int
    batch_size,  # int
):
    """Golden reference pytorch native implementation of copy_and_expand_dflash_inputs."""

    # Initialize metadata
    has_num_rejected = num_rejected_tokens is not None

    for req_idx in range(batch_size):
        # Load context token range for this request
        ctx_start = query_start_loc[req_idx]
        ctx_end = query_start_loc[req_idx + 1]
        num_ctx = ctx_end - ctx_start
        total_tokens = num_ctx + num_query_per_req

        # Process each token in the sequence
        for j in range(total_tokens):
            in_bounds = True
            is_ctx = j < num_ctx
            is_query = not is_ctx

            if is_ctx:
                # --- Context positions ---
                ctx_pos_idx = min(ctx_start + j, total_input_tokens - 1)
                ctx_pos = target_positions[ctx_pos_idx]
                positions = ctx_pos

                # Store context position
                out_context_positions[ctx_start + j] = ctx_pos

            else:
                # --- Query positions ---
                query_off = j - num_ctx  # offset within query portion (0-indexed)

                # Query: last_valid_pos + 1 + query_off
                # In padded mode, ctx_end includes rejected tokens; use valid_ctx_end
                # to find the last accepted context position.
                if has_num_rejected:
                    num_rejected = num_rejected_tokens[req_idx]
                    valid_ctx_end = ctx_end - num_rejected
                else:
                    valid_ctx_end = ctx_end

                last_pos = target_positions[valid_ctx_end - 1]
                query_pos = last_pos + 1 + query_off
                positions = query_pos

                # Store query position
                query_out = req_idx * num_query_per_req + query_off
                out_query_positions[query_out] = query_pos

                # --- Input IDs (query tokens only) ---
                bonus_token = next_token_ids[req_idx]
                is_bonus = query_off == 0
                if is_bonus:
                    input_id = bonus_token
                else:
                    input_id = parallel_drafting_token_id

                out_input_ids[query_out] = input_id

                # --- Token indices to sample (mask tokens, skip the bonus token) ---
                is_sample = query_off > 0
                if is_sample:
                    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
                    out_token_indices[sample_out_idx] = query_out

            # --- Slot mapping (block_table lookup for all positions) ---
            block_num = positions // block_size
            # Clamp block_number to avoid OOB when position is at max
            block_num = min(block_num, block_table.size(1) - 1)
            block_id = block_table[req_idx, block_num]
            slot = block_id * block_size + (positions % block_size)

            if is_ctx:
                out_context_slot_mapping[ctx_start + j] = slot
            else:
                query_out = req_idx * num_query_per_req + (j - num_ctx)
                out_query_slot_mapping[query_out] = slot

def generate_test_case(num_reqs, num_speculative_tokens, block_size, max_tokens_per_req=64):
    rng = np.random.default_rng(42)

    num_reqs_prefill = num_reqs // 2
    num_reqs_decode = num_reqs - num_reqs_prefill

    next_token_ids = rng.integers(1, 50000, size=num_reqs, dtype=np.int32)

    query_lens = rng.integers(1, max_tokens_per_req, size=num_reqs)
    query_lens[:num_reqs_decode] = 1 + num_speculative_tokens
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(query_lens)

    seq_lens = rng.integers(1, max_tokens_per_req, size=num_reqs)
    seq_lens = np.maximum(seq_lens, query_lens)

    num_rejected_tokens = rng.integers(0, num_speculative_tokens, size=num_reqs)
    num_rejected_tokens[num_reqs_decode: ] = 0
    num_rejected_tokens = np.minimum(num_rejected_tokens, query_lens)

    total_input_tokens = query_start_loc[-1]
    target_positions = np.zeros(total_input_tokens, dtype=np.int32)

    for i in range(num_reqs):
        last_pos = seq_lens[i]
        first_pos = seq_lens[i] - query_lens[i]
        target_positions[query_start_loc[i]: query_start_loc[i+1]] = np.arange(first_pos, last_pos, dtype=np.int32)

    max_blocks = (np.max(seq_lens) + block_size - 1) // block_size
    num_blocks = num_reqs * max_blocks
    block_table_tensor = np.arange(0, num_blocks, dtype=np.int32).reshape(num_reqs, max_blocks)

    return (
            torch.from_numpy(next_token_ids).to(torch.int32).npu(),
            torch.from_numpy(target_positions).to(torch.int64).npu(),
            torch.from_numpy(query_start_loc).to(torch.int32).npu(),
            torch.from_numpy(num_rejected_tokens).to(torch.int32).npu(),
            torch.from_numpy(block_table_tensor).to(torch.int32).npu()
            )


@pytest.mark.parametrize("num_reqs", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("num_speculative_tokens", [3, 7, 15])
def test_copy_and_expand_dflash_inputs(num_reqs, num_speculative_tokens):
    block_size = 128

    next_token_ids, target_positions, query_start_loc, num_rejected_tokens, block_table_tensor = generate_test_case(
        num_reqs, num_speculative_tokens, block_size, 64)

    parallel_drafting_token_id = 151669
    num_speculative_tokens = 15

    num_query_per_req = num_speculative_tokens + 1

    num_query_total = num_reqs * (1 + num_speculative_tokens)
    num_context = query_start_loc[-1]

    # Prepare output tensors for Ascend C operator (in-place)
    out_input_ids = torch.zeros(num_query_total, device='npu:0', dtype=torch.int32)
    out_context_positions = torch.zeros(num_context, device='npu:0', dtype=torch.int32)
    out_query_positions = torch.zeros(num_query_total, device='npu:0', dtype=torch.int32)
    out_context_slot_mapping = torch.zeros(num_context, device='npu:0', dtype=torch.int32)
    out_query_slot_mapping = torch.zeros(num_query_total, device='npu:0', dtype=torch.int32)
    out_token_indices = torch.zeros(num_reqs * num_speculative_tokens, device='npu:0', dtype=torch.int32)

    # Call Ascend C operator
    torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs(
        # Inputs
        next_token_ids,
        target_positions,
        query_start_loc,
        num_rejected_tokens,
        block_table_tensor,
        # Scalars
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_tokens,
        block_size,
        # In-place outputs
        out_input_ids,
        out_context_positions,
        out_query_positions,
        out_context_slot_mapping,
        out_query_slot_mapping,
        out_token_indices
    )

    torch_out_input_ids = torch.zeros(num_query_total, device='npu:0', dtype=torch.int32)
    torch_out_context_positions = torch.zeros(num_context, device='npu:0', dtype=torch.int32)
    torch_out_query_positions = torch.zeros(num_query_total, device='npu:0', dtype=torch.int32)
    torch_out_context_slot_mapping = torch.zeros(num_context, device='npu:0', dtype=torch.int32)
    torch_out_query_slot_mapping = torch.zeros(num_query_total, device='npu:0', dtype=torch.int32)
    torch_out_token_indices = torch.zeros(num_reqs * num_speculative_tokens, device='npu:0', dtype=torch.int32)

    # Call PyTorch native implementation
    golden_copy_and_expand_dflash_inputs_torch(
        # Inputs
        next_token_ids,
        target_positions,
        # Outputs
        torch_out_input_ids,
        torch_out_context_positions,
        torch_out_query_positions,
        torch_out_context_slot_mapping,
        torch_out_query_slot_mapping,
        torch_out_token_indices,
        # Block table
        block_table_tensor,
        # Metadata
        query_start_loc,
        num_rejected_tokens,
        # Scalars
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_tokens,
        num_context,
        num_reqs
    )

    # Check if outputs are equal
    input_ids_match = torch.equal(out_input_ids, torch_out_input_ids)
    context_positions_match = torch.equal(out_context_positions, torch_out_context_positions)
    query_positions_match = torch.equal(out_query_positions, torch_out_query_positions)
    context_slot_mapping_match = torch.equal(out_context_slot_mapping, torch_out_context_slot_mapping)
    query_slot_mapping_match = torch.equal(out_query_slot_mapping, torch_out_query_slot_mapping)
    token_indices_match = torch.equal(out_token_indices, torch_out_token_indices)

    all_match = (input_ids_match and context_positions_match and
                 query_positions_match and context_slot_mapping_match and
                 query_slot_mapping_match and token_indices_match)

    assert all_match
