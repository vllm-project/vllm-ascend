import gc

import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_and_dspark_inputs_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.spec_decode.dflash_proposer import (
    _COPY_EXPAND_TILE_SIZE,
    _compute_num_programs,
)

PARALLEL_DRAFTING_TOKEN_ID = 151643
KV_BLOCK_SIZE = 128


def copy_and_expand_dflash_dspark_ref(
    next_token_ids,
    target_positions,
    context_slot_mapping,
    block_table,
    query_start_loc,
    seq_lens,
    num_rejected_tokens,
    num_query_per_req,
    num_speculative_tokens,
    sample_from_anchor,
):
    """Pure-PyTorch reference of the serial kernel logic (golden)."""
    batch_size = next_token_ids.shape[0]
    num_context = target_positions.shape[0]
    num_query_total = batch_size * num_query_per_req
    num_sample_total = batch_size * num_speculative_tokens
    device = next_token_ids.device

    out_input_ids = torch.empty(num_query_total, dtype=torch.int32, device=device)
    out_context_positions = torch.empty(num_context, dtype=torch.int32, device=device)
    out_query_positions = torch.empty(num_query_total, dtype=torch.int32, device=device)
    out_context_slot_mapping = torch.empty(num_context, dtype=torch.int32, device=device)
    out_query_slot_mapping = torch.empty(num_query_total, dtype=torch.int32, device=device)
    out_token_indices = torch.zeros(num_sample_total, dtype=torch.int32, device=device)

    for req_idx in range(batch_size):
        ctx_start = query_start_loc[req_idx].item()
        ctx_end = query_start_loc[req_idx + 1].item()

        out_context_positions[ctx_start:ctx_end] = target_positions[ctx_start:ctx_end]
        out_context_slot_mapping[ctx_start:ctx_end] = context_slot_mapping[ctx_start:ctx_end]

        if num_rejected_tokens is not None:
            num_rejected = num_rejected_tokens[req_idx].item()
            valid_ctx_end = ctx_end - num_rejected
        else:
            num_rejected = 0
            valid_ctx_end = ctx_end

        seq_len = seq_lens[req_idx].item()
        effective_seq_len = seq_len - num_rejected
        last_pos = target_positions[valid_ctx_end - 1].item()

        for q_idx in range(num_query_per_req):
            query_pos = last_pos + 1 + q_idx
            query_out_idx = req_idx * num_query_per_req + q_idx

            out_query_positions[query_out_idx] = query_pos

            query_cache_pos = effective_seq_len + q_idx
            block_num_q = query_cache_pos // KV_BLOCK_SIZE
            block_id_q = block_table[req_idx, block_num_q].to(torch.int64).item()
            slot_q = block_id_q * KV_BLOCK_SIZE + (query_cache_pos % KV_BLOCK_SIZE)
            out_query_slot_mapping[query_out_idx] = slot_q

            if q_idx == 0:
                out_input_ids[query_out_idx] = next_token_ids[req_idx]
            else:
                out_input_ids[query_out_idx] = PARALLEL_DRAFTING_TOKEN_ID

            if sample_from_anchor:
                sample_out_idx = req_idx * num_speculative_tokens + q_idx
                out_token_indices[sample_out_idx] = query_out_idx
            elif q_idx > 0:
                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                out_token_indices[sample_out_idx] = query_out_idx

    return {
        "out_input_ids": out_input_ids,
        "out_context_positions": out_context_positions,
        "out_query_positions": out_query_positions,
        "out_context_slot_mapping": out_context_slot_mapping,
        "out_query_slot_mapping": out_query_slot_mapping,
        "out_token_indices": out_token_indices,
    }


# (batch_size, ctx_lens, num_spec, sample_from_anchor, has_num_rejected)
CONFIGS = [
    (1, [4], 3, False, False),
    (64, [4] * 64, 3, False, False),
    (256, [4] * 256, 3, False, False),
    (1, [2048], 3, False, False),
    (4, [1024] * 4, 3, False, False),
    (8, [512] * 8, 3, False, False),
    (64, [4] * 64, 3, True, False),
    (64, [4] * 64, 3, False, True),
    (8, [512] * 8, 3, True, True),
]


@pytest.mark.parametrize("batch_size,ctx_lens,num_spec,sample_from_anchor,has_num_rejected", CONFIGS)
def test_copy_and_expand_dflash_dspark(batch_size, ctx_lens, num_spec, sample_from_anchor, has_num_rejected):
    init_device_properties_triton()
    device = "npu"
    torch.manual_seed(0)

    num_query_per_req = num_spec if sample_from_anchor else 1 + num_spec
    num_query_total = batch_size * num_query_per_req
    num_sample_total = batch_size * num_spec

    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.cumsum(ctx, dim=0)
    total_ctx = int(query_start_loc[-1].item())

    history = torch.randint(0, 100, (batch_size,), dtype=torch.int32, device=device)
    seq_lens = ctx + history
    target_positions = torch.cat(
        [
            torch.arange(
                seq_lens[i].item() - ctx[i].item(),
                seq_lens[i].item(),
                dtype=torch.int32,
                device=device,
            )
            for i in range(batch_size)
        ]
    )
    max_blocks = int((seq_lens.max() + num_query_per_req + KV_BLOCK_SIZE) // KV_BLOCK_SIZE) + 2
    block_table = torch.randint(1, 10000, (batch_size, max_blocks), dtype=torch.int32, device=device)

    if has_num_rejected:
        num_rejected_tokens = torch.minimum(
            torch.randint(0, num_spec + 1, (batch_size,), dtype=torch.int32, device=device),
            ctx - 1,
        ).clamp(min=0)
    else:
        num_rejected_tokens = None

    next_token_ids = torch.randint(0, 150000, (batch_size,), dtype=torch.int32, device=device)
    context_slot_mapping = torch.randint(0, 1 << 30, (total_ctx,), dtype=torch.int32, device=device)

    # Run PyTorch reference
    ref = copy_and_expand_dflash_dspark_ref(
        next_token_ids,
        target_positions,
        context_slot_mapping,
        block_table,
        query_start_loc,
        seq_lens,
        num_rejected_tokens,
        num_query_per_req,
        num_spec,
        sample_from_anchor,
    )

    # Run Triton kernel
    out_input_ids = torch.empty(num_query_total, dtype=torch.int32, device=device)
    out_context_positions = torch.empty(total_ctx, dtype=torch.int32, device=device)
    out_query_positions = torch.empty(num_query_total, dtype=torch.int32, device=device)
    out_context_slot_mapping = torch.empty(total_ctx, dtype=torch.int32, device=device)
    out_query_slot_mapping = torch.empty(num_query_total, dtype=torch.int32, device=device)
    out_token_indices = torch.zeros(num_sample_total, dtype=torch.int32, device=device)

    grid = (_compute_num_programs(total_ctx, num_query_total),)

    copy_and_expand_dflash_and_dspark_inputs_kernel[grid](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        context_slot_mapping_ptr=context_slot_mapping,
        out_input_ids_ptr=out_input_ids,
        out_context_positions_ptr=out_context_positions,
        out_query_positions_ptr=out_query_positions,
        out_context_slot_mapping_ptr=out_context_slot_mapping,
        out_query_slot_mapping_ptr=out_query_slot_mapping,
        out_token_indices_ptr=out_token_indices,
        block_table_ptr=block_table,
        block_table_stride=max_blocks,
        query_start_loc_ptr=query_start_loc,
        seq_lens_ptr=seq_lens,
        num_rejected_tokens_ptr=(num_rejected_tokens if num_rejected_tokens is not None else 0),
        parallel_drafting_token_id=PARALLEL_DRAFTING_TOKEN_ID,
        block_size=KV_BLOCK_SIZE,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_spec,
        total_input_tokens=total_ctx,
        batch_size=batch_size,
        HAS_NUM_REJECTED=has_num_rejected,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        TILE_SIZE=_COPY_EXPAND_TILE_SIZE,
    )

    torch.testing.assert_close(out_input_ids, ref["out_input_ids"])
    torch.testing.assert_close(out_context_positions, ref["out_context_positions"])
    torch.testing.assert_close(out_query_positions, ref["out_query_positions"])
    torch.testing.assert_close(out_context_slot_mapping, ref["out_context_slot_mapping"])
    torch.testing.assert_close(out_query_slot_mapping, ref["out_query_slot_mapping"])
    torch.testing.assert_close(out_token_indices, ref["out_token_indices"])

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
