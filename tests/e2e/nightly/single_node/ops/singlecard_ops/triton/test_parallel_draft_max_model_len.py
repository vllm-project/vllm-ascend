import torch
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def test_dspark_disables_a_whole_query_window_past_max_model_len():
    init_device_properties_triton()
    device = "npu"
    block_size = 4
    num_query_per_req = 5
    num_speculative_tokens = 5
    batch_size = 2

    # Request 0 fits exactly: 3 + 5 == max_model_len. Request 1 does not:
    # 6 + 5 > max_model_len, so every draft slot in its non-causal block must
    # be padding rather than preserving a partial in-range prefix.
    next_token_ids = torch.tensor([42, 43], dtype=torch.int64, device=device)
    target_positions = torch.tensor([2, 5], dtype=torch.int32, device=device)
    context_slot_mapping = torch.tensor([41, 121], dtype=torch.int32, device=device)
    block_table = torch.tensor([[10, 20], [30, 40]], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([3, 6], dtype=torch.int32, device=device)
    request_window_ok = torch.tensor([True, False], dtype=torch.bool, device=device)

    num_query_total = batch_size * num_query_per_req
    input_ids = torch.empty(num_query_total, dtype=torch.int64, device=device)
    context_positions = torch.empty(batch_size, dtype=torch.int32, device=device)
    query_positions = torch.empty(num_query_total, dtype=torch.int32, device=device)
    context_slots = torch.empty(batch_size, dtype=torch.int32, device=device)
    query_slots = torch.empty(num_query_total, dtype=torch.int32, device=device)
    token_indices = torch.empty(
        batch_size * num_speculative_tokens,
        dtype=torch.int32,
        device=device,
    )

    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        context_slot_mapping_ptr=context_slot_mapping,
        out_input_ids_ptr=input_ids,
        out_context_positions_ptr=context_positions,
        out_query_positions_ptr=query_positions,
        out_context_slot_mapping_ptr=context_slots,
        out_query_slot_mapping_ptr=query_slots,
        out_token_indices_ptr=token_indices,
        block_table_ptr=block_table,
        block_table_stride=block_table.stride(0),
        query_start_loc_ptr=query_start_loc,
        seq_lens_ptr=seq_lens,
        num_rejected_tokens_ptr=0,
        parallel_drafting_token_id=0,
        block_size=block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_speculative_tokens,
        total_input_tokens=batch_size,
        batch_size=batch_size,
        request_window_ok_ptr=request_window_ok,
        padding_slot_id=PADDING_SLOT_ID,
        HAS_NUM_REJECTED=False,
        SAMPLE_FROM_ANCHOR=True,
        CHECK_REQUEST_WINDOW=True,
    )

    torch.npu.synchronize()
    torch.testing.assert_close(
        input_ids.cpu(),
        torch.tensor([42, 0, 0, 0, 0, 43, 0, 0, 0, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(context_positions.cpu(), target_positions.cpu())
    torch.testing.assert_close(context_slots.cpu(), context_slot_mapping.cpu())
    torch.testing.assert_close(
        query_positions.cpu(),
        torch.tensor([3, 4, 5, 6, 7, 0, 0, 0, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        query_slots.cpu(),
        torch.tensor([43, 80, 81, 82, 83, -1, -1, -1, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(token_indices.cpu(), torch.arange(10, dtype=torch.int32))
