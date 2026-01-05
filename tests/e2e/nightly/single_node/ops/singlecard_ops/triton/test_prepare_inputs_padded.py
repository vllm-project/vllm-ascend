import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.spec_decode.utils import \
    prepare_inputs_padded_kernel
from vllm_ascend.ops.triton.triton_utils import (get_vectorcore_num,
                                                 init_device_properties_triton)
from vllm_ascend.spec_decode.eagle_proposer import \
    _PREPARE_INPUTS_BLOCK_SIZE as BLOCK_SIZE


def prepare_inputs_padded_ref(
    cu_num_draft_tokens,
    valid_sampled_tokens_count,
    query_start_loc,
):
    num_draft_tokens = torch.cat([
        cu_num_draft_tokens[0:1],
        cu_num_draft_tokens[1:] - cu_num_draft_tokens[:-1],
    ])

    num_rejected_tokens = torch.where(
        num_draft_tokens > 0,
        num_draft_tokens + 1 - valid_sampled_tokens_count,
        torch.zeros_like(num_draft_tokens),
    )

    token_indices_to_sample = query_start_loc[1:] - 1 - num_rejected_tokens

    return token_indices_to_sample.to(torch.int32)


@pytest.mark.parametrize("num_reqs", [1, 7, 32, 128, 2048])
def test_prepare_inputs_padded(num_reqs):
    init_device_properties_triton()
    device = "npu"
    torch.manual_seed(0)

    draft_lens = torch.randint(1,
                               6, (num_reqs, ),
                               device=device,
                               dtype=torch.int32)

    cu_num_draft_tokens = torch.cumsum(draft_lens, dim=0).to(torch.int32)

    valid_sampled_tokens_count = torch.zeros_like(draft_lens)
    for i in range(num_reqs):
        valid_sampled_tokens_count[i] = torch.randint(0, draft_lens[i] + 2,
                                                      (1, )).item()

    seq_lens = draft_lens + 1
    query_start_loc = torch.zeros(num_reqs + 1,
                                  device=device,
                                  dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(seq_lens, dim=0)

    # Run PyTorch reference
    out_ref = prepare_inputs_padded_ref(cu_num_draft_tokens,
                                        valid_sampled_tokens_count,
                                        query_start_loc)

    # Run Triton kernel
    out_tri = torch.empty(num_reqs, dtype=torch.int32, device=device)

    num_blocks_needed = triton.cdiv(num_reqs, BLOCK_SIZE)
    num_vector_core = get_vectorcore_num()
    grid_size = min(num_blocks_needed, num_vector_core)
    grid = (grid_size, )

    prepare_inputs_padded_kernel[grid](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        out_tri,
        num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    torch.testing.assert_close(out_tri, out_ref)
