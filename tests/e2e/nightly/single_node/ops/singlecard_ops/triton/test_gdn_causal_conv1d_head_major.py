import pytest
import torch

from vllm_ascend.utils import enable_custom_op

PAD_SLOT_ID = -1


@pytest.mark.parametrize("spec_update", [False, True])
def test_gdn_causal_conv1d_head_major_output(spec_update):
    torch.manual_seed(0)
    enable_custom_op()

    heads, tokens, head_dim, width = 7, 13, 16, 4
    dim = heads * head_dim
    x = torch.randn(tokens, dim, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(width, dim, device="npu", dtype=torch.bfloat16)
    flat_state = torch.zeros((1, width - 1 + tokens, dim), device="npu", dtype=torch.bfloat16)
    head_major_state = flat_state.clone()
    flat_output = torch.empty_like(x)
    head_major_output = torch.empty((heads, tokens, head_dim), device="npu", dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, tokens], device="npu", dtype=torch.int32)
    cache_indices = torch.zeros(1, device="npu", dtype=torch.int32)
    num_accepted_tokens = torch.tensor([1], device="npu", dtype=torch.int32) if spec_update else None

    common_args = dict(
        weight=weight,
        bias_opt=None,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=0,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=1,
    )
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        flat_output,
        x,
        conv_state=flat_state,
        head_num=0,
        **common_args,
    )
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        head_major_output,
        x,
        conv_state=head_major_state,
        head_num=heads,
        **common_args,
    )

    torch.testing.assert_close(
        head_major_output.movedim(0, 1).reshape_as(flat_output),
        flat_output,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(head_major_state, flat_state, rtol=1e-2, atol=1e-2)
