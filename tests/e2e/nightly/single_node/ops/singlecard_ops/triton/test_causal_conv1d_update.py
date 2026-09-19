import pytest
import torch

from vllm_ascend._310p.ops.causal_conv1d import (
    causal_conv1d_update as causal_conv1d_update_ref,
)
from vllm_ascend.ops.triton.mamba.causal_conv1d import PAD_SLOT_ID
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("itype", [torch.bfloat16])
def test_causal_conv1d_varlen_update_max_query_len(itype):
    torch.random.manual_seed(0)
    enable_custom_op()

    device = "npu"
    dim, width = 16, 3
    query_start_loc = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
    cache_indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
    x = torch.randn(5, dim, device=device, dtype=itype)
    weight = torch.randn(width, dim, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype)
    conv_states = torch.randn(3, width - 1, dim, device=device, dtype=itype)
    conv_states_ref = conv_states.detach().cpu().clone()

    out_ref = causal_conv1d_update_ref(
        x.detach().cpu(),
        conv_states_ref,
        weight.detach().cpu(),
        bias=bias.detach().cpu(),
        activation="silu",
        conv_state_indices=cache_indices.cpu(),
        query_start_loc=query_start_loc.cpu(),
    )

    out = torch.empty_like(x)
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        out,
        x,
        weight,
        conv_state=conv_states,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=None,
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=-1,
        run_mode=1,
        max_query_len=3,
    )

    torch.testing.assert_close(out.cpu(), out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(conv_states.cpu(), conv_states_ref, rtol=1e-2, atol=1e-2)
