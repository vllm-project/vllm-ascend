import gc

import pytest
import torch

from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_ref
from vllm_ascend.ops.triton.mamba.causal_conv1d import PAD_SLOT_ID, causal_conv1d_fn
from vllm_ascend.utils import enable_custom_op


def validate_cmp(y_cal, y_ref, dtype, device="npu"):
    y_cal = y_cal.to(device)
    y_ref = y_ref.to(device)
    if dtype == torch.float16:
        torch.testing.assert_close(y_ref, y_cal, rtol=3e-03, atol=1e-02, equal_nan=True)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-02, atol=1e-02, equal_nan=True)
    elif dtype == torch.float32:
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=4e-03, equal_nan=True)
    elif (
        dtype == torch.int32
        or dtype == torch.int64
        or dtype == torch.int16
        or dtype == torch.int8
        or dtype == torch.uint32
        or dtype == torch.bool
    ):
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("seq_len", [[128, 1024, 2048, 4096]])
@pytest.mark.parametrize("extra_state_len", [0, 2])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("dim", [2048])
def test_ascend_causal_conv1d(
    dim, width, extra_state_len, seq_len, has_bias, silu_activation, itype, has_initial_state
):
    torch.random.manual_seed(0)
    enable_custom_op()
    device = "npu"
    cu_seqlen, num_seq = sum(seq_len), len(seq_len)
    state_len = width - 1 + extra_state_len

    x = torch.randn(cu_seqlen, dim, device=device, dtype=itype).transpose(0, 1)
    weight = torch.randn(dim, width, device=device, dtype=itype)  #
    query_start_loc = torch.cumsum(torch.tensor([0] + seq_len, device=device, dtype=torch.int32), dim=0).to(
        dtype=torch.int32
    )
    cache_indices = torch.arange(num_seq, device=device, dtype=torch.int32)
    has_initial_state_tensor = torch.tensor([has_initial_state] * num_seq, device=device, dtype=torch.bool)
    activation = None if not silu_activation else "silu"

    if has_initial_state:
        conv_states = torch.randn((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2)
        conv_states_ref = (
            torch.randn((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2).copy_(conv_states)
        )
    else:
        conv_states = torch.zeros((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2)
        conv_states_ref = torch.zeros((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2)

    if has_bias:
        bias = torch.randn(dim, device=device, dtype=itype)
    else:
        bias = None

    out_ref = causal_conv1d_fn_ref(
        x,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states_ref,
        has_initial_state=has_initial_state_tensor,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
    # out = causal_conv1d_fn(x,
    #                        weight,
    #                        bias=bias,
    #                        activation=activation,
    #                        conv_states=conv_states,
    #                        has_initial_state=has_initial_state_tensor,
    #                        cache_indices=cache_indices,
    #                        query_start_loc=query_start_loc)
    x_origin = x.transpose(-1, -2)
    weight_origin = weight.transpose(-1, -2)
    conv_states_origin = conv_states.transpose(-1, -2)
    activation_num = 1 if activation else 0
    out = torch.empty_like(x_origin)
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        out,
        x_origin,
        weight_origin,
        conv_state=conv_states_origin,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=has_initial_state_tensor,
        num_accepted_tokens_opt=None,
        activation_mode=activation_num,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=0,
    )
    out = out.transpose(-1, -2)
    validate_cmp(out, out_ref, itype)
    validate_cmp(conv_states, conv_states_ref, itype)


@pytest.mark.skip(
    reason="To use this tirton ops:causal_conv1d_fn, you need to set `get_forward_context`. After\
          the model side dumps the data, Zeng Tian has made the necessary fixes."
)
@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("seq_len", [[128, 1024, 2048, 4096]])
@pytest.mark.parametrize("extra_state_len", [0, 2])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("dim", [4160])
def test_causal_conv1d(dim, width, extra_state_len, seq_len, has_bias, silu_activation, itype, has_initial_state):
    torch.random.manual_seed(0)

    device = "npu"
    cu_seqlen, num_seq = sum(seq_len), len(seq_len)
    state_len = width - 1 + extra_state_len

    x = torch.randn(cu_seqlen, dim, device=device, dtype=itype).transpose(0, 1)
    weight = torch.randn(dim, width, device=device, dtype=itype)
    query_start_loc = torch.cumsum(torch.tensor([0] + seq_len, device=device, dtype=torch.int32), dim=0)
    cache_indices = torch.arange(num_seq, device=device, dtype=torch.int32)
    has_initial_state_tensor = torch.tensor([has_initial_state] * num_seq, device=device, dtype=torch.bool)
    activation = None if not silu_activation else "silu"

    if has_initial_state:
        conv_states = torch.randn((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2)
        conv_states_ref = (
            torch.randn((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2).copy_(conv_states)
        )
    else:
        conv_states = torch.zeros((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2)
        conv_states_ref = torch.zeros((num_seq, state_len, dim), device=device, dtype=itype).transpose(-1, -2)

    if has_bias:
        bias = torch.randn(dim, device=device, dtype=itype)
    else:
        bias = None

    out_ref = causal_conv1d_fn_ref(
        x,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states_ref,
        has_initial_state=has_initial_state_tensor,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
    out = causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states,
        has_initial_state=has_initial_state_tensor,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )

    validate_cmp(out, out_ref, itype)
    validate_cmp(conv_states, conv_states_ref, itype)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
