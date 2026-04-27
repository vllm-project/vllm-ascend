import gc

import pytest
import torch

from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_ref
from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_update as causal_conv1d_update_ref
from vllm_ascend.ops.triton.mamba.causal_conv1d import PAD_SLOT_ID, causal_conv1d_fn
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu as causal_conv1d_update
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


def to_int64_tuple(t):
    t = t.to(torch.int64)
    if t.dim() == 0:
        return (t.item(),)
    return tuple(t.tolist())


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
    out = torch.ops._C_ascend.npu_causal_conv1d_custom(
        x_origin,
        weight_origin,
        conv_state=conv_states_origin,
        bias_opt=bias,
        query_start_loc_opt=to_int64_tuple(query_start_loc),
        cache_indices_opt=to_int64_tuple(cache_indices),
        initial_state_mode_opt=to_int64_tuple(has_initial_state_tensor),
        num_accepted_tokens_opt=[],
        activation_mode=activation_num,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=0,
    ).transpose(-1, -2)
    validate_cmp(out, out_ref, itype)
    validate_cmp(conv_states, conv_states_ref, itype)


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


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1, 3])
@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("dim", [2048 + 16, 4096])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.parametrize("batch_size", [3, 64])
def test_causal_conv1d_update_with_batch_gather(
    batch_size, with_padding, dim, width, seqlen, has_bias, silu_activation, itype
):
    device = "npu"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    # total_entries = number of cache line
    total_entries = 10 * batch_size

    # x will be (batch, dim, seqlen) with contiguous along dim-axis
    x = torch.randn(padded_batch_size, seqlen, dim, device=device, dtype=itype).transpose(1, 2)

    x_ref = x.clone()

    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(dtype=torch.int32, device=device)
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[conv_state_indices] = False
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    # conv_state will be (cache_lines, dim, state_len)
    # with contiguous along dim-axis
    conv_state = torch.randn(total_entries, width - 1, dim, device=device, dtype=itype).transpose(1, 2)

    conv_state_for_padding_test = conv_state.clone()

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size].transpose(1, 2), conv_state_ref, weight, bias, activation=activation
    ).transpose(1, 2)

    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.equal(conv_state[unused_states_bool], conv_state_for_padding_test[unused_states_bool])
    assert torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_causal_conv1d_update_qwen3_next_shape():
    device = "npu"
    itype = torch.bfloat16
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    total_tokens = 192
    dim = 4096
    kernel_size = 4
    batch_size = 96
    num_states = 929

    x = torch.randn(total_tokens, dim, dtype=itype, device=device)
    conv_state = torch.randn(num_states, dim, kernel_size, dtype=itype, device=device)
    weight = torch.randn(dim, kernel_size, dtype=itype, device=device)
    bias = None
    conv_state_indices = torch.randint(0, num_states, (batch_size,), dtype=torch.int32, device=device)
    num_accepted_tokens = torch.ones(total_tokens, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(0, total_tokens + 1, dtype=torch.int32, device=device)

    activation = "silu"
    max_query_len = 2
    pad_slot_id = -1
    validate_data = False

    block_idx_last_scheduled_token = None
    initial_state_idx = None

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        pad_slot_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
        validate_data,
    )

    x_ref = x.clone()
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size].transpose(1, 2), conv_state_ref, weight, bias, activation=activation
    ).transpose(1, 2)

    assert torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
