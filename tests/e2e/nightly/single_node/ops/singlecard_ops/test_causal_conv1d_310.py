import pytest
import torch
import torch_npu
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_ref
from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_update as causal_conv1d_update_ref
from vllm_ascend.utils import enable_custom_op

torch_npu.npu.set_compile_mode(jit_compile=False)


def validate_cmp(y_cal, y_ref, device="npu"):
    y_cal = y_cal.to(device)
    y_ref = y_ref.to(device)
    torch.testing.assert_close(y_ref, y_cal, rtol=3e-03, atol=1e-02, equal_nan=True)


def to_int64_tuple(t):
    t = t.to(torch.int64)
    if t.dim() == 0:
        return (t.item(),)
    return tuple(t.tolist())


@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("seq_len", [[128, 1024, 2048, 4096]])
@pytest.mark.parametrize("extra_state_len", [0, 2])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("dim", [2048])
def test_ascend_causal_conv1d_310_fn(
    dim, width, extra_state_len, seq_len, has_bias, silu_activation, has_initial_state
):
    torch.random.manual_seed(0)
    enable_custom_op()
    device = "npu"
    cu_seqlen, num_seq = sum(seq_len), len(seq_len)
    state_len = width - 1 + extra_state_len

    x = torch.randn(cu_seqlen, dim, device=device, dtype=torch.float16).transpose(0, 1)
    weight = torch.randn(dim, width, device=device, dtype=torch.float16)
    query_start_loc = torch.cumsum(torch.tensor([0] + seq_len, device=device, dtype=torch.int32), dim=0).to(
        dtype=torch.int32
    )
    cache_indices = torch.arange(num_seq, device=device, dtype=torch.int32)
    has_initial_state_tensor = torch.tensor([has_initial_state] * num_seq, device=device, dtype=torch.bool)
    activation = None if not silu_activation else "silu"

    if has_initial_state:
        conv_states = torch.randn((num_seq, state_len, dim), device=device, dtype=torch.float16).transpose(-1, -2)
        conv_states_ref = (
            torch.randn((num_seq, state_len, dim), device=device, dtype=torch.float16)
            .transpose(-1, -2)
            .copy_(conv_states)
        )
    else:
        conv_states = torch.zeros((num_seq, state_len, dim), device=device, dtype=torch.float16).transpose(-1, -2)
        conv_states_ref = torch.zeros((num_seq, state_len, dim), device=device, dtype=torch.float16).transpose(-1, -2)

    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float16)
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

    x_origin = x.transpose(-1, -2)
    weight_origin = weight.transpose(-1, -2)
    conv_states_origin = conv_states.transpose(-1, -2)
    activation_mode = 1 if activation else 0
    out = torch.ops._C_ascend.npu_causal_conv1d_310(
        x_origin,
        weight_origin,
        bias=bias,
        conv_states=conv_states_origin,
        query_start_loc=to_int64_tuple(query_start_loc),
        cache_indices=to_int64_tuple(cache_indices),
        initial_state_mode=to_int64_tuple(has_initial_state_tensor),
        num_accepted_tokens=[],
        activation_mode=activation_mode,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=0,
    ).transpose(-1, -2)
    validate_cmp(out, out_ref)
    validate_cmp(conv_states, conv_states_ref)


@pytest.mark.parametrize("itype", [torch.float16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1, 3])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("dim", [2048, 4096])
@pytest.mark.parametrize("batch_size", [4, 8])
def test_causal_conv1d_310_update(batch_size, dim, width, seqlen, has_bias, silu_activation, itype):
    device = "npu"
    # total_entries = number of cache line
    total_entries = 10 * batch_size

    # x will be (batch, dim, seqlen) with contiguous along dim-axis
    x = torch.randn(batch_size, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)

    x_ref = x.clone()

    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(dtype=torch.int32, device=device)
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[conv_state_indices] = False

    # conv_states will be (cache_lines, dim, state_len)
    # with contiguous along dim-axis
    conv_states = torch.randn(total_entries, width, dim, device=device, dtype=itype).transpose(-1, -2)

    conv_state_for_padding_test = conv_states.detach().clone()

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_states[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"

    activation_mode = 1 if activation else 0
    has_initial_state_tensor = torch.tensor([True] * batch_size, device=device, dtype=torch.bool)
    conv_states_origin = conv_states.transpose(-1, -2)
    out = torch.ops._C_ascend.npu_causal_conv1d_310(
        x.transpose(-1, -2),
        weight.transpose(-1, -2),
        bias=bias,
        conv_states=conv_states_origin,
        query_start_loc=[],
        cache_indices=to_int64_tuple(conv_state_indices),
        initial_state_mode=to_int64_tuple(has_initial_state_tensor),
        num_accepted_tokens=[],
        activation_mode=activation_mode,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=1,
    ).transpose(-1, -2)

    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size].transpose(-1, -2), conv_state_ref, weight, bias, activation=activation
    ).transpose(-1, -2)
    validate_cmp(out[:batch_size], out_ref)
    validate_cmp(conv_states_origin[conv_state_indices, :], conv_state_ref.transpose(-1, -2))
    validate_cmp(
        conv_states_origin[unused_states_bool], conv_state_for_padding_test[unused_states_bool].transpose(-1, -2)
    )
