import gc

import pytest
import torch
import torch.nn.functional as F

from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update


def causal_conv1d_fn_ref(x, weight, bias, conv_states, has_initial_state, cache_indices, query_start_loc, activation):
    """Compute the varlen prefill depthwise-convolution golden on CPU."""
    x_ref = x.detach().cpu().float()
    weight_ref = weight.detach().cpu().float().T.contiguous()
    bias_ref = None if bias is None else bias.detach().cpu().float()
    state_ref = conv_states.detach().cpu().float().clone()
    starts = query_start_loc.cpu().tolist()
    indices = cache_indices.cpu().tolist()
    initial = has_initial_state.cpu().tolist()
    width = weight.shape[0]
    outputs = []
    for request, state_idx in enumerate(indices):
        tokens = x_ref[starts[request] : starts[request + 1]].T.unsqueeze(0)
        history = state_ref[state_idx, : width - 1].T.unsqueeze(0) if initial[request] else None
        if history is None:
            y = F.conv1d(tokens, weight_ref.unsqueeze(1), bias_ref, padding=width - 1, groups=x.shape[-1])
        else:
            y = F.conv1d(torch.cat((history, tokens), dim=-1), weight_ref.unsqueeze(1), bias_ref, groups=x.shape[-1])
        y = y[..., : tokens.shape[-1]]
        outputs.append((F.silu(y) if activation == "silu" else y).squeeze(0).T)
        all_tokens = tokens if history is None else torch.cat((history, tokens), dim=-1)
        state_ref[state_idx, : width - 1].copy_(all_tokens[0, :, -(width - 1) :].T)
    return torch.cat(outputs).to(x.dtype), state_ref.to(conv_states.dtype)


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
    device = "npu"
    cu_seqlen, num_seq = sum(seq_len), len(seq_len)
    state_len = width - 1 + extra_state_len

    # FLA uses [T, D], [W, D], and [B, state, D] layouts.
    x = torch.randn(cu_seqlen, dim, device=device, dtype=itype)
    weight = torch.randn(width, dim, device=device, dtype=itype)
    query_start_loc = torch.cumsum(torch.tensor([0] + seq_len, device=device, dtype=torch.int32), dim=0).to(
        dtype=torch.int32
    )
    cache_indices = torch.arange(1, num_seq + 1, device=device, dtype=torch.int32)
    has_initial_state_tensor = torch.tensor([has_initial_state] * num_seq, device=device, dtype=torch.bool)
    activation = None if not silu_activation else "silu"

    if has_initial_state:
        conv_states = torch.randn((num_seq + 1, state_len, dim), device=device, dtype=itype)
    else:
        conv_states = torch.zeros((num_seq + 1, state_len, dim), device=device, dtype=itype)

    if has_bias:
        bias = torch.randn(dim, device=device, dtype=itype)
    else:
        bias = None

    out_ref, conv_states_ref = causal_conv1d_fn_ref(
        x, weight, bias, conv_states, has_initial_state_tensor, cache_indices, query_start_loc, activation
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
        null_block_id=0,
    )
    validate_cmp(out, out_ref, itype)
    validate_cmp(conv_states, conv_states_ref, itype)


@pytest.mark.parametrize("itype", [torch.bfloat16])
def test_ascend_causal_conv1d_update(itype):
    """Compare one-token decode output and in-place state updates with a CPU golden."""
    torch.random.manual_seed(0)
    device = "npu"
    activation = "silu"
    batch_size, dim, width = 2, 16, 4
    state_len = width - 1

    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    weight = torch.randn(width, dim, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype)
    # Block 0 is reserved as the null block; use only non-zero indices.
    conv_states = torch.randn(batch_size + 1, state_len, dim, device=device, dtype=itype)
    conv_states_ref = conv_states.cpu().float().clone()
    x_ref = x.cpu().float()
    weight_ref = weight.cpu().float()
    bias_ref = bias.cpu().float()
    conv_state_indices = torch.arange(1, batch_size + 1, device=device, dtype=torch.int32)

    expected = []
    for request_idx, state_idx in enumerate(conv_state_indices.tolist()):
        state = conv_states_ref[state_idx]
        window = torch.cat((state, x_ref[request_idx : request_idx + 1]), dim=0)
        y = (window * weight_ref).sum(dim=0) + bias_ref
        if activation == "silu":
            y = F.silu(y)
        expected.append(y.to(itype))
        conv_states_ref[state_idx].copy_(window[-state_len:])
    expected = torch.stack(expected)

    actual = causal_conv1d_update(
        x,
        conv_states,
        weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
    )

    validate_cmp(actual, expected, itype)
    validate_cmp(conv_states, conv_states_ref.to(itype), itype)


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
    from vllm_ascend.ops.causal_conv1d import causal_conv1d_fn as legacy_ref
    from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn as triton_causal_conv1d_fn
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

    out_ref = legacy_ref(
        x,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states_ref,
        has_initial_state=has_initial_state_tensor,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
    out = triton_causal_conv1d_fn(
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
