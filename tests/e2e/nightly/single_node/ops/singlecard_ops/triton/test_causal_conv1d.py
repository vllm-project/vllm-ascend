from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.mamba.causal_conv1d import (PAD_SLOT_ID,
                                                        causal_conv1d_fn)
from vllm_ascend.ops.triton.mamba.causal_conv1d import \
    causal_conv1d_update_npu as causal_conv1d_update


def validate_cmp(y_cal, y_ref, dtype, device='npu'):
    y_cal = y_cal.to(device)
    y_ref = y_ref.to(device)
    if dtype == torch.float16:
        torch.testing.assert_close(y_ref,
                                   y_cal,
                                   rtol=3e-03,
                                   atol=1e-02,
                                   equal_nan=True)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(y_ref,
                                   y_cal,
                                   rtol=1e-02,
                                   atol=1e-02,
                                   equal_nan=True)
    elif dtype == torch.float32:
        torch.testing.assert_close(y_ref,
                                   y_cal,
                                   rtol=1e-03,
                                   atol=4e-03,
                                   equal_nan=True)
    elif dtype == torch.int32 or dtype == torch.int64 or dtype == torch.int16 or dtype == torch.int8 or dtype == torch.uint32:
        assert torch.equal(y_cal, y_ref)
    elif dtype == torch.bool:
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError(
            'Invalid parameter \"dtype\" is found : {}'.format(dtype))


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x,
                       weight.unsqueeze(1),
                       bias,
                       padding=width - 1,
                       groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_states: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]][..., :(
                    width - 1)].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]][..., :(width - 1)]
                if has_initial_state[i] else None))
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor


@pytest.mark.parametrize('has_initial_state', [False, True])
@pytest.mark.parametrize('itype', [torch.bfloat16])
@pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize('seq_len', [[128, 1024, 2048, 4096]])
@pytest.mark.parametrize('extra_state_len', [0, 2])
@pytest.mark.parametrize('width', [2, 4])
@pytest.mark.parametrize('dim', [4160])
def test_causal_conv1d(dim, width, extra_state_len, seq_len, has_bias,
                       silu_activation, itype, has_initial_state):

    torch.random.manual_seed(0)

    device = "npu"
    cu_seqlen, num_seq = sum(seq_len), len(seq_len)
    state_len = width - 1 + extra_state_len

    x = torch.randn(cu_seqlen, dim, device=device, dtype=itype).transpose(0, 1)
    weight = torch.randn(dim, width, device=device, dtype=itype)
    query_start_loc = torch.cumsum(torch.tensor([0] + seq_len,
                                                device=device,
                                                dtype=torch.int32),
                                   dim=0)
    cache_indices = torch.arange(num_seq, device=device, dtype=torch.int32)
    has_initial_state_tensor = torch.tensor([has_initial_state] * num_seq,
                                            device=device,
                                            dtype=torch.bool)
    activation = None if not silu_activation else "silu"

    if has_initial_state:
        conv_states = torch.randn((num_seq, state_len, dim),
                                  device=device,
                                  dtype=itype).transpose(-1, -2)
        conv_states_ref = torch.randn(
            (num_seq, state_len, dim), device=device,
            dtype=itype).transpose(-1, -2).copy_(conv_states)
    else:
        conv_states = torch.zeros((num_seq, state_len, dim),
                                  device=device,
                                  dtype=itype).transpose(-1, -2)
        conv_states_ref = torch.zeros((num_seq, state_len, dim),
                                      device=device,
                                      dtype=itype).transpose(-1, -2)

    if has_bias:
        bias = torch.randn(dim, device=device, dtype=itype)
    else:
        bias = None

    out_ref = causal_conv1d_fn_pytorch(
        x,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states_ref,
        has_initial_state=has_initial_state_tensor,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc)
    out = causal_conv1d_fn(x,
                           weight,
                           bias=bias,
                           activation=activation,
                           conv_states=conv_states,
                           has_initial_state=has_initial_state_tensor,
                           cache_indices=cache_indices,
                           query_start_loc=query_start_loc)

    validate_cmp(out, out_ref, itype)
    validate_cmp(conv_states, conv_states_ref, itype)


def causal_conv1d_update_ref(x,
                             conv_state,
                             weight,
                             bias=None,
                             activation=None,
                             cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (torch.remainder(width_idx, state_len).unsqueeze(1).expand(
            -1, dim, -1))
        x_new = torch.cat([conv_state.gather(2, width_idx), x],
                          dim=-1).to(weight.dtype)
        copy_idx = torch.arange(
            seqlen, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx,
                                   state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0,
                   groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1, 3])
@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("dim", [2048 + 16, 4096])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.parametrize("batch_size", [3, 64])
def test_causal_conv1d_update_with_batch_gather(batch_size, with_padding, dim,
                                                width, seqlen, has_bias,
                                                silu_activation, itype):
    device = "npu"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    # total_entries = number of cache line
    total_entries = 10 * batch_size

    # x will be (batch, dim, seqlen) with contiguous along dim-axis
    x = torch.randn(padded_batch_size, seqlen, dim, device=device,
                    dtype=itype).transpose(1, 2)

    x_ref = x.clone()

    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device)
    unused_states_bool = torch.ones(total_entries,
                                    dtype=torch.bool,
                                    device=device)
    unused_states_bool[conv_state_indices] = False
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor(
                [PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    # conv_state will be (cache_lines, dim, state_len)
    # with contiguous along dim-axis
    conv_state = torch.randn(total_entries,
                             width - 1,
                             dim,
                             device=device,
                             dtype=itype).transpose(1, 2)

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
    out_ref = causal_conv1d_update_ref(x_ref[:batch_size],
                                       conv_state_ref,
                                       weight,
                                       bias,
                                       activation=activation)

    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.equal(conv_state[unused_states_bool],
                       conv_state_for_padding_test[unused_states_bool])
    assert torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)
