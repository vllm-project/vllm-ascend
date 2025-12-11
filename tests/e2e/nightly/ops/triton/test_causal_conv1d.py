from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.mamba.causal_conv1d import (PAD_SLOT_ID,
                                                        causal_conv1d_fn)


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
@pytest.mark.parametrize('itype',
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('silu_activation', [False, True])
@pytest.mark.parametrize('has_bias', [False, True])
@pytest.mark.parametrize('seq_len', [[
    1, 2, 8, 16, 32, 64, 128, 129, 130, 151, 256, 372, 512, 784, 1024, 1134,
    2048, 4096
]])
@pytest.mark.parametrize('extra_state_len', [0, 2])
@pytest.mark.parametrize('width', [2, 3, 4])
@pytest.mark.parametrize('dim', [64, 4160])
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