from typing import Optional

import gc
import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.mamba.causal_conv1d import (PAD_SLOT_ID,
                                                        causal_conv1d_fn)
from vllm_ascend.ops.triton.mamba.causal_conv1d import \
    causal_conv1d_update_npu as causal_conv1d_update
from vllm_ascend._310p.ops.causal_conv1d import (
    causal_conv1d_fn as causal_conv1d_fn_ref,
    causal_conv1d_update as causal_conv1d_update_ref
)
from vllm_ascend.utils import enable_custom_op

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

def to_int64_tuple(t):
    t = t.to(torch.int64)
    if t.dim() == 0:
        return (t.item(),)
    return tuple(t.tolist())

@pytest.mark.parametrize('has_initial_state', [False, True])
@pytest.mark.parametrize('itype', [torch.bfloat16])
@pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize('seq_len', [[128, 1024, 2048, 4096]])
@pytest.mark.parametrize('extra_state_len', [0, 2])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize('dim', [2048])
def test_ascend_causal_conv1d(dim, width, extra_state_len, seq_len, has_bias,
                       silu_activation, itype, has_initial_state):

    torch.random.manual_seed(0)
    enable_custom_op()
    device = "npu"
    cu_seqlen, num_seq = sum(seq_len), len(seq_len)
    state_len = width - 1 + extra_state_len

    x = torch.randn(cu_seqlen, dim, device=device, dtype=itype).transpose(0, 1)
    weight = torch.randn(dim, width, device=device, dtype=itype)#
    query_start_loc = torch.cumsum(torch.tensor([0] + seq_len,
                                                device=device,
                                                dtype=torch.int32),
                                   dim=0).to(dtype=torch.int32)
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

    out_ref = causal_conv1d_fn_ref(
        x,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states_ref,
        has_initial_state=has_initial_state_tensor,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc)
    # out = causal_conv1d_fn(x,
    #                        weight,
    #                        bias=bias,
    #                        activation=activation,
    #                        conv_states=conv_states,
    #                        has_initial_state=has_initial_state_tensor,
    #                        cache_indices=cache_indices,
    #                        query_start_loc=query_start_loc)
    x_origin=x.transpose(-1, -2)
    weight_origin=weight.transpose(-1, -2)
    conv_states_origin=conv_states.transpose(-1, -2)
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
                    run_mode=0
                ).transpose(-1, -2)
    validate_cmp(out, out_ref, itype)
    validate_cmp(conv_states, conv_states_ref, itype)


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

    out_ref = causal_conv1d_fn_ref(
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
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@pytest.mark.parametrize('has_initial_state', [False, True])
@pytest.mark.parametrize('itype', [torch.bfloat16])
@pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize('batch_size', [4, 16])
@pytest.mark.parametrize('seq_len_base', [128, 512])
@pytest.mark.parametrize('seq_len_fluctuation', [0, 64])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize('dim', [2048])
def test_causal_conv1d_fn_batch_consistency(
    dim, width, batch_size, seq_len_base, seq_len_fluctuation, has_bias,
    silu_activation, itype, has_initial_state):

    torch.random.manual_seed(42)
    device = "npu"
    activation = None if not silu_activation else "silu"

    seq_len = [
        seq_len_base + torch.randint(-seq_len_fluctuation,
                                     seq_len_fluctuation + 1, (1,)).item()
        for _ in range(batch_size)
    ]

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None

    num_seq = len(seq_len)
    cu_seqlen = sum(seq_len)
    state_len = width - 1

    x_all = torch.randn(cu_seqlen, dim, device=device, dtype=itype)

    if has_initial_state:
        conv_states_all = torch.randn(num_seq, state_len, dim,
                                      device=device, dtype=itype)
    else:
        conv_states_all = torch.zeros(num_seq, state_len, dim,
                                      device=device, dtype=itype)
    has_initial_state_tensor = torch.tensor([has_initial_state] * num_seq,
                                            device=device, dtype=torch.bool)

    query_start_loc_batch = torch.cumsum(
        torch.tensor([0] + seq_len, device=device, dtype=torch.int32),
        dim=0)
    cache_indices_batch = torch.arange(num_seq, device=device, dtype=torch.int32)

    x_batch = x_all.transpose(0, 1)
    conv_states_batch = conv_states_all.transpose(-1, -2).clone()

    out_batch = causal_conv1d_fn(
        x_batch,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states_batch,
        has_initial_state=has_initial_state_tensor,
        cache_indices=cache_indices_batch,
        query_start_loc=query_start_loc_batch,
    )

    offset = 0
    for seq_idx, sl in enumerate(seq_len):
        x_single = x_all[offset:offset + sl].transpose(0, 1)
        conv_states_single = conv_states_all[seq_idx:seq_idx + 1].transpose(
            -1, -2).clone()
        has_init_single = torch.tensor([has_initial_state],
                                       device=device, dtype=torch.bool)
        cache_idx_single = torch.tensor([0], device=device, dtype=torch.int32)
        qsl_single = torch.tensor([0, sl], device=device, dtype=torch.int32)

        out_single = causal_conv1d_fn(
            x_single,
            weight,
            bias=bias,
            activation=activation,
            conv_states=conv_states_single,
            has_initial_state=has_init_single,
            cache_indices=cache_idx_single,
            query_start_loc=qsl_single,
        )

        validate_cmp(out_batch[:, offset:offset + sl], out_single, itype)
        validate_cmp(conv_states_batch[seq_idx:seq_idx + 1],
                     conv_states_single, itype)

        offset += sl

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize('has_initial_state', [False, True])
@pytest.mark.parametrize('itype', [torch.bfloat16])
@pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize('batch_size', [4, 16])
@pytest.mark.parametrize('seq_len_base', [128, 512])
@pytest.mark.parametrize('seq_len_fluctuation', [0, 64])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize('dim', [2048])
def test_npu_causal_conv1d_custom_batch_consistency(
    dim, width, batch_size, seq_len_base, seq_len_fluctuation, has_bias,
    silu_activation, itype, has_initial_state):

    torch.random.manual_seed(42)
    enable_custom_op()
    device = "npu"
    activation = None if not silu_activation else "silu"
    activation_num = 1 if activation else 0

    seq_len = [
        seq_len_base + torch.randint(-seq_len_fluctuation,
                                     seq_len_fluctuation + 1, (1,)).item()
        for _ in range(batch_size)
    ]

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None

    num_seq = len(seq_len)
    cu_seqlen = sum(seq_len)
    state_len = width - 1

    x_all = torch.randn(cu_seqlen, dim, device=device, dtype=itype)

    if has_initial_state:
        conv_states_init = torch.randn(num_seq, state_len, dim,
                                       device=device, dtype=itype)
    else:
        conv_states_init = torch.zeros(num_seq, state_len, dim,
                                       device=device, dtype=itype)
    has_initial_state_tensor = torch.tensor([has_initial_state] * num_seq,
                                            device=device, dtype=torch.bool)

    query_start_loc_batch = torch.cumsum(
        torch.tensor([0] + seq_len, device=device, dtype=torch.int32), dim=0)
    cache_indices_batch = torch.arange(num_seq, device=device, dtype=torch.int32)

    weight_T = weight.transpose(-1, -2)
    conv_states_batch = conv_states_init.clone()

    out_batch = torch.ops._C_ascend.npu_causal_conv1d_custom(
        x_all,
        weight_T,
        conv_state=conv_states_batch,
        bias_opt=bias,
        query_start_loc_opt=to_int64_tuple(query_start_loc_batch),
        cache_indices_opt=to_int64_tuple(cache_indices_batch),
        initial_state_mode_opt=to_int64_tuple(has_initial_state_tensor),
        num_accepted_tokens_opt=[],
        activation_mode=activation_num,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=0,
    )

    offset = 0
    for seq_idx, sl in enumerate(seq_len):
        conv_states_single = conv_states_init[seq_idx:seq_idx + 1].clone()
        has_init_single = torch.tensor([has_initial_state],
                                       device=device, dtype=torch.bool)
        cache_idx_single = torch.tensor([0], device=device, dtype=torch.int32)
        qsl_single = torch.tensor([0, sl], device=device, dtype=torch.int32)

        out_single = torch.ops._C_ascend.npu_causal_conv1d_custom(
            x_all[offset:offset + sl],
            weight_T,
            conv_state=conv_states_single,
            bias_opt=bias,
            query_start_loc_opt=to_int64_tuple(qsl_single),
            cache_indices_opt=to_int64_tuple(cache_idx_single),
            initial_state_mode_opt=to_int64_tuple(has_init_single),
            num_accepted_tokens_opt=[],
            activation_mode=activation_num,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=0,
        )

        validate_cmp(out_batch[offset:offset + sl], out_single, itype)
        validate_cmp(conv_states_batch[seq_idx:seq_idx + 1],
                     conv_states_single, itype)

        offset += sl

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize('itype', [torch.bfloat16])
@pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize('batch_size', [4, 64])
@pytest.mark.parametrize('seqlen', [1, 3])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize('dim', [2048])
def test_causal_conv1d_update_npu_batch_consistency(
    batch_size, dim, width, seqlen, has_bias, silu_activation, itype):

    torch.random.manual_seed(42)
    device = "npu"
    activation = None if not silu_activation else "silu"

    total_entries = 10 * batch_size

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None

    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device)

    x_all = torch.randn(batch_size, seqlen, dim, device=device, dtype=itype)
    conv_state_storage_init = torch.randn(total_entries, width - 1, dim,
                                          device=device, dtype=itype)

    conv_state_storage_batch = conv_state_storage_init.clone()
    conv_state_batch = conv_state_storage_batch.transpose(1, 2)

    out_batch = causal_conv1d_update(
        x_all.transpose(1, 2),
        conv_state_batch,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )

    for i in range(batch_size):
        conv_state_storage_single = conv_state_storage_init.clone()
        conv_state_single = conv_state_storage_single.transpose(1, 2)
        single_idx = conv_state_indices[i:i + 1]

        out_single = causal_conv1d_update(
            x_all[i:i + 1].transpose(1, 2),
            conv_state_single,
            weight,
            bias,
            activation=activation,
            conv_state_indices=single_idx,
            pad_slot_id=PAD_SLOT_ID,
        )

        validate_cmp(out_batch[i:i + 1], out_single, itype)
        cache_idx = conv_state_indices[i].item()
        validate_cmp(conv_state_storage_batch[cache_idx:cache_idx + 1],
                     conv_state_storage_single[cache_idx:cache_idx + 1], itype)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize('itype', [torch.bfloat16])
@pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize('batch_size', [4, 16])
@pytest.mark.parametrize('seq_len_base', [2, 8])
@pytest.mark.parametrize('seq_len_fluctuation', [0, 4])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize('dim', [2048])
def test_causal_conv1d_update_npu_varlen_batch_consistency(
    dim, width, batch_size, seq_len_base, seq_len_fluctuation, has_bias,
    silu_activation, itype):

    torch.random.manual_seed(42)
    device = "npu"
    activation = None if not silu_activation else "silu"

    seq_len = [
        seq_len_base + torch.randint(-seq_len_fluctuation,
                                     seq_len_fluctuation + 1, (1,)).item()
        for _ in range(batch_size)
    ]

    num_seq = len(seq_len)
    total_tokens = sum(seq_len)
    max_seqlen = max(seq_len)
    total_entries = 10 * num_seq

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None

    conv_state_indices = torch.randperm(total_entries)[:num_seq].to(
        dtype=torch.int32, device=device)

    x_all = torch.randn(total_tokens, dim, device=device, dtype=itype)
    conv_state_storage_init = torch.randn(total_entries, width - 1, dim,
                                          device=device, dtype=itype)

    query_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len, device=device, dtype=torch.int32), dim=0)

    conv_state_storage_varlen = conv_state_storage_init.clone()
    conv_state_varlen = conv_state_storage_varlen.transpose(1, 2)

    out_varlen = causal_conv1d_update(
        x_all,
        conv_state_varlen,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        query_start_loc=query_start_loc,
        max_query_len=max_seqlen,
        pad_slot_id=PAD_SLOT_ID,
    )

    offset = 0
    for seq_idx, sl in enumerate(seq_len):
        single_idx = conv_state_indices[seq_idx:seq_idx + 1]
        qsl_single = torch.tensor([0, sl], device=device, dtype=torch.int32)
        conv_state_storage_single = conv_state_storage_init.clone()
        conv_state_single = conv_state_storage_single.transpose(1, 2)

        out_single = causal_conv1d_update(
            x_all[offset:offset + sl],
            conv_state_single,
            weight,
            bias,
            activation=activation,
            conv_state_indices=single_idx,
            query_start_loc=qsl_single,
            max_query_len=sl,
            pad_slot_id=PAD_SLOT_ID,
        )

        validate_cmp(out_varlen[offset:offset + sl], out_single, itype)
        cache_idx = conv_state_indices[seq_idx].item()
        validate_cmp(conv_state_storage_varlen[cache_idx:cache_idx + 1],
                     conv_state_storage_single[cache_idx:cache_idx + 1], itype)

        offset += sl

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
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size].transpose(1, 2), conv_state_ref, weight, bias, activation=activation
    ).transpose(1, 2)

    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.equal(conv_state[unused_states_bool],
                       conv_state_for_padding_test[unused_states_bool])
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