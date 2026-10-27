# SPDX-License-Identifier: Apache-2.0
"""Compare native GDN updates on dense and block-interleaved mutable states."""

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from fla_npu.ops.ascendc import recurrent_gated_delta_rule as native_recurrent_gated_delta_rule

from vllm_ascend.models.qwen4_exp.cache_config import make_slab_view
from vllm_ascend.ops.gdn_attn_builder import _build_actual_seq_lengths
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_


@pytest.fixture
def states():
    torch.manual_seed(1234)
    device = torch.device("npu:0")
    conv_bytes = 6 * 2560 * 2
    ssm_bytes = 12 * 128 * 128 * 4
    stride = 896 * 1024
    backing = torch.full((10 * stride,), 90, dtype=torch.int8, device=device)
    conv = make_slab_view(
        backing, dtype=torch.bfloat16, num_blocks=10, item_shape=(6, 2560), storage_offset=0, block_stride=stride
    )
    ssm = make_slab_view(
        backing,
        dtype=torch.float32,
        num_blocks=10,
        item_shape=(12, 128, 128),
        storage_offset=conv_bytes,
        block_stride=stride,
    )
    conv.copy_(torch.randn(conv.shape, device=device, dtype=conv.dtype) * 0.01)
    ssm.copy_(torch.randn(ssm.shape, device=device, dtype=ssm.dtype) * 0.01)
    yield backing, conv, ssm, conv.clone(), ssm.clone()
    torch.npu.synchronize()
    padding = backing.view(10, stride)[:, conv_bytes + ssm_bytes :]
    assert torch.all(padding == 90).item()


def check_states(conv, ssm, dense_conv, dense_ssm):
    torch.testing.assert_close(conv, dense_conv, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ssm, dense_ssm, rtol=1e-3, atol=1e-3)


def test_conv_prefill_decode_and_speculative_updates(states):
    _, conv, ssm, dense_conv, dense_ssm = states
    device = conv.device
    weight = torch.randn((4, 2560), dtype=conv.dtype, device=device) * 0.01
    indices = torch.tensor([2, 7], dtype=torch.int32, device=device)
    x = torch.randn((12, 2560), dtype=conv.dtype, device=device)
    kwargs = dict(
        query_start_loc=torch.tensor([0, 7, 12], dtype=torch.int32, device=device),
        cache_indices=indices,
        has_initial_state=torch.ones(2, dtype=torch.bool, device=device),
        activation="silu",
        null_block_id=0,
    )
    expected = causal_conv1d_fn(x, weight, None, conv_states=dense_conv, **kwargs)
    actual = causal_conv1d_fn(x, weight, None, conv_states=conv, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    check_states(conv, ssm, dense_conv, dense_ssm)
    for width in (1, 4):
        x = torch.randn((2 * width, 2560), dtype=conv.dtype, device=device)
        kwargs = dict(
            conv_state_indices=indices,
            activation="silu",
            null_block_id=0,
            query_start_loc=torch.tensor([0, width, 2 * width], dtype=torch.int32, device=device),
            max_query_len=width,
        )
        if width > 1:
            kwargs["num_accepted_tokens"] = torch.tensor([1, 3], dtype=torch.int32, device=device)
        actual = causal_conv1d_update(x.clone(), conv, weight, out=torch.empty_like(x), **kwargs)
        expected = causal_conv1d_update(x.clone(), dense_conv, weight, out=torch.empty_like(x), **kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        check_states(conv, ssm, dense_conv, dense_ssm)


def test_recurrent_decode_speculation_and_indexed_state_io(states):
    _, conv, ssm, dense_conv, dense_ssm = states
    device = ssm.device
    for width in (1, 4):
        tokens = 2 * width
        q = torch.nn.functional.normalize(torch.randn((tokens, 4, 128), device=device), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn((tokens, 4, 128), device=device), dim=-1).bfloat16()
        v = torch.randn((tokens, 12, 128), dtype=torch.bfloat16, device=device)
        indices = [2, 7] if width == 1 else list(range(1, 9))
        kwargs = dict(
            beta=torch.full((tokens, 12), 0.5, dtype=torch.bfloat16, device=device),
            g=torch.full((tokens, 12), -0.1, device=device),
            scale=128**-0.5,
            actual_seq_lengths=_build_actual_seq_lengths(
                torch.tensor([0, width, 2 * width], dtype=torch.int32, device=device), 2
            ),
            ssm_state_indices=torch.tensor(indices, dtype=torch.int32, device=device),
        )
        if width > 1:
            kwargs["num_accepted_tokens"] = torch.tensor([1, 3], dtype=torch.int32, device=device)
        expected = native_recurrent_gated_delta_rule(q, k, v, dense_ssm, **kwargs)
        actual = native_recurrent_gated_delta_rule(q, k, v, ssm, **kwargs)
        assert torch.isfinite(expected).all().item()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        check_states(conv, ssm, dense_conv, dense_ssm)
    state_indices = torch.tensor([7, 2], dtype=torch.int32, device=device)
    mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    selected = gather_ssm_states(ssm, state_indices, mask)
    expected = dense_ssm[state_indices.long()].clone()
    expected[1].zero_()
    torch.testing.assert_close(selected, expected)
    selected.add_(0.25)
    scatter_ssm_states_(ssm, state_indices, selected)
    dense_ssm[state_indices.long()] = selected
    check_states(conv, ssm, dense_conv, dense_ssm)
