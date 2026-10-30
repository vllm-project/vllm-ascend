# SPDX-License-Identifier: Apache-2.0
"""Compare GDN updates on dense and block-interleaved mutable states."""

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import causal_conv1d_fn, causal_conv1d_update
from fla_npu.ops.ascendc import recurrent_gated_delta_rule as native_recurrent_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule

from vllm_ascend.models.qwen4_exp.cache_config import make_slab_view
from vllm_ascend.ops.gdn_attn_builder import _build_actual_seq_lengths
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_


@pytest.fixture
def states(request):
    torch.manual_seed(1234)
    device = torch.device("npu:0")
    max_query_len = getattr(request, "param", 4)
    state_len = max_query_len + 2
    num_blocks = 2 * max_query_len + 2
    conv_bytes = state_len * 2560 * 2
    ssm_bytes = 12 * 128 * 128 * 4
    stride = 896 * 1024
    backing = torch.full((num_blocks * stride,), 90, dtype=torch.int8, device=device)
    conv = make_slab_view(
        backing,
        dtype=torch.bfloat16,
        num_blocks=num_blocks,
        item_shape=(state_len, 2560),
        storage_offset=0,
        block_stride=stride,
    )
    ssm = make_slab_view(
        backing,
        dtype=torch.float32,
        num_blocks=num_blocks,
        item_shape=(12, 128, 128),
        storage_offset=conv_bytes,
        block_stride=stride,
    )
    conv.copy_(torch.randn(conv.shape, device=device, dtype=conv.dtype) * 0.01)
    ssm.copy_(torch.randn(ssm.shape, device=device, dtype=ssm.dtype) * 0.01)
    yield backing, conv, ssm, conv.clone(), ssm.clone()
    torch.npu.synchronize()
    padding = backing.view(num_blocks, stride)[:, conv_bytes + ssm_bytes :]
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


@pytest.mark.parametrize("states", [4, 10], indirect=True, ids=["three-drafts", "nine-drafts"])
def test_adaptive_speculation_matches_independent_requests(states):
    """Retain accepted states when the next step trims to one token, with padding."""
    _, conv, ssm, dense_conv, dense_ssm = states
    device = ssm.device
    max_query_len = conv.shape[1] - 2
    slots = torch.arange(1, 2 * max_query_len + 1, dtype=torch.int32, device=device).reshape(2, max_query_len)
    slots = torch.cat([slots, torch.zeros((1, max_query_len), dtype=torch.int32, device=device)])
    conv_indices = slots[:, 0].contiguous()
    weight = torch.randn((4, 2560), dtype=conv.dtype, device=device) * 0.01
    steps = [
        ([max_query_len, max_query_len], [1, 1]),
        ([1, max_query_len - 1], [max_query_len, max_query_len - 1]),
        ([1, 1], [1, max_query_len - 1]),
        ([max_query_len - 1, 1], [1, 1]),
    ]
    for lengths, accepted in steps:
        total = sum(lengths)
        boundaries = [0, lengths[0], total, total]
        qsl = torch.tensor(boundaries, dtype=torch.int32, device=device)
        accepted_gpu = torch.tensor(accepted + [1], dtype=torch.int32, device=device)
        x = torch.randn((total, 2560), dtype=conv.dtype, device=device)
        actual_conv = causal_conv1d_update(
            x.clone(),
            conv,
            weight,
            conv_state_indices=conv_indices,
            num_accepted_tokens=accepted_gpu,
            query_start_loc=qsl,
            max_query_len=max_query_len,
            null_block_id=0,
            activation="silu",
            out=torch.empty_like(x),
        )
        reference_conv = []
        for row, length in enumerate(lengths):
            segment = x[boundaries[row] : boundaries[row + 1]].clone()
            reference_conv.append(
                causal_conv1d_update(
                    segment,
                    dense_conv,
                    weight,
                    conv_state_indices=conv_indices[row : row + 1],
                    num_accepted_tokens=accepted_gpu[row : row + 1],
                    query_start_loc=torch.tensor([0, length], dtype=torch.int32, device=device),
                    max_query_len=max_query_len,
                    null_block_id=0,
                    activation="silu",
                    out=torch.empty_like(segment),
                )
            )
        torch.testing.assert_close(actual_conv, torch.cat(reference_conv), rtol=1e-3, atol=1e-3)

        q = torch.nn.functional.normalize(torch.randn((1, total, 4, 128), device=device), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn((1, total, 4, 128), device=device), dim=-1).bfloat16()
        v = torch.randn((1, total, 12, 128), dtype=torch.bfloat16, device=device)
        g = torch.full((1, total, 12), -0.1, device=device)
        beta = torch.full((1, total, 12), 0.5, dtype=torch.bfloat16, device=device)
        if max_query_len == 4 and lengths == [4, 4]:
            # At the full-width shape both APIs are valid. Check numerical
            # parity with the existing native operator before testing trims.
            native_state = ssm.clone()
            native_output = native_recurrent_gated_delta_rule(
                q[0],
                k[0],
                v[0],
                native_state,
                g=g[0],
                beta=beta[0],
                scale=128**-0.5,
                actual_seq_lengths=_build_actual_seq_lengths(qsl[:3], 2),
                ssm_state_indices=slots[:2].flatten(),
                num_accepted_tokens=accepted_gpu[:2],
            )
        actual, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=ssm,
            cu_seqlens=qsl,
            ssm_state_indices=slots,
            num_accepted_tokens=accepted_gpu,
        )
        if max_query_len == 4 and lengths == [4, 4]:
            torch.testing.assert_close(actual[0], native_output, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(ssm, native_state, rtol=1e-3, atol=1e-3)
        reference = []
        for row, length in enumerate(lengths):
            start, end = boundaries[row : row + 2]
            output, _ = fused_recurrent_gated_delta_rule(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                initial_state=dense_ssm,
                cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=device),
                ssm_state_indices=slots[row : row + 1],
                num_accepted_tokens=accepted_gpu[row : row + 1],
            )
            reference.append(output)
        torch.testing.assert_close(actual, torch.cat(reference, dim=1), rtol=0, atol=0)
        torch.testing.assert_close(ssm, dense_ssm, rtol=0, atol=0)
        torch.testing.assert_close(conv, dense_conv, rtol=1e-3, atol=1e-3)


def test_adaptive_recurrent_reads_full_state_row_after_nine_drafts():
    """A one-token query must read column nine, independently for each request."""
    device = torch.device("npu:0")
    state = torch.arange(21, dtype=torch.float32, device=device).view(21, 1, 1, 1)
    state = state.expand(21, 4, 128, 128).contiguous()
    slot_storage = torch.zeros((3, 12), dtype=torch.int32, device=device)
    slot_storage[:2, :10] = torch.arange(1, 21, dtype=torch.int32, device=device).reshape(2, 10)
    # A view into a wider block table must retain its physical row stride.
    slots = slot_storage[:, :10]
    q = torch.zeros((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    g = torch.zeros((1, 2, 4), device=device)
    beta = torch.zeros((1, 2, 4), dtype=torch.bfloat16, device=device)
    output, _ = fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=1.0,
        initial_state=state,
        cu_seqlens=torch.tensor([0, 1, 2, 2], dtype=torch.int32, device=device),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor([10, 10, 1], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(output[0, 0], torch.full_like(output[0, 0], 10), rtol=0, atol=0)
    torch.testing.assert_close(output[0, 1], torch.full_like(output[0, 1], 20), rtol=0, atol=0)
    torch.testing.assert_close(state[1], torch.full_like(state[1], 10), rtol=0, atol=0)
    torch.testing.assert_close(state[11], torch.full_like(state[11], 20), rtol=0, atol=0)
    assert torch.count_nonzero(state[0]) == 0
