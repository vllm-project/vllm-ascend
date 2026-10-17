# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

conv_ops = importlib.import_module("vllm_ascend.models.glm5next.ops.causal_conv1d")


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("run_mode", [0, 1])
def test_packed_first_row_is_not_a_null_block(monkeypatch, packed, run_mode):
    state = torch.zeros(4, 3, 8)
    if packed:
        state = state[::2]
    indices = torch.tensor([[1]], dtype=torch.int32)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.ones(1, 8)
    calls = []

    def copy_state(cache, staging, ids, query_starts, staging_ids, *, write_back):
        calls.append(write_back)
        if not write_back:
            staging.copy_(cache[1:2])
            staging_ids.zero_()

    def prefill(x, weight, bias, **kwargs):
        assert kwargs["null_block_id"] == 0
        assert kwargs["cache_indices"].item() == 1
        assert kwargs["cache_indices"].item() != kwargs["null_block_id"]
        return x + 2

    def decode(x, state, weight, **kwargs):
        torch.testing.assert_close(kwargs["out"], torch.zeros_like(x))
        assert kwargs["out"].data_ptr() != x.data_ptr()
        kwargs["cache_indices"] = kwargs["conv_state_indices"]
        return prefill(x, weight, **kwargs)

    monkeypatch.setattr(conv_ops, "copy_conv_state", copy_state)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", prefill)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", decode)
    result = conv_ops.causal_conv1d(x, torch.ones(4, 8), state, starts, indices, run_mode=run_mode)
    torch.testing.assert_close(result, x + 2)
    assert calls == ([False, True] if packed else [])


@pytest.mark.parametrize("run_mode", [0, 1])
@pytest.mark.parametrize("inactive_slot,inactive_tokens", [(1, 0), (-1, 1), (4, 1)])
def test_packed_inactive_request_uses_reserved_null_slot(monkeypatch, run_mode, inactive_slot, inactive_tokens):
    backing = torch.arange(8 * 3 * 8, dtype=torch.float32).reshape(8, 3, 8)
    state = backing[::2]
    expected = backing.clone()
    indices = torch.tensor([2, inactive_slot, 3], dtype=torch.int32)
    starts = torch.tensor([0, 1, 1 + inactive_tokens, 2 + inactive_tokens], dtype=torch.int32)
    x = torch.ones(3, 8)
    staged = []

    def copy_state(cache, staging, ids, query_starts, staging_ids, *, write_back):
        assert cache is state
        assert ids is indices
        assert query_starts is starts
        if not write_back:
            staging[0].copy_(cache[2])
            staging[1].zero_()
            staging[2].copy_(cache[3])
            staging_ids.copy_(torch.tensor([0, -1, 2], dtype=torch.int32))
            staged.append(staging)
        else:
            assert staging is staged[0]
            cache[2].copy_(staging[0])
            cache[3].copy_(staging[2])

    def check_kernel_state(kernel_state, kernel_indices, null_block_id):
        assert null_block_id == 0
        assert kernel_state.shape == (4, 3, 8)
        assert kernel_state.is_contiguous()
        torch.testing.assert_close(kernel_indices, torch.tensor([1, 0, 3], dtype=torch.int32))
        assert staged[0].data_ptr() == kernel_state[1:].data_ptr()
        torch.testing.assert_close(kernel_state[1], state[2])
        torch.testing.assert_close(kernel_state[3], state[3])
        kernel_state[1].add_(7)
        kernel_state[3].add_(7)

    # Keep strict API signatures: prefill has no out parameter, decode does.
    def prefill(
        x,
        weight,
        bias,
        *,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
    ):
        assert pad_slot_id == -1
        check_kernel_state(conv_states, cache_indices, null_block_id)
        return x + 2

    def decode(
        x,
        kernel_state,
        weight,
        *,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        null_block_id,
        out,
    ):
        check_kernel_state(kernel_state, conv_state_indices, null_block_id)
        torch.testing.assert_close(out, torch.zeros_like(x))
        out[0].fill_(3)
        out[2].fill_(3)
        return out

    monkeypatch.setattr(conv_ops, "copy_conv_state", copy_state)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", prefill)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", decode)
    result = conv_ops.causal_conv1d(x, torch.ones(4, 8), state, starts, indices, run_mode=run_mode)
    expected[4].add_(7)
    expected[6].add_(7)
    torch.testing.assert_close(backing, expected)
    torch.testing.assert_close(x, torch.ones_like(x))
    if run_mode == 1:
        torch.testing.assert_close(result[1], torch.zeros_like(result[1]))


def test_empty_batch_skips_fla_and_staging(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Empty batches must not launch a convolution or state copy")

    monkeypatch.setattr(conv_ops, "copy_conv_state", unexpected_call)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", unexpected_call)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", unexpected_call)
    x = torch.ones(1, 8)
    result = conv_ops.causal_conv1d(
        x,
        torch.ones(4, 8),
        torch.zeros(4, 3, 8)[::2],
        torch.zeros(1, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        run_mode=1,
    )
    torch.testing.assert_close(result, torch.zeros_like(x))
