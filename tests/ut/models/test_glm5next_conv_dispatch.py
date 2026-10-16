# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

conv_ops = importlib.import_module("vllm_ascend.models.glm5next.ops.causal_conv1d")


@pytest.mark.parametrize("layout", ["contiguous", "page_strided", "row_strided", "dim_first"])
@pytest.mark.parametrize("run_mode", [0, 1])
def test_conv_state_updates_original_cache_and_preserves_unselected_slots(monkeypatch, layout, run_mode):
    state = torch.zeros(4, 3, 8)
    if layout == "page_strided":
        state = torch.zeros(8, 3, 8)[::2]
    elif layout == "row_strided":
        state = torch.zeros(4, 6, 8)[:, ::2]
    elif layout == "dim_first":
        state = torch.zeros(4, 8, 3).transpose(1, 2)
    packed = layout == "dim_first"
    indices = torch.tensor([[1, 0]], dtype=torch.int32)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.ones(1, 8)
    calls = []

    def copy_state(cache, staging, ids, query_starts, staging_ids, *, write_back):
        calls.append(write_back)
        if not write_back:
            staging.copy_(cache[1:2])
            staging_ids.zero_()
        else:
            cache[1].copy_(staging[0])

    def prefill(x, weight, bias, **kwargs):
        assert kwargs["cache_indices"].ndim == 1
        assert kwargs["null_block_id"] == (1 if packed else 4)
        assert kwargs["cache_indices"].item() == (0 if packed else 1)
        assert kwargs["cache_indices"].item() != kwargs["null_block_id"]
        if not packed:
            assert kwargs["conv_states"] is state
        kwargs["conv_states"][0 if packed else 1].fill_(5)
        return x + 2

    def decode(x, state, weight, **kwargs):
        kwargs["cache_indices"] = kwargs["conv_state_indices"]
        kwargs["conv_states"] = state
        return prefill(x, weight, **kwargs)

    def prepare_indices(ids, query_starts, num_slots):
        flat = ids[:, 0] if ids.ndim == 2 else ids
        active = (flat >= 0) & (flat < num_slots) & (query_starts[1:] > query_starts[:-1])
        return torch.where(active, flat, num_slots).to(torch.int32)

    monkeypatch.setattr(conv_ops, "copy_conv_state", copy_state)
    monkeypatch.setattr(conv_ops, "prepare_conv_indices", prepare_indices)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", prefill)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", decode)
    result = conv_ops.causal_conv1d(x, torch.ones(4, 8), state, starts, indices, run_mode=run_mode)
    torch.testing.assert_close(result, x + 2)
    assert calls == ([False, True] if packed else [])
    torch.testing.assert_close(state[1], torch.full_like(state[1], 5))
    assert torch.count_nonzero(state[0]).item() == 0
    assert torch.count_nonzero(state[2:]).item() == 0


@pytest.mark.parametrize("run_mode", [0, 1])
def test_prepared_indices_are_reused_without_per_layer_preparation(monkeypatch, run_mode):
    state = torch.zeros(8, 3, 8)[::2]
    starts = torch.tensor([0, 1, 1], dtype=torch.int32)
    raw = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32)
    prepared = torch.tensor([0, 4], dtype=torch.int32)
    x = torch.ones(1, 8)

    def unexpected(*args, **kwargs):
        pytest.fail("Prepared SD states must not launch index preparation or state copies per layer")

    def prefill(value, weight, bias, **kwargs):
        assert kwargs["cache_indices"] is prepared
        assert kwargs["conv_states"] is state
        assert kwargs["null_block_id"] == 4
        return value

    def decode(value, cache, weight, **kwargs):
        assert kwargs["conv_state_indices"] is prepared
        assert cache is state
        assert kwargs["null_block_id"] == 4
        return value

    monkeypatch.setattr(conv_ops, "prepare_conv_indices", unexpected)
    monkeypatch.setattr(conv_ops, "copy_conv_state", unexpected)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", prefill)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", decode)
    result = conv_ops.causal_conv1d(
        x,
        torch.ones(4, 8),
        state,
        starts,
        raw,
        run_mode=run_mode,
        prepared_cache_indices=prepared,
        prepared_null_block_id=4,
    )
    assert result is x


def test_prepared_indices_require_their_null_slot():
    with pytest.raises(ValueError, match="null_block_id"):
        conv_ops.causal_conv1d(
            torch.ones(1, 8),
            torch.ones(4, 8),
            torch.zeros(4, 3, 8),
            torch.tensor([0, 1]),
            torch.tensor([0]),
            run_mode=1,
            prepared_cache_indices=torch.tensor([0], dtype=torch.int32),
        )
