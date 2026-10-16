# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

conv_ops = importlib.import_module("vllm_ascend.models.glm5next.ops.causal_conv1d")


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("run_mode", [0, 1])
def test_conv_state_dispatch_reuses_sd_indices_and_stages_ds_state(monkeypatch, dim_first, run_mode):
    state = torch.zeros(4, 8, 3).transpose(1, 2) if dim_first else torch.zeros(8, 3, 8)[::2]
    indices = torch.tensor([[1, 0]], dtype=torch.int32)
    prepared = torch.tensor([1], dtype=torch.int32)
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
        assert kwargs["null_block_id"] == (1 if dim_first else 4)
        assert kwargs["cache_indices"].item() == (0 if dim_first else 1)
        assert kwargs["cache_indices"].item() != kwargs["null_block_id"]
        if not dim_first:
            assert kwargs["conv_states"] is state
            assert kwargs["cache_indices"] is prepared
        kwargs["conv_states"][0 if dim_first else 1].fill_(5)
        return x + 2

    def decode(x, state, weight, **kwargs):
        kwargs["cache_indices"] = kwargs["conv_state_indices"]
        kwargs["conv_states"] = state
        return prefill(x, weight, **kwargs)

    def prepare_indices(ids, query_starts, num_slots):
        assert dim_first, "Prepared SD indices must bypass per-layer preparation"
        assert ids.tolist() == [0]
        return ids

    monkeypatch.setattr(conv_ops, "copy_conv_state", copy_state)
    monkeypatch.setattr(conv_ops, "prepare_conv_indices", prepare_indices)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", prefill)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", decode)
    result = conv_ops.causal_conv1d(
        x,
        torch.ones(4, 8),
        state,
        starts,
        indices,
        run_mode=run_mode,
        prepared_cache_indices=prepared,
        prepared_null_block_id=4,
    )
    torch.testing.assert_close(result, x + 2)
    assert calls == ([False, True] if dim_first else [])
    torch.testing.assert_close(state[1], torch.full_like(state[1], 5))
    assert torch.count_nonzero(state[0]).item() == 0
    assert torch.count_nonzero(state[2:]).item() == 0
