# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

conv_ops = importlib.import_module("vllm_ascend.models.glm5next.ops.causal_conv1d")


@pytest.mark.parametrize("layout", ["page_strided", "row_strided", "dim_first"])
@pytest.mark.parametrize("run_mode", [0, 1])
@pytest.mark.parametrize("use_prepared", [False, True])
def test_conv_dispatch_reuses_prepared_sd_indices_and_stages_ds(monkeypatch, layout, run_mode, use_prepared):
    dim_first = layout == "dim_first"
    if dim_first:
        state = torch.zeros(4, 8, 3).transpose(1, 2)
    elif layout == "row_strided":
        state = torch.zeros(8, 6, 8)[::2, ::2]
    else:
        state = torch.zeros(8, 3, 8)[::2]
    indices = torch.tensor([[1, 3, 2, 0], [2, 1, 3, 0]], dtype=torch.int32)
    prepared = torch.tensor([1, 2], dtype=torch.int32)
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    x = torch.ones(2, 8)
    calls = []
    preparations = []

    def copy_state(cache, staging, ids, query_starts, staging_ids, *, write_back):
        calls.append(write_back)
        if not write_back:
            staging.copy_(cache[1:3])
            staging_ids.copy_(torch.tensor([0, 1], dtype=torch.int32))
        else:
            cache[1:3].copy_(staging)

    def prefill(x, weight, bias, **kwargs):
        kernel_indices = kwargs["cache_indices"]
        assert kernel_indices.ndim == 1 and kernel_indices.stride() == (1,)
        assert kwargs["null_block_id"] == (2 if dim_first else 4)
        expected = torch.tensor([0, 1] if dim_first else [1, 2], dtype=torch.int32)
        torch.testing.assert_close(kernel_indices, expected)
        assert torch.all(kernel_indices != kwargs["null_block_id"])
        if not dim_first:
            assert kwargs["conv_states"] is state
            if use_prepared:
                assert kernel_indices is prepared
        first_slot = 0 if dim_first else 1
        kwargs["conv_states"][first_slot : first_slot + 2].fill_(5)
        return x + 2

    def decode(x, state, weight, **kwargs):
        kwargs["cache_indices"] = kwargs["conv_state_indices"]
        kwargs["conv_states"] = state
        return prefill(x, weight, **kwargs)

    def prepare_indices(ids, query_starts, num_slots):
        assert dim_first or not use_prepared, "Prepared SD indices must bypass per-layer preparation"
        assert query_starts is starts
        assert num_slots == (2 if dim_first else 4)
        preparations.append(True)
        return ids if dim_first else ids[:, 0].contiguous()

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
        prepared_cache_indices=prepared if use_prepared else None,
        prepared_null_block_id=4 if use_prepared else None,
    )
    torch.testing.assert_close(result, x + 2)
    assert calls == ([False, True] if dim_first else [])
    assert preparations == ([True] if dim_first or not use_prepared else [])
    torch.testing.assert_close(state[1:3], torch.full_like(state[1:3], 5))
    assert torch.count_nonzero(state[0]).item() == 0
    assert torch.count_nonzero(state[3:]).item() == 0
