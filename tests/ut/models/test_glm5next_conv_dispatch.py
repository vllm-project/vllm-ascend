# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from unittest.mock import patch

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
        assert kwargs["null_block_id"] == (-1 if packed else 0)
        assert kwargs["cache_indices"].item() == (0 if packed else 1)
        assert kwargs["cache_indices"].item() != kwargs["null_block_id"]
        return x + 2

    def decode(x, state, weight, **kwargs):
        kwargs["cache_indices"] = kwargs["conv_state_indices"]
        return prefill(x, weight, **kwargs)

    monkeypatch.setattr(conv_ops, "copy_conv_state", copy_state)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", prefill)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", decode)
    with patch.object(torch, "zeros_like", wraps=torch.zeros_like) as zeros:
        result = conv_ops.causal_conv1d(x, torch.ones(4, 8), state, starts, indices, run_mode=run_mode)
    zeros.assert_not_called()
    torch.testing.assert_close(result, x + 2)
    assert calls == ([False, True] if packed else [])


@pytest.mark.parametrize("run_mode", [0, 1])
def test_empty_request_batch_keeps_zero_output(run_mode):
    x = torch.ones(2, 8)
    with (
        patch.object(conv_ops, "causal_conv1d_fn") as prefill,
        patch.object(conv_ops, "causal_conv1d_update") as decode,
    ):
        result = conv_ops.causal_conv1d(
            x,
            torch.ones(4, 8),
            torch.zeros(4, 3, 8),
            torch.tensor([0], dtype=torch.int32),
            torch.empty((0, 1), dtype=torch.int32),
            run_mode=run_mode,
        )
    prefill.assert_not_called()
    decode.assert_not_called()
    torch.testing.assert_close(result, torch.zeros_like(x))
