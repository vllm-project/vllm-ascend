# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.patch.worker import patch_qwen3_dflash as module


@pytest.mark.parametrize("per_layer_slots", [False, True])
@pytest.mark.parametrize("stack_budget", [0, 4 * 1024 * 1024])
def test_context_norm_stacking_respects_the_byte_budget(monkeypatch, per_layer_slots, stack_budget):
    """The context path stacks the per-layer norms only within the byte budget."""
    monkeypatch.setattr(module, "_MAX_STACKED_CONTEXT_NORM_BYTES", stack_budget)
    captured = []
    stack = MagicMock(wraps=torch.stack)
    monkeypatch.setattr(torch, "stack", stack)

    def rotate(pos, query, key):
        # The context path always passes K only.
        assert key is None
        captured.append(pos.clone())
        return query + 2, None

    updates = [MagicMock(), MagicMock()]
    caches = [object(), object()]
    model = SimpleNamespace(
        _num_attn_layers=2,
        _kv_size=8,
        _head_dim=4,
        _num_kv_heads=2,
        _fused_kv_weight=torch.arange(32 * 3, dtype=torch.float32).view(32, 3) / 100,
        _fused_kv_bias=None,
        hidden_norm=lambda x: x,
        layers=[SimpleNamespace(self_attn=SimpleNamespace(k_norm=lambda x: x, rotary_emb=rotate)) for _ in range(2)],
        _attn_layers=[
            SimpleNamespace(kv_cache=cache, impl=SimpleNamespace(do_kv_cache_update=update))
            for cache, update in zip(caches, updates)
        ],
    )
    model._project_context_kv = lambda *args: module._project_context_kv(model, *args)

    states = torch.arange(9, dtype=torch.float32).view(3, 3) / 10
    positions = torch.tensor([2, 4, 8])
    slots = [torch.tensor([3, 7, 9]), torch.tensor([11, 15, 17])] if per_layer_slots else torch.tensor([3, 7, 9])
    expected = torch.nn.functional.linear(states, model._fused_kv_weight).view(3, 2, 2, 2, 4)

    module.precompute_and_store_context_kv(model, states, positions, slots)

    torch.testing.assert_close(captured[0], positions.repeat(2))
    assert stack.call_count == (1 if stack_budget else 0)
    for i, update in enumerate(updates):
        update.assert_called_once()
        attn, k, v, cache, current_slots = update.call_args.args
        assert attn is model._attn_layers[i]
        assert cache is caches[i]
        assert current_slots is (slots[i] if per_layer_slots else slots)
        torch.testing.assert_close(k, expected[:, i, 0] + 2)
        torch.testing.assert_close(v, expected[:, i, 1], rtol=0, atol=0)
