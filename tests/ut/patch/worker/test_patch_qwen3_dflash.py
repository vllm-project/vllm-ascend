# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding
from vllm_ascend.patch.worker import patch_qwen3_dflash as module


@pytest.mark.parametrize(
    "v2,triton,standard", [(True, True, True), (False, True, True), (True, False, True), (True, True, False)]
)
@pytest.mark.parametrize("per_layer_slots", [False, True])
@pytest.mark.parametrize("stack_budget", [0, 4 * 1024 * 1024])
def test_context_rotates_only_k_when_supported_and_preserves_each_layers_v_and_slots(
    monkeypatch, v2, triton, standard, per_layer_slots, stack_budget
):
    monkeypatch.setattr(module, "_MAX_STACKED_CONTEXT_NORM_BYTES", stack_budget)
    monkeypatch.setattr(module, "HAS_TRITON", triton)
    monkeypatch.setattr(module, "get_current_vllm_config", lambda: SimpleNamespace(use_v2_model_runner=v2))
    monkeypatch.setattr(module, "_orig_build_fused_kv_buffers", lambda model: None)
    captured = []
    stack = MagicMock(wraps=torch.stack)
    monkeypatch.setattr(torch, "stack", stack)

    def rotate(positions, query, key):
        captured.append((positions.clone(), key))
        if key is None:
            # A supported backend may return a replacement query tensor.
            return query + 2, None
        query.add_(1)
        key.add_(99)
        return query, key

    if standard:
        rotary = AscendRotaryEmbedding.__new__(AscendRotaryEmbedding)
        torch.nn.Module.__init__(rotary)
        rotary.forward = rotate
    else:
        rotary = rotate
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
        layers=[SimpleNamespace(self_attn=SimpleNamespace(k_norm=lambda x: x, rotary_emb=rotary)) for _ in range(2)],
        _attn_layers=[
            SimpleNamespace(kv_cache=cache, impl=SimpleNamespace(do_kv_cache_update=update))
            for cache, update in zip(caches, updates)
        ],
    )
    states = torch.arange(9, dtype=torch.float32).view(3, 3) / 10
    positions = torch.tensor([2, 4, 8])
    model._build_fused_kv_buffers = lambda: module._build_fused_kv_buffers(model)
    slots = [torch.tensor([3, 7, 9]), torch.tensor([11, 15, 17])] if per_layer_slots else torch.tensor([3, 7, 9])
    expected = torch.nn.functional.linear(states, model._fused_kv_weight).view(3, 2, 2, 2, 4)
    module.precompute_and_store_context_kv(model, states, positions, slots)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][0], positions.repeat(2))
    query_only = v2 and triton and standard
    assert (captured[0][1] is None) == query_only
    for i, update in enumerate(updates):
        update.assert_called_once()
        attn, k, v, cache, current_slots = update.call_args.args
        assert attn is model._attn_layers[i]
        assert cache is caches[i]
        assert current_slots is (slots[i] if per_layer_slots else slots)
        torch.testing.assert_close(k, expected[:, i, 0] + (2 if query_only else 1))
        torch.testing.assert_close(v, expected[:, i, 1], rtol=0, atol=0)

    # A proposal has no current-config scope. The gate must already be bound,
    # while context values and positions still come from this proposal.
    def unavailable_config():
        raise AssertionError("runtime proposal must not query current config")

    monkeypatch.setattr(module, "get_current_vllm_config", unavailable_config)
    captured.clear()
    for update in updates:
        update.reset_mock()
    states = states + 1
    positions = positions + 3
    expected = torch.nn.functional.linear(states, model._fused_kv_weight).view(3, 2, 2, 2, 4)
    module.precompute_and_store_context_kv(model, states, positions, slots)
    torch.testing.assert_close(captured[0][0], positions.repeat(2))
    for i, update in enumerate(updates):
        _, k, v, _, current_slots = update.call_args.args
        assert current_slots is (slots[i] if per_layer_slots else slots)
        torch.testing.assert_close(k, expected[:, i, 0] + (2 if query_only else 1))
        torch.testing.assert_close(v, expected[:, i, 1], rtol=0, atol=0)

    assert stack.call_count == (2 if query_only and stack_budget else 0)
