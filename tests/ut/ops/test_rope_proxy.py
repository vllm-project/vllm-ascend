# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import rope_dsv4
from vllm_ascend.ops.rope_dsv4 import (
    _ROPE_STATE,
    ComplexExpRotaryEmbedding,
    RopeDataProxy,
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)


def test_full_rope_lookup_resolves_exact_layer_config(monkeypatch):
    first = (torch.randn(4, 1, 1, 8), torch.randn(4, 1, 1, 8))
    second = (torch.randn(4, 1, 1, 8), torch.randn(4, 1, 1, 8))
    monkeypatch.setattr(
        rope_dsv4._ROPE_STATE,
        "layer_info",
        {
            "model.layers.0.self_attn.attn": ("base", ["default"]),
            "model.layers.2.self_attn.attn": ("compressed", ["default"]),
        },
    )
    monkeypatch.setattr(
        rope_dsv4._ROPE_STATE,
        "full_rope_cache",
        {"base": first, "compressed": second},
    )

    actual = get_full_cos_and_sin_dsa_for_layer("model.layers.2.self_attn.attn")

    assert actual[0] is second[0]
    assert actual[1] is second[1]
    with pytest.raises(KeyError, match="not registered"):
        get_full_cos_and_sin_dsa_for_layer("missing")


def test_plain_rope_disables_yarn_explicitly():
    dim = 8
    base = 10000
    actual = ComplexExpRotaryEmbedding.precompute_freqs_cis(
        dim,
        seqlen=65536,
        original_seq_len=4096,
        apply_yarn_scaling=False,
        base=base,
        factor=16,
        beta_fast=32,
        beta_slow=1,
    )
    expected = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    torch.testing.assert_close(actual, expected)


# ──────────────────────────────────────────────
# Equivalence: pad_to + slice  vs  pad-positions + gather + slice
# ──────────────────────────────────────────────


def _gather_rope(
    positions: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
) -> tuple[RopeDataProxy, RopeDataProxy]:
    """Index ``rope_cos`` / ``rope_sin`` by ``positions`` and wrap in ``RopeDataProxy``.

    This mirrors what ``get_cos_and_sin_dsa`` does internally — lookup the
    RoPE table at the given positions — without depending on the global
    ``_ROPE_STATE`` singleton.
    """
    cos_t = rope_cos[positions]  # [N, 1, 1, D]
    sin_t = rope_sin[positions]
    data_map = {"_test": {"default": (cos_t, sin_t)}}
    return RopeDataProxy(data_map, is_cos=True), RopeDataProxy(data_map, is_cos=False)


def _extract_tensor(proxy: RopeDataProxy) -> torch.Tensor:
    """Extract the raw tensor from a singled-group proxy for comparison."""
    for groups in proxy._data.values():
        for tensors in groups.values():
            return tensors[proxy.idx]
    raise AssertionError("empty proxy")


def _tp_params(num_input_tokens: int, tp_size: int):
    """Yield ``(tp_rank, local_start, local_end, num_tokens_pad)`` for each TP rank."""
    num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    for tp_rank in range(tp_size):
        local_start = tp_rank * tokens_per_rank
        local_end = local_start + tokens_per_rank
        yield tp_rank, local_start, local_end, num_tokens_pad


class TestEquivalenceWithGatherSemantics:
    """Verify that ``proxy.pad_to(N)[s:e]`` is semantically equivalent to
    the original ``get_cos_and_sin_dsa`` approach of padding positions first.
    """

    # (num_input_tokens, tp_size)
    CASES = [
        (32, 8),
        (33, 8),
        (39, 8),
        (40, 8),
        (45, 8),
        (1, 2),
        (2, 4),
        (7, 8),
        (100, 16),
        (102, 16),
    ]

    def test_all_cases(self):
        for num_input_tokens, tp_size in self.CASES:
            self._run_equivalence(num_input_tokens, tp_size)

    def _run_equivalence(self, num_input_tokens: int, tp_size: int):
        """Run the equivalence check for one (N, tp_size) combination."""
        max_pos = num_input_tokens + tp_size + 5  # rope table large enough
        rotary_dim = 32
        rng = torch.Generator().manual_seed(42)
        rope_cos = torch.randn(max_pos, 1, 1, rotary_dim, generator=rng)
        rope_sin = torch.randn(max_pos, 1, 1, rotary_dim, generator=rng)
        input_positions = torch.randint(0, max_pos - 1, (num_input_tokens,), generator=rng)

        # Gather from UNPADDED positions — this is what the optimised path does.
        ref_cos_proxy, ref_sin_proxy = _gather_rope(input_positions, rope_cos, rope_sin)

        for tp_rank, local_start, local_end, num_tokens_pad in _tp_params(num_input_tokens, tp_size):
            # ── Original path: pad positions → gather → slice ──
            padded_pos = torch.nn.functional.pad(input_positions, (0, num_tokens_pad - num_input_tokens), value=0)
            orig_cos_p, orig_sin_p = _gather_rope(padded_pos, rope_cos, rope_sin)
            orig_cos = _extract_tensor(orig_cos_p[local_start:local_end])
            orig_sin = _extract_tensor(orig_sin_p[local_start:local_end])

            # ── Optimised path: gather → pad_to → slice ──
            opt_cos_p = ref_cos_proxy.pad_to(num_tokens_pad)
            opt_sin_p = ref_sin_proxy.pad_to(num_tokens_pad)
            opt_cos = _extract_tensor(opt_cos_p[local_start:local_end])
            opt_sin = _extract_tensor(opt_sin_p[local_start:local_end])

            # ── Compare ──
            # Real-token region: exact match expected.
            real_end = min(local_end, num_input_tokens) - local_start
            if real_end > 0:
                assert torch.equal(orig_cos[:real_end], opt_cos[:real_end]), (
                    f"cos mismatch in real region, N={num_input_tokens}, tp_size={tp_size}, rank={tp_rank}"
                )
                assert torch.equal(orig_sin[:real_end], opt_sin[:real_end]), (
                    f"sin mismatch in real region, N={num_input_tokens}, tp_size={tp_size}, rank={tp_rank}"
                )


def test_get_cos_and_sin_filters_configs_by_layer():
    old_state = (
        _ROPE_STATE.full_rope_cache,
        _ROPE_STATE.registry_summary,
        _ROPE_STATE.layer_info,
    )
    try:
        ordinary = torch.arange(24, dtype=torch.float32).view(6, 1, 1, 4)
        compressed = ordinary + 100
        _ROPE_STATE.full_rope_cache = {
            "ordinary": (ordinary, -ordinary),
            "compressed": (compressed, -compressed),
        }
        _ROPE_STATE.registry_summary = {
            "ordinary": {"default"},
            "compressed": {"default"},
        }
        _ROPE_STATE.layer_info = {
            "mtp.0.self_attn.attn": ("ordinary", ["default"]),
            "model.layers.0.self_attn.attn": ("compressed", ["default"]),
        }

        cos, sin = get_cos_and_sin_dsa(
            torch.tensor([1, 3]),
            layer_names="mtp.0.self_attn.swa_cache",
        )

        assert set(cos._data) == {"ordinary"}
        assert set(sin._data) == {"ordinary"}
        assert torch.equal(cos["mtp.0.self_attn.attn"], ordinary[[1, 3]])
        assert torch.equal(sin["mtp.0.self_attn.attn"], -ordinary[[1, 3]])
    finally:
        (
            _ROPE_STATE.full_rope_cache,
            _ROPE_STATE.registry_summary,
            _ROPE_STATE.layer_info,
        ) = old_state


@pytest.mark.parametrize("shared_config", [True, False])
def test_dspark_context_rope_reuses_all_layer_configs(monkeypatch, shared_config):
    from vllm_ascend.models.deepseek_v4 import dspark

    names = [f"mtp.{i}.self_attn.attn" for i in range(3)]
    configs = ["first", "first" if shared_config else "second", "first"]
    tables = {
        key: torch.arange(24, dtype=torch.float32, device="cpu").view(6, 1, 1, 4) + offset
        for key, offset in [("first", 0), ("second", 100), ("unused", 200)]
    }
    monkeypatch.setattr(_ROPE_STATE, "full_rope_cache", {key: (table, -table) for key, table in tables.items()})
    monkeypatch.setattr(_ROPE_STATE, "registry_summary", {key: {"default"} for key in tables})
    monkeypatch.setattr(_ROPE_STATE, "layer_info", {name: (key, ["default"]) for name, key in zip(names, configs)})
    positions = torch.tensor([1, 3], device="cpu")
    states = torch.zeros(2, 4, device="cpu")
    slots = [torch.tensor([i, i + 1], device="cpu") for i in range(3)]
    layers = {
        name: SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=SimpleNamespace(layername=name))) for name in names
    }
    lookup = Mock(wraps=get_cos_and_sin_dsa)
    monkeypatch.setattr(dspark, "get_cos_and_sin_dsa", lookup)
    seen = []

    def project(hidden_states, input_positions, attn, rope):
        assert hidden_states is states
        assert input_positions is positions
        cos, sin = rope
        assert set(cos._data) == set(configs)
        assert set(sin._data) == set(configs)
        name = attn.rotary_emb.layername
        table = tables[configs[names.index(name)]]
        assert torch.equal(cos[name], table[positions])
        assert torch.equal(sin[name], -table[positions])
        seen.append(rope)
        return hidden_states

    model = SimpleNamespace(
        layers=layers,
        _project_shared_kv=Mock(side_effect=project),
        _store_standard_swa_kv=Mock(),
    )
    dspark.DeepseekV4DSparkModel.precompute_and_store_context_kv(model, states, positions, slots)

    lookup.assert_called_once_with(positions, layer_names=names)
    assert len(seen) == len(layers)
    assert all(rope is seen[0] for rope in seen)
    assert model._store_standard_swa_kv.call_count == len(layers)
    for call, slot, layer in zip(model._store_standard_swa_kv.call_args_list, slots, layers.values()):
        assert call.args[0] is states
        assert call.args[1] is slot
        assert call.args[2] is layer.self_attn
