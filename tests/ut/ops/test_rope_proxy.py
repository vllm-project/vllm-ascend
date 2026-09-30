# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE, RopeDataProxy, get_cos_and_sin_dsa

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


def test_get_cos_and_sin_filters_configs_by_consuming_layer(monkeypatch):
    regular_cos = torch.arange(32, dtype=torch.float32).view(8, 1, 1, 4)
    regular_sin = regular_cos + 100
    compressed_cos = regular_cos + 200
    compressed_sin = regular_cos + 300
    monkeypatch.setattr(
        _ROPE_STATE,
        "full_rope_cache",
        {
            "regular": (regular_cos, regular_sin),
            "compressed": (compressed_cos, compressed_sin),
        },
    )
    monkeypatch.setattr(
        _ROPE_STATE,
        "registry_summary",
        {
            "regular": {"default"},
            "compressed": {"default", "c4"},
        },
    )
    monkeypatch.setattr(
        _ROPE_STATE,
        "layer_info",
        {
            "mtp.0.attn": ("regular", ["default"]),
            "model.layers.3.attn": ("compressed", ["default", "c4"]),
        },
    )

    positions = torch.tensor([1, 4, 7])
    cos, sin = get_cos_and_sin_dsa(positions, layer_names="mtp.0.attn")

    assert set(cos._data) == {"regular"}
    assert set(sin._data) == {"regular"}
    assert torch.equal(cos["mtp.0.attn"], regular_cos[positions])
    assert torch.equal(sin["mtp.0.attn"], regular_sin[positions])


def test_get_cos_and_sin_rejects_unknown_layer():
    with pytest.raises(KeyError, match="missing.attn"):
        get_cos_and_sin_dsa(torch.tensor([0]), layer_names="missing.attn")


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
