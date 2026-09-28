# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Unit tests for the fused gating top-k / mapping / recording operator."""

from types import SimpleNamespace

import torch

from vllm_ascend.ascend_forward_context import AscendAttentionState
from vllm_ascend.ops.fused_moe.router import fused_topk_router
from vllm_ascend.ops.triton.gating_top_k_map_and_record import (
    GATING_TOP_K_MAP_RECORD_MAX_TOKENS,
)


def test_custom_ops_registered_with_fake_impl():
    """Importing the eplb ops module registers the fused ops on the vllm namespace."""
    import vllm_ascend.ops.fused_moe.eplb  # noqa: F401  (triggers registration)

    assert torch.ops.vllm.gating_top_k_map_and_record is not None
    assert torch.ops.vllm.hash_gating_top_k_map_and_record is not None


def test_bucket_threshold_matches_crossover():
    """T=512 measured parity and T=1024 a regression; the guard sits between."""
    assert 0 < GATING_TOP_K_MAP_RECORD_MAX_TOKENS <= 1024
    assert GATING_TOP_K_MAP_RECORD_MAX_TOKENS in (256, 512)


def test_is_decode_step_phase_matrix(monkeypatch):
    """Only decode steps (including MTP spec decode) may take the fused route."""
    cases = [
        (AscendAttentionState.DecodeOnly, True),
        (AscendAttentionState.SpecDecoding, True),
        (AscendAttentionState.PrefillNoCache, False),
        (AscendAttentionState.PrefillCacheHit, False),
        (AscendAttentionState.ChunkedPrefill, False),
        # Unclassified steps (dummy/profile runs) keep the unfused route.
        (None, False),
    ]
    for attn_state, expected in cases:
        monkeypatch.setattr(
            fused_topk_router,
            "_EXTRA_CTX",
            SimpleNamespace(attn_state=attn_state),
        )
        assert fused_topk_router._is_decode_step() is expected, attn_state
