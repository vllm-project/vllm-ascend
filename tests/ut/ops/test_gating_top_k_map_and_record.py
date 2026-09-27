# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Unit tests for the fused gating top-k / mapping / recording operator."""

import torch

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
