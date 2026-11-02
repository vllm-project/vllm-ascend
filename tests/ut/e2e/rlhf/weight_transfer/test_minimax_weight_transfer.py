# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax checkpoint conversion and fused-projection regression checks."""

from unittest.mock import MagicMock

import pytest
import torch

from tests.e2e.pull_request.rlhf.weight_transfer.models.minimax import MINIMAX_M3_CASE, minimax_m3_checkpoint_name
from tests.ut.e2e.rlhf.weight_transfer.weight_transfer_test_helpers import checkpoint_source


def test_minimax_checkpoint_source_splits_fused_projections():
    model = MagicMock()
    model.named_parameters.return_value = [
        ("model.layers.0.mlp.gate_up_proj.weight", torch.empty((12, 16), device="meta")),
        ("model.layers.3.mlp.shared_experts.gate_up_proj.weight", torch.empty((12, 16), device="meta")),
        ("model.layers.3.mlp.experts.gate_up_proj", torch.empty((2, 12, 16), device="meta")),
    ]
    model.named_buffers.return_value = []
    model.state_dict.return_value = {}
    source = checkpoint_source(model, MINIMAX_M3_CASE)
    actual = {meta.name: (meta.shape, meta.dtype) for meta in source.metadata()}
    expected_names = {
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj.weight",
        *{
            f"language_model.model.layers.3.block_sparse_moe.experts.{expert}.{shard}.weight"
            for expert in range(2)
            for shard in ("w1", "w3")
        },
    }
    assert actual == {name: ((6, 16), torch.bfloat16) for name in expected_names}


@pytest.mark.parametrize("module", ["q_proj", "k_proj", "q_norm", "k_norm"])
def test_minimax_indexer_mapping_accepts_published_weights(module):
    prefix = "language_model.model.layers.3.self_attn"
    assert minimax_m3_checkpoint_name(f"{prefix}.indexer.{module}.weight") == f"{prefix}.index_{module}.weight"


@pytest.mark.parametrize("suffix", ["new_proj.weight", "q_proj.nested.weight", "q_norm.bias", "nested.q_proj.weight"])
def test_minimax_indexer_mapping_rejects_unknown_state(suffix):
    name = f"language_model.model.layers.3.self_attn.indexer.{suffix}"
    with pytest.raises(ValueError, match="Unsupported MiniMax-M3 indexer parameter"):
        minimax_m3_checkpoint_name(name)
