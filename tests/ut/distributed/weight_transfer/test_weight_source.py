# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Checkpoint payload coverage without downloading checkpoint weights."""

from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig

from tests.e2e.common.rlhf.weight_transfer import weight_source as utils
from tests.e2e.common.rlhf.weight_transfer.case import MINIMAX_M3_CASE


def test_minimax_native_checkpoint_metadata():
    # Use the real native config/model so missing registrations and HF layout
    # changes fail here instead of being hidden behind a mocked meta model.
    config = AutoConfig.for_model("minimax_m3_vl")
    with patch.object(utils.AutoConfig, "from_pretrained", return_value=config) as load_config:
        source = utils.FixedRandomWeightSource(MINIMAX_M3_CASE, torch.device("cpu"))
    load_config.assert_called_once_with(MINIMAX_M3_CASE.model, trust_remote_code=False)
    metadata = {meta.name: (meta.shape, meta.dtype) for meta in source.metadata}
    prefix = "language_model.model"
    assert len(metadata) == 77
    expected = {
        f"{prefix}.layers.0.mlp.gate_proj.weight": (12288, 6144),
        f"{prefix}.layers.0.mlp.up_proj.weight": (12288, 6144),
        f"{prefix}.layers.3.self_attn.index_q_proj.weight": (512, 6144),
        f"{prefix}.layers.3.self_attn.index_k_norm.weight": (128,),
        f"{prefix}.layers.3.block_sparse_moe.shared_experts.gate_proj.weight": (3072, 6144),
        "language_model.lm_head.weight": (200064, 6144),
        **{
            f"{prefix}.layers.3.block_sparse_moe.experts.{expert}.{shard}.weight": shape
            for expert in range(8)
            for shard, shape in (("w1", (3072, 6144)), ("w2", (6144, 3072)), ("w3", (3072, 6144)))
        },
    }
    for name, shape in expected.items():
        assert metadata[name] == (shape, torch.bfloat16)
    bias = f"{prefix}.layers.3.block_sparse_moe.e_score_correction_bias"
    assert metadata[bias] == ((8,), torch.bfloat16)
    assert all(dtype == torch.bfloat16 for _, dtype in metadata.values())
    assert not any("gate_up_proj" in name or "vision" in name for name in metadata)


def test_fused_projection_rejects_odd_dimension():
    with pytest.raises(AssertionError, match="odd fused gate/up dimension"):
        utils.split_fused_mlp_params("model.layers.0.mlp.gate_up_proj.weight", (7, 16))
