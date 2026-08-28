#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend embed_tokensect.
#
"""CPU-only tests for Qwen3 DSpark weight loading."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

import vllm_ascend.models.qwen3_dspark as qwen3_dspark


class TestQwen3DSparkWeightLoading:
    """Tests for Qwen3 DSpark weight loading."""

    def test_rotates_only_fc_weights(self) -> None:
        """Rotate FC weights and preserve all other weights before delegation."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM

        # ``load_weights`` only reads ``rotation_path`` / ``enable_confidence_head``
        # from the model. Bypass the full model constructor and nn.Module
        # attribute handling to keep this a focused CPU unit test.
        model = model_cls.__new__(model_cls)
        rotation_path = "quarot.safetensors"
        object.__setattr__(model, "rotation_path", rotation_path)
        object.__setattr__(model, "enable_confidence_head", False)

        # Use a non-identity matrix so an unrotated FC weight fails the assertion.
        rotation_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        fc_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        non_fc_weight = torch.tensor([[5.0, 6.0]])
        weights_to_load = [
            ("model.fc.weight", fc_weight),
            ("model.embed_tokens.weight", non_fc_weight),
            ("lm_head.weight", non_fc_weight),
        ]
        expected_fc_weight = torch.matmul(fc_weight, rotation_matrix)

        # Capture the final delegation without invoking the real model loader.
        with (
            patch.object(qwen3_dspark, "get_rotation_matrix", return_value=rotation_matrix) as mock_get_rotation_matrix,
            patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent_load_weights,
        ):
            # DefaultModelLoader passes a one-shot iterator. Exercise that path
            # so materializing the weights before rotation cannot exhaust them.
            model.load_weights(iter(weights_to_load))

        mock_get_rotation_matrix.assert_called_once_with(rotation_path)
        mock_parent_load_weights.assert_called_once()

        processed_weights = mock_parent_load_weights.call_args.args[0]
        torch.testing.assert_close(processed_weights[0][1], expected_fc_weight)
        torch.testing.assert_close(processed_weights[1][1], non_fc_weight)
        torch.testing.assert_close(processed_weights[2][1], non_fc_weight)


def test_quarot_loads_missing_target_vocab_shards(tmp_path) -> None:
    embed_name = "language_model.model.embed_tokens.weight"
    head_name = "language_model.lm_head.weight"
    shard_name = "model-00001-of-00001.safetensors"
    target_weight = torch.arange(8, dtype=torch.float32).view(4, 2)
    save_file({embed_name: target_weight, head_name: target_weight + 10}, tmp_path / shard_name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {embed_name: shard_name, head_name: shard_name}}),
        encoding="utf-8",
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotation_path = tmp_path / "rotation.safetensors"
    save_file({"global_rotation": rotation}, rotation_path)

    model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.rotation_path = rotation_path
    model.target_model_path = tmp_path
    model.enable_confidence_head = False
    model.model = SimpleNamespace(embed_tokens=nn.Linear(2, 3, bias=False))
    model.lm_head = nn.Linear(2, 3, bias=False)
    for layer in (model.model.embed_tokens, model.lm_head):
        layer.weight.data.fill_(99)
        layer.shard_indices = SimpleNamespace(org_vocab_start_index=1, org_vocab_end_index=3)

    # Draft omits its vocabulary weights; load and align the target's local TP shard.
    with patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights"):
        model.load_weights(iter([]))

    for layer, weight in ((model.model.embed_tokens, target_weight), (model.lm_head, target_weight + 10)):
        expected = torch.cat((weight[1:3] @ rotation.T, torch.zeros(1, 2)))
        torch.testing.assert_close(layer.weight, expected)
    assert model.has_own_embed_tokens
    assert model.has_own_lm_head
