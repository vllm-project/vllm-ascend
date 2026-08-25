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

from unittest.mock import patch

import torch

import vllm_ascend.models.qwen3_dspark as qwen3_dspark


class TestQwen3DSparkWeightLoading:
    """Tests for Qwen3 DSpark weight loading."""

    def test_rotates_only_fc_weights(self) -> None:
        """Rotate FC weights and preserve all other weights before delegation."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM

        # ``load_weights`` only reads ``rotation_path`` from the model. Bypass the
        # full model constructor and nn.Module attribute handling to keep this a
        # focused CPU unit test.
        model = model_cls.__new__(model_cls)
        rotation_path = "quarot.safetensors"
        object.__setattr__(model, "rotation_path", rotation_path)
        object.__setattr__(model, "_quarot_fc_rotated", False)
        object.__setattr__(model, "_quarot_fc_weights_rotated", 0)

        # Use a non-identity matrix so an unrotated FC weight fails the assertion.
        rotation_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        fc_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        non_fc_weight = torch.tensor([[5.0, 6.0]])
        weights_to_load = [("model.fc.weight", fc_weight), ("model.embed_tokens.weight", non_fc_weight)]
        expected_fc_weight = torch.matmul(fc_weight, rotation_matrix)

        # Capture the final delegation without invoking the real model loader.
        with (
            patch.object(
                qwen3_dspark, "get_rotataion_matrix", return_value=rotation_matrix
            ) as mock_get_rotation_matrix,
            patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent_load_weights,
        ):
            model.load_weights(weights_to_load)

        mock_get_rotation_matrix.assert_called_once_with(rotation_path)
        mock_parent_load_weights.assert_called_once()

        processed_weights = mock_parent_load_weights.call_args.args[0]
        torch.testing.assert_close(processed_weights[0][1], expected_fc_weight)
        torch.testing.assert_close(processed_weights[1][1], non_fc_weight)
        assert model._quarot_fc_rotated is True
        assert model._quarot_fc_weights_rotated == 1

    def test_does_not_confirm_rotation_when_batch_has_no_fc_weights(self) -> None:
        """Leave final fail-fast to the proposer after all weight batches load."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        object.__setattr__(model, "rotation_path", "quarot.safetensors")
        object.__setattr__(model, "_quarot_fc_rotated", False)
        object.__setattr__(model, "_quarot_fc_weights_rotated", 0)
        weights_to_load = [("model.embed_tokens.weight", torch.ones(1, 2))]

        with (
            patch.object(
                qwen3_dspark,
                "get_rotataion_matrix",
                return_value=torch.eye(2),
            ) as mock_get_rotation_matrix,
            patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent_load_weights,
        ):
            model.load_weights(weights_to_load)

        mock_get_rotation_matrix.assert_not_called()
        mock_parent_load_weights.assert_called_once()
        assert model._quarot_fc_rotated is False
        assert model._quarot_fc_weights_rotated == 0

    def test_non_quarot_checkpoint_delegates_without_fc_requirement(self) -> None:
        """A target without QuaRot must keep the existing loading behavior."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        object.__setattr__(model, "rotation_path", None)
        object.__setattr__(model, "_quarot_fc_rotated", False)
        object.__setattr__(model, "_quarot_fc_weights_rotated", 0)
        weights_to_load = [("model.embed_tokens.weight", torch.ones(1, 2))]
        sentinel = object()

        with patch.object(
            qwen3_dspark.Qwen3DSparkForCausalLM,
            "load_weights",
            return_value=sentinel,
        ) as mock_parent_load_weights:
            result = model.load_weights(weights_to_load)

        assert result is sentinel
        mock_parent_load_weights.assert_called_once_with(weights_to_load)
        assert model._quarot_fc_rotated is False
        assert model._quarot_fc_weights_rotated == 0

    def test_fc_rotation_state_survives_later_weight_batches(self) -> None:
        """Incremental loaders may invoke load_weights more than once."""
        model_cls = qwen3_dspark.AscendQwen3DSparkForCausalLM
        model = model_cls.__new__(model_cls)
        object.__setattr__(model, "rotation_path", "quarot.safetensors")
        object.__setattr__(model, "_quarot_fc_rotated", False)
        object.__setattr__(model, "_quarot_fc_weights_rotated", 0)
        fc_weights = [("model.fc.weight", torch.eye(2))]
        other_weights = [("model.embed_tokens.weight", torch.ones(1, 2))]

        with (
            patch.object(
                qwen3_dspark,
                "get_rotataion_matrix",
                return_value=torch.eye(2),
            ) as mock_get_rotation_matrix,
            patch.object(qwen3_dspark.Qwen3DSparkForCausalLM, "load_weights") as mock_parent_load_weights,
        ):
            model.load_weights(other_weights)
            model.load_weights(fc_weights)
            model.load_weights(other_weights)

        assert mock_parent_load_weights.call_count == 3
        mock_get_rotation_matrix.assert_called_once_with("quarot.safetensors")
        assert model._quarot_fc_rotated is True
        assert model._quarot_fc_weights_rotated == 1
