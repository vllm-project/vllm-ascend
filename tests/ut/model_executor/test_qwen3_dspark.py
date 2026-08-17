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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm_ascend.models.qwen3_dspark as qwen3_dspark
import vllm_ascend.models.qwen3_omni_dspark as qwen3_omni_dspark


class TestQwen3DSparkWeightLoading:
    """Tests for Qwen3 DSpark weight loading."""

    def test_qwen3_omni_uses_dedicated_backbone(self) -> None:
        assert qwen3_omni_dspark.AscendQwen3OmniDSparkForCausalLM.model_cls is qwen3_omni_dspark.Qwen3OmniDSparkModel

    def test_rotation_path_uses_target_model_quantization(self) -> None:
        target_model_config = SimpleNamespace(
            model="/models/qwen3-omni",
            quantization="ascend",
        )
        vllm_config = SimpleNamespace(
            model_config=target_model_config,
            load_config=SimpleNamespace(),
            quant_config=None,
        )
        target_quant_config = SimpleNamespace(
            quant_description={"optional": {"quarot": {"rotation_map": {"global_rotation": "rotation.safetensors"}}}}
        )

        with patch.object(
            qwen3_dspark.VllmConfig,
            "get_quantization_config",
            return_value=target_quant_config,
        ) as mock_get_quantization_config:
            result = qwen3_dspark.get_rotation_path(vllm_config)

        mock_get_quantization_config.assert_called_once_with(target_model_config, vllm_config.load_config)
        assert result == Path("/models/qwen3-omni/rotation.safetensors")

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
        weights_to_load = [("model.fc.weight", fc_weight), ("model.embed_tokens.weight", non_fc_weight)]
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
