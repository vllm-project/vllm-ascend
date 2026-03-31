#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.sharded_state_loader_310p import ShardedStateLoader310


class MockQuantConfig:
    """Mock quantization config for testing."""

    def __init__(self, quant_type: str = "FLOAT"):
        self.quant_description = {"model_quant_type": quant_type}


class MockModel(torch.nn.Module):
    """Mock model for testing."""

    def __init__(self, quant_config=None, with_int_weights: bool = False):
        super().__init__()
        self.quant_config = quant_config
        self.with_int_weights = with_int_weights
        if with_int_weights:
            self.linear = torch.nn.Linear(10, 10)
            self.linear.weight = torch.nn.Parameter(
                torch.randint(-127, 127, (10, 10), dtype=torch.int8), requires_grad=False
            )
            self.linear.bias = torch.nn.Parameter(torch.zeros(10, dtype=torch.int32), requires_grad=False)
        else:
            self.linear = torch.nn.Linear(10, 10)


class TestShardedStateLoader310(TestBase):
    """Test cases for ShardedStateLoader310."""

    @patch("vllm.model_executor.model_loader.ShardedStateLoader._filter_subtensors")
    @patch("vllm.distributed.get_tensor_model_parallel_rank")
    @patch("safetensors.torch.save_file")
    def test_save_model_with_nd_format_310(self, mock_save_file, mock_get_rank, mock_filter):
        """Test save_model with ND format tensors (no conversion needed)."""
        mock_get_rank.return_value = 0
        mock_filter.side_effect = lambda x: x
        mock_tensor = MagicMock(spec=torch.Tensor)

        model = MockModel()
        with (
            patch.object(model, "state_dict", return_value={"linear.weight": mock_tensor}),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            ShardedStateLoader310.save_model(model, tmpdir)

            mock_save_file.assert_called_once()

    @patch("vllm.model_executor.model_loader.ShardedStateLoader._filter_subtensors")
    def test_generate_quant_description_float_model_310(self, mock_filter):
        """Test generate_quant_description for float model."""
        mock_filter.side_effect = lambda x: x
        quant_config = MockQuantConfig(quant_type="FLOAT")
        model = MockModel(quant_config=quant_config, with_int_weights=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            ShardedStateLoader310.generate_quant_description(model, tmpdir)

            json_path = Path(tmpdir) / "parameters_type_map.json"
            self.assertTrue(json_path.exists())

            with open(json_path, encoding="utf-8") as f:
                quant_description = json.load(f)

            self.assertEqual(quant_description["model_quant_type"], "FLOAT")
            self.assertEqual(quant_description["version"], "1.0.0")
            self.assertIn("linear.weight", quant_description)
            self.assertEqual(quant_description["linear.weight"], "FLOAT")
            self.assertIn("linear.bias", quant_description)
            self.assertEqual(quant_description["linear.bias"], "FLOAT")

    @patch("vllm.model_executor.model_loader.ShardedStateLoader._filter_subtensors")
    def test_generate_quant_description_int_model_310(self, mock_filter):
        """Test generate_quant_description for int8 quantized model."""
        mock_filter.side_effect = lambda x: x
        quant_config = MockQuantConfig(quant_type="W8A8")
        model = MockModel(quant_config=quant_config, with_int_weights=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            ShardedStateLoader310.generate_quant_description(model, tmpdir)

            json_path = Path(tmpdir) / "parameters_type_map.json"
            self.assertTrue(json_path.exists())

            with open(json_path, encoding="utf-8") as f:
                quant_description = json.load(f)

            self.assertEqual(quant_description["model_quant_type"], "W8A8")
            self.assertEqual(quant_description["version"], "1.0.0")
            self.assertIn("linear.weight", quant_description)
            self.assertEqual(quant_description["linear.weight"], "W8A8")
            self.assertIn("linear.bias", quant_description)
            self.assertEqual(quant_description["linear.bias"], "W8A8")
