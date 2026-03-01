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
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


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
            self.linear.weight = torch.nn.Parameter(torch.randint(-127, 127, (10, 10), dtype=torch.int8))
            self.linear.bias = torch.nn.Parameter(torch.zeros(10, dtype=torch.int32))
        else:
            self.linear = torch.nn.Linear(10, 10)


class TestShardedStateLoader310(TestBase):
    """Test cases for ShardedStateLoader310."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.load_config = MagicMock()

    def test_init_310(self):
        """Test initialization of ShardedStateLoader310."""
        loader = ShardedStateLoader310(self.load_config)
        self.assertIsNotNone(loader)
        self.assertIsInstance(loader, ShardedStateLoader310)

    @patch("vllm_ascend._310p.sharded_state_loader_310p.get_tensor_model_parallel_rank")
    @patch("vllm_ascend._310p.sharded_state_loader_310p.save_file")
    @patch("torch_npu.get_npu_format")
    @patch("torch_npu.npu_format_cast")
    def test_save_model_with_nz_format_310(self, mock_cast, mock_get_format, mock_save_file, mock_get_rank):
        """Test save_model with NZ format tensors that need conversion."""
        mock_get_rank.return_value = 0
        mock_get_format.return_value = ACL_FORMAT_FRACTAL_NZ

        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_cast.return_value = mock_tensor

        model = MockModel()
        with (
            patch.object(model, "state_dict", return_value={"linear.weight": mock_tensor}),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            ShardedStateLoader310.save_model(model, tmpdir)

            mock_cast.assert_called_once_with(mock_tensor, ACL_FORMAT_FRACTAL_ND)
            mock_save_file.assert_called_once()
            call_args = mock_save_file.call_args[0]
            self.assertTrue(call_args[1].endswith("model-00000-of-00001.safetensors"))

    @patch("vllm_ascend._310p.sharded_state_loader_310p.get_tensor_model_parallel_rank")
    @patch("vllm_ascend._310p.sharded_state_loader_310p.save_file")
    @patch("torch_npu.get_npu_format")
    @patch("torch_npu.npu_format_cast")
    def test_save_model_with_nd_format_310(self, mock_cast, mock_get_format, mock_save_file, mock_get_rank):
        """Test save_model with ND format tensors (no conversion needed)."""
        mock_get_rank.return_value = 0
        mock_get_format.return_value = ACL_FORMAT_FRACTAL_ND

        mock_tensor = MagicMock(spec=torch.Tensor)

        model = MockModel()
        with (
            patch.object(model, "state_dict", return_value={"linear.weight": mock_tensor}),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            ShardedStateLoader310.save_model(model, tmpdir)

            mock_cast.assert_not_called()
            mock_save_file.assert_called_once()

    def test_generate_quant_description_float_model_310(self):
        """Test generate_quant_description for float model."""
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

    def test_generate_quant_description_int_model_310(self):
        """Test generate_quant_description for int8 quantized model."""
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

    def test_generate_quant_description_mixed_precision_310(self):
        """Test generate_quant_description with mixed precision tensors."""
        quant_config = MockQuantConfig(quant_type="W8A8")

        class MixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_config = quant_config
                self.int_weight = torch.nn.Parameter(torch.randint(-127, 127, (10, 10), dtype=torch.int8))
                self.float_weight = torch.nn.Parameter(torch.randn(10, 10))

        model = MixedModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            ShardedStateLoader310.generate_quant_description(model, tmpdir)

            json_path = Path(tmpdir) / "parameters_type_map.json"
            self.assertTrue(json_path.exists())

            with open(json_path, encoding="utf-8") as f:
                quant_description = json.load(f)

            self.assertEqual(quant_description["model_quant_type"], "W8A8")
            self.assertEqual(quant_description["int_weight"], "W8A8")
            self.assertEqual(quant_description["float_weight"], "FLOAT")
            self.assertEqual(quant_description["float_buffer"], "FLOAT")
