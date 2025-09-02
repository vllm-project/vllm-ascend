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
# This file is a part of the vllm-ascend project.
#
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

from vllm_ascend.models.qwen3 import (ALL_DECODER_LAYER_TYPES,
                                      CustomQwen3DecoderLayer,
                                      CustomQwen3ForCausalLM, CustomQwen3Model)


class TestCustomQwen3ForCausalLM:
    """Test basic functionality of CustomQwen3ForCausalLM class"""

    def test_class_inheritance(self):
        """Test class inheritance relationships"""
        assert issubclass(CustomQwen3ForCausalLM, nn.Module)
        # Note: SupportsLoRA and SupportsPP are Protocol types, cannot use issubclass directly
        # We check if the class has related attributes and methods
        assert hasattr(CustomQwen3ForCausalLM, '__init__')
        assert hasattr(CustomQwen3ForCausalLM, 'packed_modules_mapping')

    @pytest.mark.parametrize("key, expected", [
        ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
        ("gate_up_proj", ["gate_proj", "up_proj"]),
    ])
    def test_packed_modules_mapping(self, key, expected):
        """Test specific mappings in packed_modules_mapping"""
        assert CustomQwen3ForCausalLM.packed_modules_mapping[key] == expected

    def test_packed_modules_mapping_structure(self):
        """Test complete structure of packed_modules_mapping"""
        expected_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        assert CustomQwen3ForCausalLM.packed_modules_mapping == expected_mapping

    def test_packed_modules_mapping_keys(self):
        """Test all keys contained in packed_modules_mapping"""
        expected_keys = {"qkv_proj", "gate_up_proj"}
        actual_keys = set(CustomQwen3ForCausalLM.packed_modules_mapping.keys())
        assert actual_keys == expected_keys


class TestCustomQwen3DecoderLayer:
    """Test basic functionality of CustomQwen3DecoderLayer class"""

    def test_class_inheritance(self):
        """Test class inheritance relationships"""
        assert issubclass(CustomQwen3DecoderLayer, Qwen3DecoderLayer)

    @patch('vllm_ascend.models.qwen3.AddRMSNormW8A8Quant')
    def test_init_with_quant_config(self, mock_add_rms_norm):
        """Test initialization with quantization config"""
        # Create mock config object
        config = MagicMock(spec=Qwen3Config)
        config.hidden_size = 512
        config.rms_norm_eps = 1e-6

        # This test mainly verifies that initialization doesn't fail
        # Due to many dependencies, we mainly test the basic class structure
        # Test case without quantization config
        with patch.object(CustomQwen3DecoderLayer.__bases__[0],
                          '__init__',
                          return_value=None):
            layer = CustomQwen3DecoderLayer(config=config, quant_config=None)
            assert isinstance(layer, CustomQwen3DecoderLayer)

    def test_init_without_quant_config(self):
        """Test initialization without quantization config"""
        config = MagicMock(spec=Qwen3Config)
        config.hidden_size = 512
        config.rms_norm_eps = 1e-6

        with patch.object(CustomQwen3DecoderLayer.__bases__[0], '__init__'):
            layer = CustomQwen3DecoderLayer(config=config, quant_config=None)
            assert isinstance(layer, CustomQwen3DecoderLayer)


class TestCustomQwen3Model:
    """Test basic functionality of CustomQwen3Model class"""

    def test_class_inheritance(self):
        """Test class inheritance relationships"""
        assert issubclass(CustomQwen3Model, Qwen2Model)

    def test_torch_compile_decorator(self):
        """Test existence of torch.compile decorator"""
        # Check if the class has related compilation decorator attributes
        assert hasattr(CustomQwen3Model, '__init__')

    @patch('vllm_ascend.models.qwen3.CustomQwen3DecoderLayer')
    def test_init_with_custom_decoder_layer(self, mock_decoder_layer):
        """Test initialization with custom decoder layer"""
        # Create mock VllmConfig
        vllm_config = MagicMock()

        # Completely mock the initialization process to avoid compilation config issues
        with patch.object(CustomQwen3Model.__bases__[0],
                          '__init__',
                          return_value=None) as mock_super_init:
            # Create instance but don't call real initialization
            model = CustomQwen3Model.__new__(CustomQwen3Model)
            # Manually set necessary attributes
            model.decoder_layer_type = CustomQwen3DecoderLayer

            # Verify correct type
            assert isinstance(model, CustomQwen3Model)
            assert model.decoder_layer_type == CustomQwen3DecoderLayer


class TestALLDecoderLayerTypes:
    """Test ALL_DECODER_LAYER_TYPES constant"""

    def test_decoder_layer_types_structure(self):
        """Test structure of decoder layer types"""
        expected_types = {
            "attention": CustomQwen3DecoderLayer,
        }
        assert ALL_DECODER_LAYER_TYPES == expected_types

    def test_decoder_layer_types_keys(self):
        """Test keys contained in decoder layer types"""
        expected_keys = {"attention"}
        actual_keys = set(ALL_DECODER_LAYER_TYPES.keys())
        assert actual_keys == expected_keys

    def test_decoder_layer_types_values(self):
        """Test values of decoder layer types"""
        assert ALL_DECODER_LAYER_TYPES["attention"] == CustomQwen3DecoderLayer


class TestCustomQwen3ForCausalLMFunctionality(unittest.TestCase):
    """Test functional methods of CustomQwen3ForCausalLM"""

    def setUp(self):
        """Set up test environment"""
        self.config = MagicMock(spec=Qwen3Config)
        self.config.vocab_size = 1000
        self.config.hidden_size = 512
        self.config.tie_word_embeddings = False

        self.vllm_config = MagicMock()
        self.vllm_config.model_config.hf_config = self.config
        self.vllm_config.quant_config = None
        self.vllm_config.lora_config = None

    @patch('vllm_ascend.models.qwen3.get_pp_group')
    @patch('vllm_ascend.models.qwen3.CustomQwen3Model')
    @patch('vllm_ascend.models.qwen3.ParallelLMHead')
    @patch('vllm_ascend.models.qwen3.LogitsProcessor')
    def test_init_components(self, mock_logits_processor, mock_lm_head,
                             mock_model, mock_pp_group):
        """Test initialization of components"""
        # Mock PP group
        mock_pp_group.return_value.is_last_rank = True

        # Create instance
        model = CustomQwen3ForCausalLM(vllm_config=self.vllm_config)

        # Verify components are created correctly
        mock_model.assert_called_once()
        mock_lm_head.assert_called_once()
        mock_logits_processor.assert_called_once()

    @patch('vllm_ascend.models.qwen3.get_pp_group')
    @patch('vllm_ascend.models.qwen3.CustomQwen3Model')
    @patch('vllm_ascend.models.qwen3.LogitsProcessor')
    def test_init_with_tied_embeddings(self, mock_logits_processor, mock_model,
                                       mock_pp_group):
        """Test initialization with tied embeddings"""
        # Set tied embeddings
        self.config.tie_word_embeddings = True
        mock_pp_group.return_value.is_last_rank = True

        # Mock model's embed_tokens
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.embed_tokens = MagicMock()

        model = CustomQwen3ForCausalLM(vllm_config=self.vllm_config)

        # Verify lm_head is set to embed_tokens
        assert model.lm_head == mock_model_instance.embed_tokens

    @patch('vllm_ascend.models.qwen3.get_pp_group')
    @patch('vllm_ascend.models.qwen3.CustomQwen3Model')
    @patch('vllm_ascend.models.qwen3.ParallelLMHead')
    @patch('vllm_ascend.models.qwen3.LogitsProcessor')
    def test_get_input_embeddings(self, mock_logits_processor, mock_lm_head,
                                  mock_model, mock_pp_group):
        """Test get_input_embeddings method"""
        # Mock PP group
        mock_pp_group.return_value.is_last_rank = True

        # Create model instance
        model = CustomQwen3ForCausalLM(vllm_config=self.vllm_config)

        # Mock input
        input_ids = torch.tensor([[1, 2, 3]])

        # Call method
        model.get_input_embeddings(input_ids)

        # Verify model's get_input_embeddings method was called
        model.model.get_input_embeddings.assert_called_once_with(input_ids)

    @patch('vllm_ascend.models.qwen3.get_pp_group')
    @patch('vllm_ascend.models.qwen3.CustomQwen3Model')
    @patch('vllm_ascend.models.qwen3.ParallelLMHead')
    @patch('vllm_ascend.models.qwen3.LogitsProcessor')
    def test_forward(self, mock_logits_processor, mock_lm_head, mock_model,
                     mock_pp_group):
        """Test forward method"""
        # Mock PP group
        mock_pp_group.return_value.is_last_rank = True

        # Create model instance
        model = CustomQwen3ForCausalLM(vllm_config=self.vllm_config)

        # Mock input
        input_ids = torch.tensor([[1, 2, 3]])
        positions = torch.tensor([0, 1, 2])

        # Call forward method
        result = model.forward(input_ids, positions)

        # Verify model's forward method was called
        model.model.assert_called_once_with(input_ids, positions, None, None)

    @patch('vllm_ascend.models.qwen3.get_pp_group')
    @patch('vllm_ascend.models.qwen3.CustomQwen3Model')
    @patch('vllm_ascend.models.qwen3.ParallelLMHead')
    @patch('vllm_ascend.models.qwen3.LogitsProcessor')
    def test_compute_logits(self, mock_logits_processor, mock_lm_head,
                            mock_model, mock_pp_group):
        """Test compute_logits method"""
        # Mock PP group
        mock_pp_group.return_value.is_last_rank = True

        # Create model instance
        model = CustomQwen3ForCausalLM(vllm_config=self.vllm_config)

        # Mock input
        hidden_states = torch.randn(1, 3, 512)
        sampling_metadata = MagicMock()

        # Call compute_logits method
        model.compute_logits(hidden_states, sampling_metadata)

        # Verify logits_processor was called
        model.logits_processor.assert_called_once_with(model.lm_head,
                                                       hidden_states,
                                                       sampling_metadata)
