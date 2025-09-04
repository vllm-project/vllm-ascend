from unittest.mock import MagicMock, patch

import torch
from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)

from tests.ut.base import TestBase
from vllm_ascend.quantization.quant_config import (AscendKVCacheMethod,
                                                   AscendQuantConfig)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD


class TestAscendQuantConfig(TestBase):

    def setUp(self):
        self.sample_config = {
            "weight": "INT8",
            "fa_quant_type": "C8",
            "kv_quant_type": "C8",
            "layer1.weight": "INT8",
            "layer2.weight": "FLOAT",
            "fused_layer.weight": "FLOAT",
            "fused_layer.shard1.weight": "FLOAT",
            "fused_layer.shard2.weight": "FLOAT",
            "shard1.weight": "FLOAT",
            "shard2.weight": "FLOAT",
        }
        self.ascend_config = AscendQuantConfig(self.sample_config)
        self.ascend_config.packed_modules_mapping = None

    def test_init(self):
        self.assertEqual(self.ascend_config.quant_description,
                         self.sample_config)

    def test_repr(self):
        repr_str = repr(self.ascend_config)
        self.assertTrue(repr_str.startswith("AscendQuantConfig:\n"))

    def test_get_name(self):
        self.assertEqual(AscendQuantConfig.get_name(),
                         ASCEND_QUANTIZATION_METHOD)

    def test_get_supported_act_dtypes(self):
        supported_dtypes = AscendQuantConfig.get_supported_act_dtypes()
        self.assertEqual(len(supported_dtypes), 3)

    def test_get_min_capability(self):
        with self.assertRaises(NotImplementedError):
            AscendQuantConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AscendQuantConfig.get_config_filenames()
        self.assertEqual(filenames, ["quant_model_description.json"])

    def test_from_config(self):
        config = AscendQuantConfig.from_config(self.sample_config)
        self.assertIsInstance(config, AscendQuantConfig)
        self.assertEqual(config.quant_description, self.sample_config)

    @patch('torch.npu.is_available')
    def test_override_quantization_method(self, mock_is_available):
        # Test when NPU is available
        mock_is_available.return_value = True
        result = AscendQuantConfig.override_quantization_method(None, None)
        self.assertEqual(result, ASCEND_QUANTIZATION_METHOD)

        # Test when NPU is not available
        mock_is_available.return_value = False
        result = AscendQuantConfig.override_quantization_method(None, None)
        self.assertIsNone(result)

    def test_get_quant_method_for_linear(self):
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with patch.object(self.ascend_config,
                          'is_layer_skipped_ascend',
                          return_value=True):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, UnquantizedLinearMethod)

        # Test quantized layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch('vllm_ascend.quantization.quant_config.AscendLinearMethod', return_value=MagicMock()) as mock_ascend_linear:

            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(
                self.ascend_config, ".attn",
                self.ascend_config.packed_modules_mapping)

    def test_get_quant_method_for_attention(self):
        attention_layer = MagicMock(spec=Attention)
        with patch('vllm_ascend.quantization.quant_config.AscendKVCacheMethod',
                   return_value=MagicMock()) as mock_ascend_kvcache:
            # Test with fa_quant_type
            method = self.ascend_config.get_quant_method(
                attention_layer, ".attn")
            self.assertIs(method, mock_ascend_kvcache.return_value)

        with patch('vllm_ascend.quantization.quant_config.AscendKVCacheMethod',
                   return_value=MagicMock()) as mock_ascend_kvcache:
            # Test with kv_quant_type
            modified_config = {"kv_quant_type": "C8"}
            config = AscendQuantConfig(modified_config)
            config.packed_modules_mapping = None
            method = config.get_quant_method(attention_layer, "attn")
            self.assertIs(method, mock_ascend_kvcache.return_value)

    def test_get_quant_method_for_fused_moe(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_layer.moe = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = MagicMock(spec=FusedMoEConfig)

        # Test skipped layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=True), \
            patch('vllm_ascend.quantization.quant_config.AscendUnquantizedFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

        # Test quantized layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch('vllm_ascend.quantization.quant_config.AscendFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

    def test_is_layer_skipped_ascend(self):
        # Test non-fused layer that should be quantized
        self.assertFalse(self.ascend_config.is_layer_skipped_ascend("layer1"))

        # Test non-fused layer that should be skipped
        self.assertTrue(self.ascend_config.is_layer_skipped_ascend("layer2"))

        # Test fused layer
        fused_mapping = {"fused_layer": ["shard1", "shard2"]}
        self.assertTrue(
            self.ascend_config.is_layer_skipped_ascend("fused_layer",
                                                       fused_mapping))

        # Test inconsistent fused layer shards
        bad_config = {"shard1.weight": "FLOAT", "shard2.weight": "INT8"}
        config = AscendQuantConfig(bad_config)
        with self.assertRaises(ValueError):
            config.is_layer_skipped_ascend("fused_layer", fused_mapping)

    def test_get_scaled_act_names(self):
        self.assertEqual(self.ascend_config.get_scaled_act_names(), [])


class TestAscendKVCacheMethod(TestBase):

    def setUp(self):
        # Setup common test fixtures
        self.mock_quant_config = MagicMock(spec=AscendQuantConfig)
        self.mock_quant_config.quant_description = {"kv_quant_type": "C8"}
        self.prefix = "layer.attn"

        # Mock quant_method
        self.mock_quant_method = MagicMock()
        self.patcher = patch(
            'vllm_ascend.quantization.quant_config.get_quant_method')
        self.mock_get_quant_method = self.patcher.start()
        self.mock_get_quant_method.return_value = self.mock_quant_method

        # Create instance
        self.kv_cache_method = AscendKVCacheMethod(self.mock_quant_config,
                                                   self.prefix)

    def tearDown(self):
        self.patcher.stop()

    def test_create_weights(self):
        """Test create_weights delegates to quant_method."""
        mock_layer = MagicMock()
        self.kv_cache_method.create_weights(mock_layer)
        self.mock_quant_method.create_weights.assert_called_once_with(
            mock_layer)

    def test_process_weights_after_loading_with_method(self):
        """Test process_weights when quant_method has the method."""
        mock_layer = MagicMock()
        self.kv_cache_method.process_weights_after_loading(mock_layer)
        self.mock_quant_method.process_weights_after_loading.assert_called_once_with(
            mock_layer)

    def test_process_weights_after_loading_without_method(self):
        """Test process_weights when quant_method lacks the method."""
        # Reset mock to remove the method
        del self.mock_quant_method.process_weights_after_loading
        mock_layer = MagicMock()

        # Should not raise exception
        self.kv_cache_method.process_weights_after_loading(mock_layer)

    def test_apply_delegation(self):
        """Test apply properly delegates to quant_method."""
        mock_layer = MagicMock()
        mock_query = torch.randn(1, 32, 128)
        mock_key = torch.randn(1, 32, 128)
        mock_value = torch.randn(1, 32, 128)
        mock_kv_cache = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_scale = 1.0
        mock_output = torch.zeros(1, 32, 128)
        mock_attn_type = MagicMock()
        expected_result = torch.randn(1, 32, 128)
        self.mock_quant_method.apply.return_value = expected_result

        result = self.kv_cache_method.apply(mock_layer, mock_query, mock_key,
                                            mock_value, mock_kv_cache,
                                            mock_attn_metadata, mock_attn_type,
                                            mock_scale, mock_output)

        self.mock_quant_method.apply.assert_called_once_with(
            mock_layer, mock_query, mock_key, mock_value, mock_kv_cache,
            mock_attn_metadata, mock_attn_type, mock_scale, mock_output)
        self.assertTrue(torch.equal(result, expected_result))
