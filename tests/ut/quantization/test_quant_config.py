from unittest.mock import MagicMock, patch

from vllm.attention.layer import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.quant_config import AscendQuantConfig
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD


class TestAscendQuantConfig(TestBase):

    def setUp(self):
        self.sample_config = {
            "weight": "INT8",
            "fa_quant_type": "C8",
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
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendQuantConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertEqual(result, "ascend")

        # Test when NPU is not available
        mock_is_available.return_value = False
        result = AscendQuantConfig.override_quantization_method(None, None)
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendQuantConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertIsNone(result)

    def test_get_quant_method_for_linear(self):
        mock_config = MagicMock()
        mock_config.model_config.hf_text_config.model_type = None
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with patch("vllm_ascend.quantization.quant_config.get_current_vllm_config", return_value=mock_config), \
            patch.object(self.ascend_config, \
                          'is_layer_skipped_ascend',
                          return_value=True):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.quant_config.get_current_vllm_config", return_value=mock_config), \
            patch('vllm_ascend.quantization.quant_config.AscendLinearMethod', return_value=MagicMock()) as mock_ascend_linear:

            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(
                self.ascend_config, ".attn",
                self.ascend_config.packed_modules_mapping, linear_layer)

    def test_get_quant_method_for_attention(self):
        attention_layer = MagicMock(spec=Attention)
        mock_config = MagicMock()
        mock_config.model_config.hf_text_config.model_type = None
        with patch("vllm_ascend.quantization.quant_config.get_current_vllm_config", return_value=mock_config), \
            patch('vllm_ascend.quantization.quant_config.AscendKVCacheMethod', \
                   return_value=MagicMock()) as mock_ascend_kvcache:
            # Test with fa_quant_type
            method = self.ascend_config.get_quant_method(
                attention_layer, ".attn")
            self.assertIs(method, mock_ascend_kvcache.return_value)

    def test_get_quant_method_for_fused_moe(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_layer.moe = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = MagicMock(spec=FusedMoEConfig)
        mock_config = MagicMock()
        mock_config.model_config.hf_text_config.model_type = None

        # Test skipped layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=True), \
            patch("vllm_ascend.quantization.quant_config.get_current_vllm_config", return_value=mock_config), \
            patch('vllm_ascend.quantization.quant_config.AscendUnquantizedFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

        # Test quantized layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.quant_config.get_current_vllm_config", return_value=mock_config), \
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
