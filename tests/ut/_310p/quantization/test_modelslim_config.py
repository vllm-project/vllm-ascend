from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend._310p.quantization.modelslim_config import AscendModelSlimConfig310


class TestAscendModelSlimConfig310(TestBase):

    def setUp(self):
        self.sample_config = {
            "weight": "INT8",
            "layer1.weight": "INT8",
            "layer2.weight": "FLOAT",
            "fused_layer.weight": "FLOAT",
            "fused_layer.shard1.weight": "FLOAT",
            "fused_layer.shard2.weight": "FLOAT",
            "shard1.weight": "FLOAT",
            "shard2.weight": "FLOAT",
        }
        self.ascend_config = AscendModelSlimConfig310(self.sample_config)
        self.ascend_config.packed_modules_mapping = None

    def test_get_quant_method_for_linear_310(self):
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch.object(self.ascend_config, \
                          'is_layer_skipped_ascend',
                          return_value=True):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        mock_scheme = MagicMock()
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend._310p.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend._310p.quantization.modelslim_config.AscendLinearMethod', return_value=MagicMock()) as mock_ascend_linear:

            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(mock_scheme)

    def test_get_quant_method_for_fused_moe_310(self):
        fused_moe_layer = MagicMock(spec=FusedMoE)
        fused_moe_layer.moe = MagicMock(spec=FusedMoEConfig)
        fused_moe_layer.moe_config = MagicMock(spec=FusedMoEConfig)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        mock_scheme = MagicMock()
        with self.assertRaises(NotImplementedError):
            with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
                patch("vllm_ascend._310p.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
                patch("vllm_ascend._310p.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
                patch('vllm_ascend._310p.quantization.modelslim_config.AscendLinearMethod', return_value=MagicMock()) as mock_ascend_moe:
                method = self.ascend_config.get_quant_method(
                    fused_moe_layer, "moe_layer")
