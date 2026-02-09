import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.modelslim_config import (
    MODELSLIM_CONFIG_FILENAME,
    AscendModelSlimConfig,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, vllm_version_is

if vllm_version_is("v0.15.0"):
    from vllm.attention.layer import Attention  # type: ignore
else:
    from vllm.model_executor.layers.attention import Attention


class TestAscendModelSlimConfig(TestBase):

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
        self.ascend_config = AscendModelSlimConfig(self.sample_config)
        self.ascend_config.packed_modules_mapping = None

    def test_init(self):
        self.assertEqual(self.ascend_config.quant_description,
                         self.sample_config)

    def test_repr(self):
        repr_str = repr(self.ascend_config)
        self.assertTrue(repr_str.startswith("AscendModelSlimConfig:\n"))

    def test_get_name(self):
        self.assertEqual(AscendModelSlimConfig.get_name(),
                         ASCEND_QUANTIZATION_METHOD)

    def test_get_supported_act_dtypes(self):
        supported_dtypes = AscendModelSlimConfig.get_supported_act_dtypes()
        self.assertEqual(len(supported_dtypes), 3)

    def test_get_min_capability(self):
        with self.assertRaises(NotImplementedError):
            AscendModelSlimConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AscendModelSlimConfig.get_config_filenames()
        self.assertEqual(filenames, [])

    def test_from_config(self):
        config = AscendModelSlimConfig.from_config(self.sample_config)
        self.assertIsInstance(config, AscendModelSlimConfig)
        self.assertEqual(config.quant_description, self.sample_config)

    @patch('torch.npu.is_available')
    def test_override_quantization_method(self, mock_is_available):
        # Test when NPU is available
        mock_is_available.return_value = True
        result = AscendModelSlimConfig.override_quantization_method(None, None)
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendModelSlimConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertEqual(result, "ascend")

        # Test when NPU is not available
        mock_is_available.return_value = False
        result = AscendModelSlimConfig.override_quantization_method(None, None)
        self.assertIsNone(result)
        hf_quant_cfg = {"quant_method": ""}
        result = AscendModelSlimConfig.override_quantization_method(
            hf_quant_cfg, None)
        self.assertIsNone(result)

    def test_get_quant_method_for_linear(self):
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        linear_layer = MagicMock(spec=LinearBase)
        # Test skipped layer
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch.object(self.ascend_config, \
                          'is_layer_skipped_ascend',
                          return_value=True):
            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod)

        # Test quantized layer
        mock_scheme = MagicMock()
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendLinearMethod', return_value=MagicMock()) as mock_ascend_linear:

            method = self.ascend_config.get_quant_method(linear_layer, ".attn")
            self.assertIs(method, mock_ascend_linear.return_value)
            mock_ascend_linear.assert_called_once_with(mock_scheme)

    def test_get_quant_method_for_attention(self):
        attention_layer = MagicMock(spec=Attention)
        mock_config = MagicMock()
        mock_config.model_config.hf_config.model_type = None
        mock_scheme = MagicMock()
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendKVCacheMethod', \
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
        mock_config.model_config.hf_config.model_type = None

        # Test skipped layer
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=True), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch('vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
            method = self.ascend_config.get_quant_method(
                fused_moe_layer, "moe_layer")
            self.assertIs(method, mock_ascend_moe.return_value)

        # Test quantized layer
        mock_scheme = MagicMock()
        with patch.object(self.ascend_config, 'is_layer_skipped_ascend', return_value=False), \
            patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_config), \
            patch("vllm_ascend.quantization.modelslim_config.create_scheme_for_layer", return_value=mock_scheme), \
            patch('vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod', return_value=MagicMock()) as mock_ascend_moe:
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
        config = AscendModelSlimConfig(bad_config)
        with self.assertRaises(ValueError):
            config.is_layer_skipped_ascend("fused_layer", fused_mapping)

    def test_init_no_args(self):
        """Test that AscendModelSlimConfig can be instantiated without args."""
        config = AscendModelSlimConfig()
        self.assertEqual(config.quant_description, {})

    def test_maybe_update_config_skips_when_populated(self):
        """Test that maybe_update_config is a no-op when config is already populated."""
        config = AscendModelSlimConfig({"layer1.weight": "INT8"})
        # Should not raise even with invalid path since config is already populated
        config.maybe_update_config("/nonexistent/path")
        self.assertEqual(config.quant_description["layer1.weight"], "INT8")

    def test_maybe_update_config_loads_config_file(self):
        """Test that maybe_update_config loads the config from model directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_config = {"layer1.weight": "W8A8_DYNAMIC", "layer2.weight": "FLOAT"}
            config_path = os.path.join(temp_dir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump(quant_config, f)

            config = AscendModelSlimConfig()
            config.maybe_update_config(temp_dir)

            self.assertEqual(config.quant_description["layer1.weight"], "W8A8_DYNAMIC")
            self.assertEqual(config.quant_description["layer2.weight"], "FLOAT")

    def test_maybe_update_config_friendly_error_for_missing_config(self):
        """Test that maybe_update_config raises a friendly error when config is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AscendModelSlimConfig()

            with self.assertRaises(ValueError) as context:
                config.maybe_update_config(temp_dir)

            error_msg = str(context.exception)
            # Check that the error message is user-friendly
            self.assertIn("ModelSlim Quantization Config Not Found", error_msg)
            self.assertIn(temp_dir, error_msg)
            self.assertIn(ASCEND_QUANTIZATION_METHOD, error_msg)
            self.assertIn("Solution 1:", error_msg)
            self.assertIn("Solution 2:", error_msg)
            self.assertIn("ModelSlim", error_msg)

    def test_maybe_update_config_shows_existing_json_files(self):
        """Test that the error message lists existing JSON files for diagnosis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some unrelated JSON files
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump({}, f)

            config = AscendModelSlimConfig()

            with self.assertRaises(ValueError) as context:
                config.maybe_update_config(temp_dir)

            error_msg = str(context.exception)
            self.assertIn("config.json", error_msg)

    def test_maybe_update_config_applies_shared_head_adaptation(self):
        """Test that maybe_update_config applies shared_head key adaptations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_config = {"layer.shared_head.weight": "INT8"}
            config_path = os.path.join(temp_dir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump(quant_config, f)

            config = AscendModelSlimConfig()
            config.maybe_update_config(temp_dir)

            # Original key should still exist
            self.assertIn("layer.shared_head.weight", config.quant_description)
            # Adapted key should also exist
            self.assertIn("layer.weight", config.quant_description)

    def test_get_scaled_act_names(self):
        self.assertEqual(self.ascend_config.get_scaled_act_names(), [])
