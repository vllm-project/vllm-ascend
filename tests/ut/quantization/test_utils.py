import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.config import KVTransferConfig

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import FAKQUANT_CONFIG, W8A8_CONFIG
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.quantization import AscendCompressedTensorsConfig
from vllm_ascend.quantization.configs.modelslim_config import MODELSLIM_CONFIG_FILENAME, AscendModelSlimConfig
from vllm_ascend.quantization.utils import (
    detect_quantization_method,
    enable_fa_quant,
    get_dynamic_mx_quant_scale_alg,
    maybe_auto_detect_quantization,
    model_uses_fa_quantization,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD


class TestDynamicMxQuantScaleAlg(TestBase):
    @staticmethod
    def _config(architecture, model_type=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=[architecture] if architecture else [],
                hf_text_config=SimpleNamespace(model_type=model_type),
            )
        )

    @patch("vllm_ascend.quantization.utils.get_current_hardware_profile")
    def test_uses_one_only_for_minimax_m3_on_a5(self, mock_profile):
        minimax_config = self._config("MiniMaxM3SparseForCausalLM")
        other_config = self._config("DeepseekV3ForCausalLM")

        mock_profile.return_value = get_hardware_profile(AscendDeviceType.A5)
        self.assertEqual(get_dynamic_mx_quant_scale_alg(minimax_config), 1)
        self.assertEqual(get_dynamic_mx_quant_scale_alg(other_config), 0)

        mock_profile.return_value = get_hardware_profile(AscendDeviceType.A3)
        self.assertEqual(get_dynamic_mx_quant_scale_alg(minimax_config), 0)

    @patch(
        "vllm_ascend.quantization.utils.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    @patch("vllm.config.get_current_vllm_config")
    def test_uses_current_vllm_config_when_config_is_omitted(self, mock_current_config, _mock_profile):
        minimax_config = self._config(None, model_type="minimax_m3")
        mock_current_config.return_value = minimax_config

        self.assertEqual(get_dynamic_mx_quant_scale_alg(), 1)

    @patch("vllm.forward_context.get_forward_context")
    @patch("vllm.forward_context.is_forward_context_available", return_value=True)
    @patch(
        "vllm_ascend.quantization.utils.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    def test_uses_forward_vllm_config_when_available(
        self,
        _mock_profile,
        _mock_forward_context_available,
        mock_forward_context,
    ):
        mock_forward_context.return_value.additional_kwargs = {"dynamic_mx_quant_scale_alg": 1}

        self.assertEqual(get_dynamic_mx_quant_scale_alg(), 1)

    @patch("vllm.config.get_current_vllm_config")
    @patch("vllm.forward_context.get_forward_context")
    @patch("vllm.forward_context.is_forward_context_available", return_value=True)
    @patch(
        "vllm_ascend.quantization.utils.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    def test_falls_back_to_current_config_when_forward_scale_is_missing(
        self,
        _mock_profile,
        _mock_forward_context_available,
        mock_forward_context,
        mock_current_config,
    ):
        minimax_config = self._config(None, model_type="minimax_m3")
        mock_forward_context.return_value.additional_kwargs = {}
        mock_current_config.return_value = minimax_config

        self.assertEqual(get_dynamic_mx_quant_scale_alg(), 1)


class TestDetectQuantizationMethod(TestBase):
    def test_returns_none_for_non_existent_path(self):
        result = detect_quantization_method("/non/existent/path")
        self.assertIsNone(result)

    def test_detects_modelslim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump({"layer.weight": "INT8"}, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, ASCEND_QUANTIZATION_METHOD)

    def test_detects_fa_quantization_from_modelslim_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump({"fa_quant_type": "FAKQuant"}, f)

            self.assertTrue(model_uses_fa_quantization(tmpdir))

    def test_fa_quantization_detection_is_model_agnostic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump({"model.layers.0.weight": "W8A8_DYNAMIC"}, f)

            self.assertFalse(model_uses_fa_quantization(tmpdir))

    def test_detects_compressed_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "compressed-tensors"}}, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, COMPRESSED_TENSORS_METHOD)

    def test_returns_none_for_no_quant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_returns_none_for_non_compressed_tensors_quant_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "gptq"}}, f)

            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_returns_none_for_config_without_quant_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "llama"}, f)

            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_returns_none_for_malformed_config_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                f.write("not valid json{{{")

            result = detect_quantization_method(tmpdir)
            self.assertIsNone(result)

    def test_modelslim_takes_priority_over_compressed_tensors(self):
        """When both ModelSlim config and compressed-tensors config exist,
        ModelSlim should take priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modelslim_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(modelslim_path, "w") as f:
                json.dump({"layer.weight": "INT8"}, f)

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"quantization_config": {"quant_method": "compressed-tensors"}}, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, ASCEND_QUANTIZATION_METHOD)


class TestMaybeAutoDetectQuantization(TestBase):
    def _make_vllm_config(self, model_path="/fake/model", quantization=None, revision=None):
        vllm_config = MagicMock()
        vllm_config.model_config.model = model_path
        vllm_config.model_config.quantization = quantization
        vllm_config.model_config.revision = revision
        return vllm_config

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=None)
    def test_no_detection_does_nothing(self, mock_detect):
        vllm_config = self._make_vllm_config()
        maybe_auto_detect_quantization(vllm_config)
        self.assertIsNone(vllm_config.model_config.quantization)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_user_specified_same_method_no_change(self, mock_detect):
        vllm_config = self._make_vllm_config(quantization=ASCEND_QUANTIZATION_METHOD)
        maybe_auto_detect_quantization(vllm_config)
        self.assertEqual(vllm_config.model_config.quantization, ASCEND_QUANTIZATION_METHOD)

    @patch("vllm.config.VllmConfig._get_quantization_config", return_value=MagicMock())
    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_auto_detect_sets_quantization_and_logs_info(self, mock_detect, mock_get_quant_config):
        """When no --quantization is specified but ModelSlim config is found,
        the method should auto-set quantization and emit an INFO log."""
        vllm_config = self._make_vllm_config(model_path="/fake/quant_model", quantization=None)

        with patch("vllm_ascend.quantization.utils.logger") as mock_logger:
            maybe_auto_detect_quantization(vllm_config)

        self.assertEqual(vllm_config.model_config.quantization, ASCEND_QUANTIZATION_METHOD)
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        self.assertIn("Auto-detected quantization method", call_args[0])
        self.assertIn(ASCEND_QUANTIZATION_METHOD, call_args)
        self.assertIn("/fake/quant_model", call_args)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_user_mismatch_logs_warning(self, mock_detect):
        """When user specifies a different method than auto-detected,
        a WARNING should be emitted and user's choice should be respected."""
        vllm_config = self._make_vllm_config(model_path="/fake/quant_model", quantization=COMPRESSED_TENSORS_METHOD)

        with patch("vllm_ascend.quantization.utils.logger") as mock_logger:
            maybe_auto_detect_quantization(vllm_config)

        self.assertEqual(vllm_config.model_config.quantization, COMPRESSED_TENSORS_METHOD)
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        self.assertIn("Auto-detected quantization method", call_args[0])
        self.assertIn(ASCEND_QUANTIZATION_METHOD, call_args)
        self.assertIn(COMPRESSED_TENSORS_METHOD, call_args)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=None)
    def test_no_detection_emits_info_log(self, mock_detect):
        """When no quantization is detected, an info log tells the user the model loads as float."""
        vllm_config = self._make_vllm_config(quantization=None)

        with patch("vllm_ascend.quantization.utils.logger") as mock_logger:
            maybe_auto_detect_quantization(vllm_config)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        self.assertIn("No quantization signature detected", call_args[0])
        self.assertIn("/fake/model", call_args)
        mock_logger.warning.assert_not_called()
        self.assertIsNone(vllm_config.model_config.quantization)

    @patch("vllm.config.VllmConfig._get_quantization_config", return_value=MagicMock())
    @patch("vllm_ascend.quantization.utils.detect_quantization_method", return_value=ASCEND_QUANTIZATION_METHOD)
    def test_passes_revision_to_detect(self, mock_detect, mock_get_quant):
        """Verify that model revision is forwarded to detect_quantization_method."""
        vllm_config = self._make_vllm_config(model_path="org/model-name", revision="v1.0", quantization=None)
        maybe_auto_detect_quantization(vllm_config)
        mock_detect.assert_called_once_with("org/model-name", revision="v1.0")


class TestEnableFaQuant(TestBase):
    def test_non_quantization_scenarios(self):
        # non quantization scene
        vllm_config = MagicMock()
        vllm_config.quant_config = None
        result = enable_fa_quant(vllm_config)
        self.assertFalse(result)

        # CompressedTensors scene
        vllm_config.quant_config = AscendCompressedTensorsConfig({}, [], "", {})
        result = enable_fa_quant(vllm_config)
        self.assertFalse(result)

        # non fa3 quant scene
        vllm_config.quant_config = AscendModelSlimConfig(W8A8_CONFIG)
        result = enable_fa_quant(vllm_config)
        self.assertFalse(result)

    def test_fa3_quantization_scenario(self):
        vllm_config = MagicMock()
        vllm_config.quant_config = AscendModelSlimConfig(FAKQUANT_CONFIG)
        vllm_config.kv_transfer_config = KVTransferConfig(kv_connector="MultiConnector", kv_role="kv_consumer")
        result = enable_fa_quant(vllm_config)
        self.assertTrue(result)
        result = enable_fa_quant(vllm_config, layer_name="test_layer")
        self.assertFalse(result)
