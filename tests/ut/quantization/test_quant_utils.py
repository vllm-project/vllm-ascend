import json
import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase
from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME
from vllm_ascend.quantization.utils import (
    detect_quantization_method,
    maybe_auto_detect_quantization,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD


class TestDetectQuantizationMethod(TestBase):

    def test_returns_none_for_non_directory(self):
        result = detect_quantization_method("/non/existent/path")
        self.assertIsNone(result)

    def test_detects_modelslim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(config_path, "w") as f:
                json.dump({"layer.weight": "INT8"}, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, ASCEND_QUANTIZATION_METHOD)

    def test_detects_compressed_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "quantization_config": {
                        "quant_method": "compressed-tensors"
                    }
                }, f)

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
                json.dump({
                    "quantization_config": {
                        "quant_method": "gptq"
                    }
                }, f)

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
            # Create ModelSlim config
            modelslim_path = os.path.join(tmpdir, MODELSLIM_CONFIG_FILENAME)
            with open(modelslim_path, "w") as f:
                json.dump({"layer.weight": "INT8"}, f)

            # Create compressed-tensors config
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "quantization_config": {
                        "quant_method": "compressed-tensors"
                    }
                }, f)

            result = detect_quantization_method(tmpdir)
            self.assertEqual(result, ASCEND_QUANTIZATION_METHOD)


class TestMaybeAutoDetectQuantization(TestBase):

    def _make_vllm_config(self, model_path="/fake/model", quantization=None):
        vllm_config = MagicMock()
        vllm_config.model_config.model = model_path
        vllm_config.model_config.quantization = quantization
        return vllm_config

    @patch("vllm_ascend.quantization.utils.detect_quantization_method",
           return_value=None)
    def test_no_detection_does_nothing(self, mock_detect):
        vllm_config = self._make_vllm_config()
        maybe_auto_detect_quantization(vllm_config)
        # quantization should remain unchanged
        self.assertIsNone(vllm_config.model_config.quantization)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method",
           return_value=ASCEND_QUANTIZATION_METHOD)
    def test_user_specified_same_method_no_change(self, mock_detect):
        vllm_config = self._make_vllm_config(
            quantization=ASCEND_QUANTIZATION_METHOD)
        maybe_auto_detect_quantization(vllm_config)
        self.assertEqual(vllm_config.model_config.quantization,
                         ASCEND_QUANTIZATION_METHOD)

    @patch("vllm.config.VllmConfig._get_quantization_config",
           return_value=MagicMock())
    @patch("vllm_ascend.quantization.utils.detect_quantization_method",
           return_value=ASCEND_QUANTIZATION_METHOD)
    def test_auto_detect_sets_quantization_and_logs_info(
            self, mock_detect, mock_get_quant_config):
        """When no --quantization is specified but ModelSlim config is found,
        the method should auto-set quantization and emit an INFO log."""
        vllm_config = self._make_vllm_config(
            model_path="/fake/quant_model", quantization=None)

        with self.assertLogs("vllm_ascend.quantization.utils",
                             level=logging.INFO) as cm:
            maybe_auto_detect_quantization(vllm_config)

        self.assertEqual(vllm_config.model_config.quantization,
                         ASCEND_QUANTIZATION_METHOD)
        log_output = "\n".join(cm.output)
        self.assertIn("Auto-detected quantization method", log_output)
        self.assertIn(ASCEND_QUANTIZATION_METHOD, log_output)
        self.assertIn("/fake/quant_model", log_output)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method",
           return_value=ASCEND_QUANTIZATION_METHOD)
    def test_user_mismatch_logs_warning(self, mock_detect):
        """When user specifies a different method than auto-detected,
        a WARNING should be emitted and user's choice should be respected."""
        vllm_config = self._make_vllm_config(
            model_path="/fake/quant_model",
            quantization=COMPRESSED_TENSORS_METHOD)

        with self.assertLogs("vllm_ascend.quantization.utils",
                             level=logging.WARNING) as cm:
            maybe_auto_detect_quantization(vllm_config)

        # User's choice is respected
        self.assertEqual(vllm_config.model_config.quantization,
                         COMPRESSED_TENSORS_METHOD)
        log_output = "\n".join(cm.output)
        self.assertIn("Auto-detected quantization method", log_output)
        self.assertIn(ASCEND_QUANTIZATION_METHOD, log_output)
        self.assertIn(COMPRESSED_TENSORS_METHOD, log_output)

    @patch("vllm_ascend.quantization.utils.detect_quantization_method",
           return_value=None)
    def test_no_detection_emits_no_log(self, mock_detect):
        """When no quantization is detected, no log should be emitted."""
        vllm_config = self._make_vllm_config(quantization=None)
        logger_name = "vllm_ascend.quantization.utils"

        with self.assertRaises(AssertionError):
            # assertLogs raises AssertionError when no logs are emitted
            with self.assertLogs(logger_name, level=logging.DEBUG):
                maybe_auto_detect_quantization(vllm_config)

        self.assertIsNone(vllm_config.model_config.quantization)
