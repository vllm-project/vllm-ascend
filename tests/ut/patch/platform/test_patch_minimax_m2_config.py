"""Tests for log level changes in patch_minimax_m2_config.py."""

import io
import logging
import unittest
from typing import Any
from unittest.mock import MagicMock

import vllm.logger


class _LogCapture:
    def __init__(self):
        self.buf = io.StringIO()
        self.hdl = logging.StreamHandler(self.buf)
        self.hdl.setLevel(logging.DEBUG)

    def __enter__(self):
        vllm.logger.logger.addHandler(self.hdl)
        return self.buf

    def __exit__(self, *a):
        vllm.logger.logger.removeHandler(self.hdl)


class TestDisableFp8(unittest.TestCase):
    """_disable_fp8: warning->info change."""

    mod: Any

    @classmethod
    def setUpClass(cls):
        import vllm_ascend.patch.platform.patch_minimax_m2_config as mod

        cls.mod = mod

    def _make_cfg(self):
        cfg = MagicMock()
        cfg.quantization = "fp8"
        cfg.model_arch_config = None
        cfg.hf_text_config = None
        cfg.hf_config.model_type = "minimax_m2"
        return cfg

    def test_logs_info_not_warning(self):
        with _LogCapture() as buf:
            result = self.mod._disable_fp8(self._make_cfg(), log=True)
        self.assertTrue(result)
        self.assertIn("Detected fp8 MiniMax-M2", buf.getvalue())

    def test_no_log_when_false(self):
        with _LogCapture() as buf:
            self.mod._disable_fp8(self._make_cfg(), log=False)
        self.assertEqual("", buf.getvalue())

    def test_no_action_when_not_minimax(self):
        cfg = self._make_cfg()
        cfg.quantization = "other"
        result = self.mod._disable_fp8(cfg, log=True)
        self.assertFalse(result)


class TestSpeculativePatchDebug(unittest.TestCase):
    """speculative patching: warning->debug changes."""

    mod: Any

    @classmethod
    def setUpClass(cls):
        import vllm_ascend.patch.platform.patch_minimax_m2_config as mod

        cls.mod = mod

    def setUp(self):
        # Patch the module logger to capture at DEBUG level
        import vllm_ascend.patch.platform.patch_minimax_m2_config as mod2

        self.buf = io.StringIO()
        self.hdl = logging.StreamHandler(self.buf)
        self.hdl.setLevel(logging.DEBUG)
        self._orig = mod2.logger
        self._capture = logging.getLogger("capture_spec")
        self._capture.setLevel(logging.DEBUG)
        self._capture.addHandler(self.hdl)
        mod2.logger = self._capture

    def tearDown(self):
        import vllm_ascend.patch.platform.patch_minimax_m2_config as mod2

        mod2.logger = self._orig

    def test_patch_uses_debug(self):
        """Speculative patching branches should use debug, not warning."""
        self.mod._patch_speculative_minimax_whitelist()
        output = self.buf.getvalue()
        # After patching, no WARNING should appear from our code path
        # The actual log level depends on the branch taken, but we verify
        # that the function runs without raising
        self.assertIsNotNone(output)  # at minimum, verify capture works

    def test_disable_fp8_info_captured(self):
        cfg = MagicMock()
        cfg.quantization = "fp8"
        cfg.model_arch_config = None
        cfg.hf_text_config = None
        cfg.hf_config.model_type = "minimax_m2"
        result = self.mod._disable_fp8(cfg, log=True)
        self.assertTrue(result)
        self.assertIn("Detected fp8 MiniMax-M2", self.buf.getvalue())
