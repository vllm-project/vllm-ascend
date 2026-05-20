import unittest
from unittest.mock import patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config


class TestSamplingConfig(TestBase):
    def test_sampling_config_defaults_are_opt_in(self):
        from vllm_ascend.ascend_config import SamplingConfig

        sampling_config = SamplingConfig()

        self.assertFalse(sampling_config.enable_sampling_optimization)
        self.assertFalse(sampling_config.enable_batch_parallel)
        self.assertEqual(sampling_config.logits_processing_mode, "default")

    def test_sampling_config_accepts_supported_logits_processing_modes(self):
        from vllm_ascend.ascend_config import SamplingConfig

        for mode in ("default", "skip", "fused"):
            with self.subTest(mode=mode):
                sampling_config = SamplingConfig(
                    {
                        "enable_sampling_optimization": True,
                        "enable_batch_parallel": True,
                        "logits_processing_mode": mode,
                    }
                )

                self.assertTrue(sampling_config.enable_sampling_optimization)
                self.assertTrue(sampling_config.enable_batch_parallel)
                self.assertEqual(sampling_config.logits_processing_mode, mode)

    def test_sampling_config_advanced_options_do_not_enable_optimization(self):
        from vllm_ascend.ascend_config import SamplingConfig

        sampling_config = SamplingConfig(
            {
                "enable_batch_parallel": True,
                "logits_processing_mode": "skip",
            }
        )

        self.assertFalse(sampling_config.enable_sampling_optimization)
        self.assertTrue(sampling_config.enable_batch_parallel)
        self.assertEqual(sampling_config.logits_processing_mode, "skip")

    def test_sampling_config_rejects_unknown_logits_processing_mode(self):
        from vllm_ascend.ascend_config import SamplingConfig

        with self.assertRaisesRegex(ValueError, "logits_processing_mode.*default.*skip.*fused"):
            SamplingConfig({"logits_processing_mode": "environment_variable"})

    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_ascend_config_parses_sampling_config_from_additional_config(self, _mock_fix_incompatible_config):
        clear_ascend_config()
        try:
            vllm_config = VllmConfig()
            vllm_config.additional_config = {
                "sampling_config": {
                    "enable_sampling_optimization": True,
                    "enable_batch_parallel": True,
                    "logits_processing_mode": "skip",
                }
            }

            ascend_config = init_ascend_config(vllm_config)

            self.assertTrue(ascend_config.sampling_config.enable_sampling_optimization)
            self.assertTrue(ascend_config.sampling_config.enable_batch_parallel)
            self.assertEqual(ascend_config.sampling_config.logits_processing_mode, "skip")
        finally:
            clear_ascend_config()

    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_ascend_config_exposes_default_sampling_config_when_absent(self, _mock_fix_incompatible_config):
        clear_ascend_config()
        try:
            ascend_config = init_ascend_config(VllmConfig())

            self.assertFalse(ascend_config.sampling_config.enable_sampling_optimization)
            self.assertFalse(ascend_config.sampling_config.enable_batch_parallel)
            self.assertEqual(ascend_config.sampling_config.logits_processing_mode, "default")
        finally:
            clear_ascend_config()


if __name__ == "__main__":
    unittest.main()
