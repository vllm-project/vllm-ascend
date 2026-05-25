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

    def test_sampling_config_can_enable_optimization(self):
        from vllm_ascend.ascend_config import SamplingConfig

        sampling_config = SamplingConfig({"enable_sampling_optimization": True})

        self.assertTrue(sampling_config.enable_sampling_optimization)

    def test_sampling_config_rejects_later_stage_options(self):
        from vllm_ascend.ascend_config import SamplingConfig

        with self.assertRaisesRegex(ValueError, "only supports"):
            SamplingConfig({"logits_processing_mode": "skip"})

    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_ascend_config_parses_sampling_config_from_additional_config(self, _mock_fix_incompatible_config):
        clear_ascend_config()
        try:
            vllm_config = VllmConfig()
            vllm_config.additional_config = {
                "sampling_config": {
                    "enable_sampling_optimization": True,
                }
            }

            ascend_config = init_ascend_config(vllm_config)

            self.assertTrue(ascend_config.sampling_config.enable_sampling_optimization)
        finally:
            clear_ascend_config()

    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_ascend_config_exposes_default_sampling_config_when_absent(self, _mock_fix_incompatible_config):
        clear_ascend_config()
        try:
            ascend_config = init_ascend_config(VllmConfig())

            self.assertFalse(ascend_config.sampling_config.enable_sampling_optimization)
        finally:
            clear_ascend_config()


if __name__ == "__main__":
    unittest.main()
