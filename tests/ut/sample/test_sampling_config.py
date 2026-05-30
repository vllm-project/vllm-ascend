import unittest
from types import SimpleNamespace

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config


class TestSamplingConfig(TestBase):
    @staticmethod
    def _make_vllm_config(additional_config=None):
        return SimpleNamespace(
            additional_config=additional_config,
            cache_config=SimpleNamespace(block_size=16),
            compilation_config=SimpleNamespace(pass_config=SimpleNamespace(enable_sp=False)),
            kv_transfer_config=None,
            model_config=None,
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                enable_expert_parallel=False,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                tensor_parallel_size=1,
            ),
            quant_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
            speculative_config=None,
        )

    def test_sampling_config_defaults_are_opt_in(self):
        from vllm_ascend.ascend_config import SamplingConfig

        sampling_config = SamplingConfig()

        self.assertFalse(sampling_config.enable_sampling_v2)
        self.assertFalse(sampling_config.enable_reduced_sampling)

    def test_sampling_config_can_enable_sampling_v2(self):
        from vllm_ascend.ascend_config import SamplingConfig

        sampling_config = SamplingConfig({"enable_sampling_v2": True})

        self.assertTrue(sampling_config.enable_sampling_v2)

    def test_sampling_config_can_enable_reduced_sampling(self):
        from vllm_ascend.ascend_config import SamplingConfig

        sampling_config = SamplingConfig({"enable_reduced_sampling": True})

        self.assertTrue(sampling_config.enable_reduced_sampling)

    def test_sampling_config_rejects_later_stage_options(self):
        from vllm_ascend.ascend_config import SamplingConfig

        with self.assertRaisesRegex(ValueError, "only supports"):
            SamplingConfig({"logits_processing_mode": "skip"})

    def test_ascend_config_parses_sampling_config_from_additional_config(self):
        clear_ascend_config()
        try:
            vllm_config = self._make_vllm_config(
                {
                    "sampling_config": {
                        "enable_sampling_v2": True,
                        "enable_reduced_sampling": True,
                    }
                }
            )

            ascend_config = init_ascend_config(vllm_config)

            self.assertTrue(ascend_config.sampling_config.enable_sampling_v2)
            self.assertTrue(ascend_config.sampling_config.enable_reduced_sampling)
        finally:
            clear_ascend_config()

    def test_ascend_config_exposes_default_sampling_config_when_absent(self):
        clear_ascend_config()
        try:
            ascend_config = init_ascend_config(self._make_vllm_config())

            self.assertFalse(ascend_config.sampling_config.enable_sampling_v2)
            self.assertFalse(ascend_config.sampling_config.enable_reduced_sampling)
        finally:
            clear_ascend_config()


if __name__ == "__main__":
    unittest.main()
