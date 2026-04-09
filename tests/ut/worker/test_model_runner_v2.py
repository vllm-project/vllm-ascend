import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType, get_mrv2_in_profile_run
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class TestNPUModelRunnerV2(unittest.TestCase):
    @staticmethod
    def _make_runner(max_num_tokens: int = 16):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.max_num_tokens = max_num_tokens
        runner.vllm_config = MagicMock()
        return runner

    @staticmethod
    def _make_configs(*, weight_prefetch: bool = False, pcp_size: int = 1, dcp_size: int = 1, dynamic_eplb: bool = False):
        vllm_config = MagicMock()
        vllm_config.parallel_config = SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        )
        ascend_config = SimpleNamespace(
            weight_prefetch_config=SimpleNamespace(enabled=weight_prefetch),
            eplb_config=SimpleNamespace(dynamic_eplb=dynamic_eplb),
        )
        return vllm_config, ascend_config

    def test_validate_supported_v2_features_rejects_weight_prefetch(self):
        vllm_config, ascend_config = self._make_configs(weight_prefetch=True)
        with self.assertRaisesRegex(NotImplementedError, "Weight prefetch"):
            NPUModelRunner.validate_supported_v2_features(vllm_config, ascend_config)

    def test_validate_supported_v2_features_rejects_context_parallel(self):
        vllm_config, ascend_config = self._make_configs(pcp_size=2)
        with self.assertRaisesRegex(NotImplementedError, "Context parallelism"):
            NPUModelRunner.validate_supported_v2_features(vllm_config, ascend_config)

    def test_validate_supported_v2_features_rejects_dynamic_eplb(self):
        vllm_config, ascend_config = self._make_configs(dynamic_eplb=True)
        with self.assertRaisesRegex(NotImplementedError, "dynamic_eplb"):
            NPUModelRunner.validate_supported_v2_features(vllm_config, ascend_config)

    def test_profile_run_marks_only_mc2_warmup_dummy_run(self):
        runner = self._make_runner(max_num_tokens=16)
        observed_runs: list[tuple[int, bool]] = []

        def fake_base_dummy_run(self, num_tokens, *args, **kwargs):
            observed_runs.append((num_tokens, get_mrv2_in_profile_run()))
            return None, None

        def fake_base_profile_run(self):
            self._dummy_run(self.max_num_tokens, skip_attn=True)

        with (
            patch.object(GPUModelRunner, "_dummy_run", new=fake_base_dummy_run),
            patch.object(GPUModelRunner, "profile_run", new=fake_base_profile_run),
            patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=8),
            patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        ):
            runner.profile_run()

        self.assertEqual(observed_runs, [(8, True), (16, True)])
        self.assertFalse(get_mrv2_in_profile_run())

    def test_profile_run_keeps_normal_dummy_run_outside_profile_override(self):
        runner = self._make_runner(max_num_tokens=16)
        observed_runs: list[tuple[int, bool]] = []

        def fake_base_dummy_run(self, num_tokens, *args, **kwargs):
            observed_runs.append((num_tokens, get_mrv2_in_profile_run()))
            return None, None

        def fake_base_profile_run(self):
            self._dummy_run(self.max_num_tokens, skip_attn=True)

        with (
            patch.object(GPUModelRunner, "_dummy_run", new=fake_base_dummy_run),
            patch.object(GPUModelRunner, "profile_run", new=fake_base_profile_run),
            patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=32),
            patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        ):
            runner.profile_run()

        self.assertEqual(observed_runs, [(16, True)])
