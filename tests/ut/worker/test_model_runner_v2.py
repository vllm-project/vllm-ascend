import unittest
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
