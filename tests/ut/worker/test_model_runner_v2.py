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
        runner.dynamic_eplb = False
        runner.eplb_enable = False
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

    def test_load_model_enables_eplb_when_expert_map_is_configured(self):
        runner = self._make_runner()
        runner.eplb_enable = True
        runner.parallel_config = MagicMock(enable_eplb=False)
        runner.vllm_config.parallel_config = runner.parallel_config

        with patch.object(GPUModelRunner, "load_model") as mock_super_load_model:
            runner.load_model()

        self.assertTrue(runner.parallel_config.enable_eplb)
        mock_super_load_model.assert_called_once_with(False)

    def test_dynamic_eplb_execute_model_runs_forward_before(self):
        runner = self._make_runner()
        runner.dynamic_eplb = True
        runner.eplb_updator = MagicMock()
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)

        with patch.object(GPUModelRunner, "execute_model", return_value="ok") as mock_super_execute:
            output = runner.execute_model(scheduler_output)

        self.assertEqual(output, "ok")
        runner.eplb_updator.forward_before.assert_called_once_with()
        mock_super_execute.assert_called_once()

    def test_dynamic_eplb_sample_tokens_runs_forward_end(self):
        runner = self._make_runner()
        runner.dynamic_eplb = True
        runner.eplb_updator = MagicMock()
        runner.execute_model_state = object()

        with patch.object(GPUModelRunner, "sample_tokens", return_value="ok") as mock_super_sample:
            output = runner.sample_tokens(None)

        self.assertEqual(output, "ok")
        runner.eplb_updator.forward_end.assert_called_once_with()
        mock_super_sample.assert_called_once_with(None)

    def test_dynamic_eplb_dummy_run_profile_clears_moe_load(self):
        runner = self._make_runner()
        runner.dynamic_eplb = True
        runner.is_eplb_warmuped = True
        runner.eplb_warmup = MagicMock()
        runner.eplb_updator = MagicMock()
        # runner.model = MagicMock(clear_all_moe_loads=MagicMock())
        runner.model = SimpleNamespace(clear_all_moe_loads=MagicMock())

        with patch.object(GPUModelRunner, "_dummy_run", return_value=(None, None)) as mock_super_dummy:
            runner._dummy_run(8, is_profile=True)

        runner.eplb_updator.forward_before.assert_not_called()
        runner.model.clear_all_moe_loads.assert_called_once_with()
        runner.eplb_updator.forward_end.assert_called_once_with()
        mock_super_dummy.assert_called_once_with(8, is_profile=True)
