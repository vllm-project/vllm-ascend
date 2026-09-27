import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerNaNDetection(unittest.TestCase):
    """Exercise NaN bookkeeping with CPU tensors only."""

    def _build_runner(self, num_reqs: int = 3, max_model_len: int = 64):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_async_scheduling = False
        runner.routed_experts_initialized = False
        runner.num_discarded_requests = 0
        runner.discard_request_indices = SimpleNamespace(np=np.array([], dtype=np.int64))
        runner.max_model_len = max_model_len
        runner.num_spec_tokens = 0
        runner.requests = {f"req-{i}": SimpleNamespace(output_token_ids=[]) for i in range(num_reqs)}
        runner.input_batch = SimpleNamespace(
            req_ids=[f"req-{i}" for i in range(num_reqs)],
            req_id_to_index={f"req-{i}": i for i in range(num_reqs)},
            generators={},
            num_tokens_no_spec=[0] * num_reqs,
            num_tokens=[0] * num_reqs,
            token_ids_cpu=np.zeros((num_reqs, max_model_len), dtype=np.int64),
            is_token_ids=np.zeros((num_reqs, max_model_len), dtype=np.bool_),
        )
        runner._to_list = lambda sampled_token_ids: sampled_token_ids.tolist()
        runner._get_prompt_logprobs_dict = lambda *args, **kwargs: {}
        return runner

    @staticmethod
    def _make_inputs(num_reqs: int):
        sampler_output = SimpleNamespace(
            sampled_token_ids=torch.arange(1, num_reqs + 1, dtype=torch.int64).view(num_reqs, 1),
            logprobs_tensors=None,
        )
        scheduler_output = SimpleNamespace(num_scheduled_tokens=num_reqs)
        hidden_states = torch.zeros(num_reqs, 4)
        return scheduler_output, sampler_output, hidden_states

    @patch("vllm_ascend.worker.model_runner_v1.envs.VLLM_COMPUTE_NANS_IN_LOGITS", True)
    def test_bookkeeping_sync_returns_per_request_nan_counts_first(self):
        runner = self._build_runner(num_reqs=3)
        logits = torch.tensor(
            [
                [float("nan"), 1.0, float("nan")],
                [1.0, 2.0, 3.0],
                [float("nan"), float("nan"), float("nan")],
            ]
        )
        scheduler_output, sampler_output, hidden_states = self._make_inputs(num_reqs=3)

        result = runner._bookkeeping_sync(scheduler_output, sampler_output, logits, hidden_states, 3, None)

        self.assertEqual(len(result), 8)
        self.assertEqual(result[0], {"req-0": 2, "req-1": 0, "req-2": 3})
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertEqual(result[3], [[1], [2], [3]])
        self.assertEqual(result[5], ["req-0", "req-1", "req-2"])

    @patch("vllm_ascend.worker.model_runner_v1.envs.VLLM_COMPUTE_NANS_IN_LOGITS", False)
    def test_bookkeeping_sync_skips_nan_computation_when_disabled(self):
        runner = self._build_runner(num_reqs=2)
        runner._get_nans_in_logits = MagicMock()
        scheduler_output, sampler_output, hidden_states = self._make_inputs(num_reqs=2)

        result = runner._bookkeeping_sync(scheduler_output, sampler_output, torch.randn(2, 8), hidden_states, 2, None)

        self.assertEqual(result[0], {})
        self.assertIsNone(result[1])
        runner._get_nans_in_logits.assert_not_called()

    @patch(
        "vllm_ascend.worker.model_runner_v1.envs.VLLM_COMPUTE_NANS_IN_LOGITS",
        True,
    )
    def test_bookkeeping_sync_keeps_nan_counts_on_device_when_async(self):
        runner = self._build_runner(num_reqs=3)
        runner.use_async_scheduling = True
        runner._get_nans_in_logits = MagicMock()
        logits = torch.tensor(
            [
                [float("nan"), 1.0, float("nan")],
                [1.0, 2.0, 3.0],
                [float("nan"), float("nan"), float("nan")],
            ]
        )
        scheduler_output, sampler_output, hidden_states = self._make_inputs(num_reqs=3)

        result = runner._bookkeeping_sync(scheduler_output, sampler_output, logits, hidden_states, 3, None)

        self.assertEqual(result[0], {})
        self.assertEqual(result[1].dtype, torch.int32)
        torch.testing.assert_close(result[1], torch.tensor([2, 0, 3], dtype=result[1].dtype))
        runner._get_nans_in_logits.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.envs.VLLM_COMPUTE_NANS_IN_LOGITS", True)
    def test_bookkeeping_sync_reports_zero_nans_for_none_logits(self):
        runner = self._build_runner(num_reqs=2)
        scheduler_output, sampler_output, hidden_states = self._make_inputs(num_reqs=2)

        result = runner._bookkeeping_sync(scheduler_output, sampler_output, None, hidden_states, 2, None)

        self.assertEqual(result[0], {"req-0": 0, "req-1": 0})

    @patch("vllm_ascend.worker.model_runner_v1.envs.VLLM_COMPUTE_NANS_IN_LOGITS", True)
    def test_bookkeeping_sync_defaults_to_zero_when_req_has_no_logits_row(self):
        runner = self._build_runner(num_reqs=3)
        logits = torch.tensor(
            [
                [float("nan"), 1.0],
                [1.0, 2.0],
            ]
        )
        scheduler_output, sampler_output, hidden_states = self._make_inputs(num_reqs=3)

        result = runner._bookkeeping_sync(scheduler_output, sampler_output, logits, hidden_states, 3, None)

        self.assertEqual(result[0], {"req-0": 1, "req-1": 0, "req-2": 0})


if __name__ == "__main__":
    unittest.main()
