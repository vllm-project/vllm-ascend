from unittest.mock import MagicMock, patch

import numpy as np
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from tests.ut.base import TestBase


class TestNPUModelRunner(TestBase):
    def _make_runner(self, prefill_use_eager: bool):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = object.__new__(NPUModelRunner)
        runner._pad_for_sequence_parallelism = lambda num_tokens: num_tokens
        runner.input_batch = MagicMock()
        runner.input_batch.num_computed_tokens_cpu = np.array([0, 0], dtype=np.int32)
        runner.input_batch.lora_id_to_lora_request = {}
        runner.speculative_config = None
        runner.uniform_decode_query_len = 1
        runner.model_config = MagicMock()
        runner.model_config.is_encoder_decoder = False
        runner.parallel_config = MagicMock()
        runner.parallel_config.data_parallel_rank = 0
        runner.vllm_config = MagicMock()
        runner.vllm_config.parallel_config = MagicMock()
        runner.vllm_config.parallel_config.data_parallel_size = 1
        runner.vllm_config.parallel_config.tensor_parallel_size = 1
        runner.vllm_config.observability_config = MagicMock()
        runner.vllm_config.observability_config.cudagraph_metrics = False
        runner.ascend_config = MagicMock()
        runner.ascend_config.ascend_compilation_config.prefill_use_eager = prefill_use_eager
        runner.cudagraph_dispatcher = MagicMock()
        runner.cudagraph_dispatcher.dispatch.return_value = (
            CUDAGraphMode.FULL,
            BatchDescriptor(num_tokens=2, num_reqs=2, uniform=True),
        )
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.enable_sp", return_value=False)
    def test_determine_batch_execution_with_prefill_aclgraph(self, _mock_enable_sp):
        runner = self._make_runner(prefill_use_eager=False)

        cudagraph_mode, batch_descriptor, _, _, _ = runner._determine_batch_execution_and_padding(
            num_tokens=2,
            num_reqs=2,
            num_scheduled_tokens_np=np.array([1, 1], dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )

        self.assertEqual(cudagraph_mode, CUDAGraphMode.FULL)
        self.assertEqual(batch_descriptor, BatchDescriptor(num_tokens=2, num_reqs=2, uniform=True))
        runner.cudagraph_dispatcher.dispatch.assert_called_once()

    @patch("vllm_ascend.worker.model_runner_v1.enable_sp", return_value=False)
    def test_determine_batch_execution_forces_eager_prefill(self, _mock_enable_sp):
        runner = self._make_runner(prefill_use_eager=True)

        cudagraph_mode, batch_descriptor, _, _, _ = runner._determine_batch_execution_and_padding(
            num_tokens=2,
            num_reqs=2,
            num_scheduled_tokens_np=np.array([1, 1], dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )

        self.assertEqual(cudagraph_mode, CUDAGraphMode.NONE)
        self.assertEqual(batch_descriptor, BatchDescriptor(num_tokens=2))
        runner.cudagraph_dispatcher.dispatch.assert_not_called()
