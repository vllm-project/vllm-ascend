from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.config import CUDAGraphMode
from vllm.sequence import IntermediateTensors

from tests.ut.base import TestBase


class TestNPUWorkerV2(TestBase):
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.enable_sp", return_value=False)
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    def test_execute_model_middle_rank_pp(
        self, mock_get_tp_group, mock_get_pp_group, mock_enable_sp, mock_get_ascend_config
    ):
        """MRV2 PP middle ranks send intermediate tensors and return None."""
        from vllm_ascend.worker.worker import NPUWorker

        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = False
        mock_get_ascend_config.return_value = mock_ascend_config

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker.use_v2_model_runner = True
            worker.profiler = None
            worker._pp_send_work = []

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = False
            mock_pp_group.is_last_rank = False
            mock_pp_group.irecv_tensor_dict.return_value = ({"tensor": "data"}, None, None)
            mock_pp_group.isend_tensor_dict.return_value = []
            mock_get_pp_group.return_value = mock_pp_group

            intermediate_output = IntermediateTensors({"output_tensor": "data"})
            worker.model_runner.execute_model.return_value = intermediate_output

            scheduler_output = MagicMock()
            scheduler_output.total_num_scheduled_tokens = 1

            result = worker.execute_model(scheduler_output)

            mock_pp_group.irecv_tensor_dict.assert_called_once_with(all_gather_group=mock_get_tp_group.return_value)
            worker.model_runner.execute_model.assert_called_once()
            mock_pp_group.isend_tensor_dict.assert_called_once_with(
                intermediate_output.tensors,
                all_gather_group=mock_get_tp_group.return_value,
            )
            self.assertIsNone(result)

    @patch("vllm_ascend.worker.worker.torch.npu.synchronize")
    @patch("vllm_ascend.worker.worker.get_pp_group")
    def test_profile_prefill_latency_v2_restores_runner_state(
        self,
        mock_get_pp_group,
        mock_synchronize,
    ):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.use_v2_model_runner = True
            worker.scheduler_config = SimpleNamespace(max_num_batched_tokens=1024)

            model_runner = MagicMock()
            model_runner.max_num_reqs = 8
            worker.model_runner = model_runner

            def check_temporary_state(**kwargs):
                self.assertEqual(model_runner.max_num_reqs, 1)

            model_runner._dummy_run.side_effect = check_temporary_state
            mock_get_pp_group.return_value.is_first_rank = True

            with patch("time.perf_counter", side_effect=[10.0, 10.25]):
                latency_ms = worker.profile_prefill_latency(512)

            self.assertEqual(latency_ms, 250.0)
            self.assertEqual(model_runner.max_num_reqs, 8)
            model_runner._dummy_run.assert_called_once_with(
                num_tokens=512,
                force_attention=True,
                is_profile=False,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            self.assertEqual(mock_synchronize.call_count, 2)

    @patch("vllm_ascend.worker.worker.torch.npu.synchronize")
    @patch("vllm_ascend.worker.worker.get_pp_group")
    def test_profile_prefill_latency_v2_restores_state_on_error(
        self,
        mock_get_pp_group,
        mock_synchronize,
    ):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.use_v2_model_runner = True
            worker.scheduler_config = SimpleNamespace(max_num_batched_tokens=1024)

            model_runner = MagicMock()
            model_runner.max_num_reqs = 8
            model_runner._dummy_run.side_effect = RuntimeError("dummy failure")
            worker.model_runner = model_runner

            mock_get_pp_group.return_value.is_first_rank = True

            with self.assertRaisesRegex(RuntimeError, "dummy failure"):
                worker.profile_prefill_latency(512)

            self.assertEqual(model_runner.max_num_reqs, 8)
            self.assertEqual(mock_synchronize.call_count, 1)

    def test_sample_tokens_v2_attaches_execution_time(self):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.use_v2_model_runner = True

            profiling_config = SimpleNamespace(need_timing=True)
            model_runner = MagicMock()
            model_runner.ascend_config = SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    profiling_chunk_config=profiling_config,
                )
            )
            model_runner._cpp_execution_time_ms = 125.0
            model_runner_output = SimpleNamespace()
            output = SimpleNamespace(model_runner_output=model_runner_output)
            model_runner.sample_tokens.return_value = output
            worker.model_runner = model_runner

            grammar_output = SimpleNamespace()
            result = worker.sample_tokens(grammar_output)

            self.assertIs(result, output)
            self.assertEqual(model_runner_output.execution_time_ms, 125.0)
            self.assertIsNone(model_runner._cpp_execution_time_ms)
            model_runner.sample_tokens.assert_called_once_with(grammar_output)

    @patch(
        "vllm_ascend.worker.worker._attach_profiling_chunk_execution_time",
    )
    def test_sample_tokens_v1_keeps_original_path(self, mock_attach):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.use_v2_model_runner = False

            model_runner = MagicMock()
            output = SimpleNamespace()
            model_runner.sample_tokens.return_value = output
            worker.model_runner = model_runner

            grammar_output = SimpleNamespace()
            result = worker.sample_tokens(grammar_output)

            self.assertIs(result, output)
            mock_attach.assert_not_called()
            model_runner.sample_tokens.assert_called_once_with(grammar_output)
