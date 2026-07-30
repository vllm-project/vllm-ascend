from unittest.mock import MagicMock, patch

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
