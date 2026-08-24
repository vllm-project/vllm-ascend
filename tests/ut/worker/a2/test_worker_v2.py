from unittest.mock import MagicMock, patch

import torch
from vllm.sequence import IntermediateTensors

from tests.ut.base import TestBase
from vllm_ascend.worker.pp_dfx import (
    PP_DFX_METADATA_KEY,
    PPTransferDFX,
    compute_tensor_fingerprints,
    verify_tensor_fingerprints,
)


class TestNPUWorkerV2(TestBase):
    @patch("vllm_ascend.worker.pp_dfx.logger.isEnabledFor", return_value=False)
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.enable_sp", return_value=False)
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    def test_execute_model_middle_rank_pp(
        self,
        mock_get_tp_group,
        mock_get_pp_group,
        mock_enable_sp,
        mock_get_ascend_config,
        mock_debug_disabled,
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
            worker.pp_transfer_dfx = PPTransferDFX(worker.use_v2_model_runner)

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

    def test_pp_dfx_fingerprint_verification(self):
        expected = compute_tensor_fingerprints(
            {
                "hidden_states": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
                "residual": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
            }
        )

        with patch("vllm_ascend.worker.pp_dfx.logger.debug") as mock_debug:
            verify_tensor_fingerprints(
                expected,
                {
                    "hidden_states": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
                    "residual": torch.tensor([[3.0, 5.0]], dtype=torch.float32),
                },
                transfer_seq=3,
                src_rank=0,
                dst_rank=1,
            )

        mock_debug.assert_called_once_with(
            "[pp-dfx] seq=%d link=%d->%d verify=failed tensor=%s",
            3,
            0,
            1,
            "residual",
        )

    @patch("vllm_ascend.worker.pp_dfx.time.perf_counter_ns", side_effect=[1_000_000, 2_500_000])
    @patch("vllm_ascend.worker.pp_dfx.torch.npu.synchronize")
    @patch("vllm_ascend.worker.pp_dfx.logger.isEnabledFor", return_value=True)
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.enable_sp", return_value=False)
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    def test_execute_model_middle_rank_pp_dfx(
        self,
        mock_get_tp_group,
        mock_get_pp_group,
        mock_enable_sp,
        mock_get_ascend_config,
        mock_debug_enabled,
        mock_synchronize,
        mock_perf_counter_ns,
    ):
        from vllm_ascend.worker.worker import NPUWorker

        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = False
        mock_get_ascend_config.return_value = mock_ascend_config

        received_tensors = {"hidden_states": torch.tensor([[1.0, 2.0]])}
        received_fingerprints = compute_tensor_fingerprints(received_tensors)
        recv_tensor_dict = dict(received_tensors)
        recv_tensor_dict[PP_DFX_METADATA_KEY] = {
            "transfer_seq": 0,
            "fingerprints": received_fingerprints,
        }
        send_handle = MagicMock()

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker.use_v2_model_runner = True
            worker.profiler = None
            worker.log_memory_stats = MagicMock()
            worker._pp_send_work = []
            worker.pp_transfer_dfx = PPTransferDFX(worker.use_v2_model_runner)

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = False
            mock_pp_group.is_last_rank = False
            mock_pp_group.rank_in_group = 1
            mock_pp_group.world_size = 3
            mock_pp_group.irecv_tensor_dict.return_value = (recv_tensor_dict, [], [])
            mock_pp_group.isend_tensor_dict.return_value = [send_handle]
            mock_get_pp_group.return_value = mock_pp_group

            intermediate_output = IntermediateTensors({"hidden_states": torch.tensor([[3.0, 4.0]])})
            worker.model_runner.execute_model.return_value = intermediate_output

            scheduler_output = MagicMock()
            scheduler_output.total_num_scheduled_tokens = 1

            result = worker.execute_model(scheduler_output)
            received = worker.model_runner.execute_model.call_args.args[1]
            self.assertNotIn(PP_DFX_METADATA_KEY, received.tensors)
            torch.testing.assert_close(
                received.tensors["hidden_states"],
                received_tensors["hidden_states"],
            )

        sent_tensors = mock_pp_group.isend_tensor_dict.call_args.args[0]
        self.assertIn(PP_DFX_METADATA_KEY, sent_tensors)
        send_handle.wait.assert_called_once_with()
        self.assertEqual(mock_synchronize.call_count, 2)
        self.assertIsNone(result)
