# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Tests for _sync_metadata_across_dp DP peer disconnection handling.

from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CUDAGraphMode


class TestSyncMetadataAcrossDp:
    """Tests for _sync_metadata_across_dp try/except behavior."""

    def _make_runner(self, dp_size=4, dp_rank=0, skip_allreduce=False):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = MagicMock(spec=NPUModelRunner)
        runner.dp_size = dp_size
        runner.dp_rank = dp_rank
        runner._dp_peer_disconnected = False
        runner.ascend_config = MagicMock()
        runner.ascend_config.dp_allreduce_on_npu = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.should_skip_allreduce_across_dp_group", return_value=True)
    def test_skip_allreduce_branch_unaffected(self, mock_skip):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = self._make_runner()
        result = NPUModelRunner._sync_metadata_across_dp(runner, num_tokens=128, cudagraph_mode=CUDAGraphMode.NONE)
        assert result is not None

    def test_dp_size_1_skips_all_reduce(self):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = self._make_runner(dp_size=1)
        result = NPUModelRunner._sync_metadata_across_dp(runner, num_tokens=128, cudagraph_mode=CUDAGraphMode.NONE)
        assert result is not None

    @patch("vllm_ascend.worker.model_runner_v1.should_skip_allreduce_across_dp_group", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_dp_group")
    @patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce")
    def test_peer_disconnect_returns_local_metadata(self, mock_all_reduce, mock_dp_group, mock_skip):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        mock_all_reduce.side_effect = RuntimeError(
            "[/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [127.0.0.1]:50332"
        )
        runner = self._make_runner()
        NPUModelRunner._sync_metadata_across_dp(runner, num_tokens=128, cudagraph_mode=CUDAGraphMode.NONE)
        assert runner._dp_peer_disconnected is True

    @patch("vllm_ascend.worker.model_runner_v1.should_skip_allreduce_across_dp_group", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_dp_group")
    @patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce")
    def test_hccl_peer_disconnect_returns_local_metadata(self, mock_all_reduce, mock_dp_group, mock_skip):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        mock_all_reduce.side_effect = RuntimeError("HCCL error: EI9999 notify wait timeout")
        runner = self._make_runner()
        NPUModelRunner._sync_metadata_across_dp(runner, num_tokens=128, cudagraph_mode=CUDAGraphMode.NONE)
        assert runner._dp_peer_disconnected is True

    @patch("vllm_ascend.worker.model_runner_v1.should_skip_allreduce_across_dp_group", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_dp_group")
    @patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce")
    def test_unrelated_runtime_error_reraises(self, mock_all_reduce, mock_dp_group, mock_skip):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        mock_all_reduce.side_effect = RuntimeError("NPU out of memory. Tried to allocate 2.00 GiB.")
        runner = self._make_runner()
        with pytest.raises(RuntimeError, match="NPU out of memory"):
            NPUModelRunner._sync_metadata_across_dp(runner, num_tokens=128, cudagraph_mode=CUDAGraphMode.NONE)
