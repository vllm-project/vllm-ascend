# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Tests for execute_dummy_batch DP peer disconnection handling.

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.worker.worker import NPUWorker


class TestExecuteDummyBatchDpDisconnect:
    """Tests for execute_dummy_batch DP peer disconnection handling."""

    def test_execute_dummy_batch_dp_peer_disconnected_gloo(self):
        """Test execute_dummy_batch handles gloo DP peer disconnection gracefully"""
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.model_runner.decode_token_per_req = 1
            worker.model_runner._dp_peer_disconnected = False
            worker.model_runner._dummy_run.side_effect = RuntimeError(
                "[/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] Connection closed by peer [127.0.0.1]:50332"
            )

            worker.execute_dummy_batch()
            worker.model_runner._dummy_run.assert_called_once()

    def test_execute_dummy_batch_dp_peer_disconnected_hccl(self):
        """Test execute_dummy_batch handles HCCL DP peer disconnection gracefully"""
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.model_runner.decode_token_per_req = 1
            worker.model_runner._dp_peer_disconnected = False
            worker.model_runner._dummy_run.side_effect = RuntimeError("HCCL error: EI9999 notify wait timeout")

            worker.execute_dummy_batch()
            worker.model_runner._dummy_run.assert_called_once()

    def test_execute_dummy_batch_unrelated_error_reraises(self):
        """Test execute_dummy_batch re-raises unrelated RuntimeError (NPU OOM)"""
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.model_runner.decode_token_per_req = 1
            worker.model_runner._dp_peer_disconnected = False
            worker.model_runner._dummy_run.side_effect = RuntimeError("NPU out of memory. Tried to allocate 2.00 GiB.")

            with pytest.raises(RuntimeError, match="NPU out of memory"):
                worker.execute_dummy_batch()

    def test_execute_dummy_batch_skipped_when_flag_set(self):
        """Test execute_dummy_batch skips when _dp_peer_disconnected is True"""
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.model_runner.decode_token_per_req = 1
            worker.model_runner._dp_peer_disconnected = True

            worker.execute_dummy_batch()
            worker.model_runner._dummy_run.assert_not_called()
