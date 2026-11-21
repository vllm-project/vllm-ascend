# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.worker.worker import NPUWorker


def _worker(*, is_mm_encoder_only: bool) -> NPUWorker:
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = SimpleNamespace(is_mm_encoder_only=is_mm_encoder_only)
    worker.device = "npu:0"
    return worker


def test_make_model_runner_builds_the_npu_runner():
    worker = _worker(is_mm_encoder_only=False)
    with patch("vllm_ascend.worker.v2.model_runner.NPUModelRunner") as runner_cls:
        runner = worker._make_model_runner()

    assert runner is runner_cls.return_value
    runner_cls.assert_called_once_with(worker.vllm_config, "npu:0")


def test_make_model_runner_wraps_the_mm_encoder_runner():
    worker = _worker(is_mm_encoder_only=True)
    wrapper = MagicMock()
    with (
        patch("vllm.v1.worker.mm_encoder_model_runner.MMEncoderModelRunner") as runner_cls,
        patch("vllm_ascend.worker.v2.utils.torch_cuda_wrapper", return_value=wrapper),
    ):
        runner = worker._make_model_runner()

    assert runner is runner_cls.return_value
    runner_cls.assert_called_once_with(worker.vllm_config, "npu:0")
    wrapper.__enter__.assert_called_once_with()
