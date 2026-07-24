import contextlib
import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestEmbeddingTPWithMM(unittest.TestCase):
    """Test embedding TP support for multimodal models.

    When embedding_tensor_parallel_size is enabled with mm models,
    DP padding must be forced to ensure uniform tensor sizes across
    DP ranks for embedding TP all_gather/reduce_scatter collectives.
    """

    def _build_runner(self, embedding_tp=False, supports_mm=True):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.vllm_config.model_config = MagicMock()
        runner.vllm_config.model_config.is_encoder_decoder = False
        runner.parallel_config = MagicMock()
        runner.parallel_config.data_parallel_rank = 0
        runner.parallel_config.data_parallel_size = 4
        runner.model_config = runner.vllm_config.model_config
        runner.dp_size = 4
        runner.dp_rank = 0
        runner.supports_mm_inputs = supports_mm
        runner.positions = torch.zeros(128, dtype=torch.int64)
        runner.pcp_size = 1
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=True)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.should_skip_allreduce_across_dp_group", return_value=True)
    def test_sync_metadata_does_not_skip_dp_allreduce_with_embedding_tp(self, mock_skip, mock_pp_group, mock_embed_tp):
        """When embedding TP is enabled, DP all_reduce must not be skipped
        even if should_skip_allreduce_across_dp_group returns True."""
        runner = self._build_runner(embedding_tp=True)
        mock_pp_group.return_value.is_first_rank = True

        with (
            patch("vllm_ascend.worker.model_runner_v1.get_dp_group") as mock_dp_group,
            patch("vllm_ascend.worker.model_runner_v1.dist") as mock_dist,
            patch("vllm_ascend.worker.model_runner_v1.CUDAGraphMode"),
        ):
            mock_dp_group.return_value.cpu_group = MagicMock()

            from vllm.config import CUDAGraphMode

            runner._sync_metadata_across_dp(
                num_tokens=10,
                cudagraph_mode=CUDAGraphMode.NONE,
                allow_dp_padding=False,
            )

        # dist.all_reduce should be called (not skipped)
        mock_dist.all_reduce.assert_called_once()

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.should_skip_allreduce_across_dp_group", return_value=True)
    def test_sync_metadata_skips_dp_allreduce_without_embedding_tp(self, mock_skip, mock_pp_group, mock_embed_tp):
        """Without embedding TP, DP all_reduce can be skipped when
        should_skip_allreduce_across_dp_group returns True."""
        runner = self._build_runner(embedding_tp=False)
        mock_pp_group.return_value.is_first_rank = True

        with (
            patch("vllm_ascend.worker.model_runner_v1.get_dp_group") as mock_dp_group,
            patch("vllm_ascend.worker.model_runner_v1.dist") as mock_dist,
            patch("vllm_ascend.worker.model_runner_v1.CUDAGraphMode"),
        ):
            mock_dp_group.return_value.cpu_group = MagicMock()

            from vllm.config import CUDAGraphMode

            runner._sync_metadata_across_dp(
                num_tokens=10,
                cudagraph_mode=CUDAGraphMode.NONE,
                allow_dp_padding=False,
            )

        # dist.all_reduce should NOT be called (skipped)
        mock_dist.all_reduce.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=True)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_preprocess_overrides_num_scheduled_tokens_with_embedding_tp(self, mock_pp_group, mock_embed_tp):
        """When embedding TP + mm model, _preprocess should override
        total_num_scheduled_tokens to num_input_tokens."""
        runner = self._build_runner(embedding_tp=True)
        mock_pp_group.return_value.is_first_rank = True

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 5
        scheduler_output.scheduled_encoder_inputs = {}
        num_input_tokens = 10

        # Capture the value at call time, since the finally block restores it
        captured_scheduled = []

        def capture_and_return(*args, **kwargs):
            captured_scheduled.append(args[1].total_num_scheduled_tokens)
            return (None, None, None, None, None, None)

        with (
            patch.object(GPUModelRunner, "_preprocess", side_effect=capture_and_return, autospec=True),
            contextlib.suppress(Exception),
        ):
            runner._preprocess(scheduler_output, num_input_tokens, None)

        assert captured_scheduled[0] == num_input_tokens, (
            f"Expected {num_input_tokens} at call time, got {captured_scheduled[0]}"
        )

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=True)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_preprocess_restores_num_scheduled_tokens_after_call(self, mock_pp_group, mock_embed_tp):
        """After _preprocess, total_num_scheduled_tokens should be restored
        to its original value."""
        runner = self._build_runner(embedding_tp=True)
        mock_pp_group.return_value.is_first_rank = True

        original_scheduled = 5
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = original_scheduled
        scheduler_output.scheduled_encoder_inputs = {}
        num_input_tokens = 10

        with (
            patch.object(
                GPUModelRunner,
                "_preprocess",
                side_effect=lambda *a, **kw: (None, None, None, None, None, None),
                autospec=True,
            ),
            contextlib.suppress(Exception),
        ):
            runner._preprocess(scheduler_output, num_input_tokens, None)

        assert scheduler_output.total_num_scheduled_tokens == original_scheduled, (
            f"Expected {original_scheduled}, got {scheduler_output.total_num_scheduled_tokens}"
        )

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=True)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_preprocess_zeros_positions_for_padding_tokens(self, mock_pp_group, mock_embed_tp):
        """Positions for padding tokens (between saved_num_scheduled and
        num_input_tokens) should be zeroed out."""
        runner = self._build_runner(embedding_tp=True)
        mock_pp_group.return_value.is_first_rank = True

        original_scheduled = 5
        num_input_tokens = 10
        runner.positions[:num_input_tokens] = torch.arange(num_input_tokens)

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = original_scheduled
        scheduler_output.scheduled_encoder_inputs = {}

        with (
            patch.object(
                GPUModelRunner,
                "_preprocess",
                side_effect=lambda *a, **kw: (None, None, None, None, None, None),
                autospec=True,
            ),
            contextlib.suppress(Exception),
        ):
            runner._preprocess(scheduler_output, num_input_tokens, None)

        # Positions for real tokens should be preserved
        assert torch.equal(runner.positions[:original_scheduled], torch.arange(original_scheduled)), (
            "Positions for real tokens should not be modified"
        )

        # Positions for padding tokens should be zeroed
        assert torch.equal(
            runner.positions[original_scheduled:num_input_tokens],
            torch.zeros(num_input_tokens - original_scheduled, dtype=torch.int64),
        ), "Positions for padding tokens should be zeroed"

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_preprocess_does_not_override_without_embedding_tp(self, mock_pp_group, mock_embed_tp):
        """Without embedding TP, _preprocess should not override
        total_num_scheduled_tokens."""
        runner = self._build_runner(embedding_tp=False)
        mock_pp_group.return_value.is_first_rank = True

        original_scheduled = 5
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = original_scheduled
        scheduler_output.scheduled_encoder_inputs = {}
        num_input_tokens = 10

        with (
            patch.object(
                GPUModelRunner,
                "_preprocess",
                side_effect=lambda *a, **kw: (None, None, None, None, None, None),
                autospec=True,
            ),
            contextlib.suppress(Exception),
        ):
            runner._preprocess(scheduler_output, num_input_tokens, None)

        assert scheduler_output.total_num_scheduled_tokens == original_scheduled, (
            "Should not override without embedding TP"
        )

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=True)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_preprocess_does_not_override_for_non_first_pp_rank(self, mock_pp_group, mock_embed_tp):
        """Embedding TP override should not apply for non-first PP ranks,
        since they don't run embed_input_ids."""
        runner = self._build_runner(embedding_tp=True)
        mock_pp_group.return_value.is_first_rank = False

        original_scheduled = 5
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = original_scheduled
        scheduler_output.scheduled_encoder_inputs = {}
        num_input_tokens = 10

        with (
            patch.object(
                GPUModelRunner,
                "_preprocess",
                side_effect=lambda *a, **kw: (None, None, None, None, None, None),
                autospec=True,
            ),
            contextlib.suppress(Exception),
        ):
            runner._preprocess(scheduler_output, num_input_tokens, None)

        assert scheduler_output.total_num_scheduled_tokens == original_scheduled, (
            "Should not override for non-first PP rank"
        )

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=True)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_preprocess_does_not_override_for_encoder_decoder(self, mock_pp_group, mock_embed_tp):
        """Embedding TP override should not apply for encoder-decoder models."""
        runner = self._build_runner(embedding_tp=True)
        runner.model_config.is_encoder_decoder = True
        mock_pp_group.return_value.is_first_rank = True

        original_scheduled = 5
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = original_scheduled
        scheduler_output.scheduled_encoder_inputs = {}
        num_input_tokens = 10

        with (
            patch.object(
                GPUModelRunner,
                "_preprocess",
                side_effect=lambda *a, **kw: (None, None, None, None, None, None),
                autospec=True,
            ),
            contextlib.suppress(Exception),
        ):
            runner._preprocess(scheduler_output, num_input_tokens, None)

        assert scheduler_output.total_num_scheduled_tokens == original_scheduled, (
            "Should not override for encoder-decoder models"
        )

    @patch("vllm_ascend.worker.model_runner_v1.embedding_tp_enable", return_value=True)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    def test_preprocess_does_not_override_for_non_mm_model(self, mock_pp_group, mock_embed_tp):
        """Embedding TP override should not apply for text-only models
        (supports_mm_inputs=False)."""
        runner = self._build_runner(embedding_tp=True, supports_mm=False)
        mock_pp_group.return_value.is_first_rank = True

        original_scheduled = 5
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = original_scheduled
        scheduler_output.scheduled_encoder_inputs = {}
        num_input_tokens = 10

        with (
            patch.object(
                GPUModelRunner,
                "_preprocess",
                side_effect=lambda *a, **kw: (None, None, None, None, None, None),
                autospec=True,
            ),
            contextlib.suppress(Exception),
        ):
            runner._preprocess(scheduler_output, num_input_tokens, None)

        assert scheduler_output.total_num_scheduled_tokens == original_scheduled, (
            "Should not override for text-only models"
        )


if __name__ == "__main__":
    unittest.main()
