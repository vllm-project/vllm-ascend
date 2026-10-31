# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the MRV2 DSpark local-argmax helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.spec_decode.dspark.local_argmax import (
    MAX_EXACT_FP32_TOKEN_ID,
    reduce_local_argmax,
)


def test_reduce_local_argmax_single_rank_masks_padded_vocab() -> None:
    scores = torch.tensor(
        [
            [1.0, 4.0, 100.0],
            [5.0, 2.0, 99.0],
        ]
    )
    all_gather = MagicMock()

    result = reduce_local_argmax(
        scores,
        vocab_start=10,
        vocab_end=12,
        vocab_size=12,
        tp_size=1,
        all_gather=all_gather,
    )

    assert result.tolist() == [11, 10]
    assert result.dtype == torch.int64
    assert torch.isneginf(scores[:, 2]).all()
    all_gather.assert_not_called()


def test_reduce_local_argmax_selects_global_winner_and_preserves_ties() -> None:
    local_scores = torch.tensor(
        [
            [4.0, 3.0],
            [1.0, 5.0],
        ],
        dtype=torch.bfloat16,
    )

    def all_gather(local_pair: torch.Tensor) -> torch.Tensor:
        assert local_pair.shape == (2, 2)
        assert local_pair.dtype == torch.float32
        remote_pair = torch.tensor(
            [
                [6.0, 3.0],
                [5.0, 2.0],
            ]
        )
        return torch.cat((local_pair, remote_pair), dim=-1)

    result = reduce_local_argmax(
        local_scores,
        vocab_start=0,
        vocab_end=2,
        vocab_size=4,
        tp_size=2,
        all_gather=all_gather,
    )

    # Request 0 is won by rank 1. Request 1 is tied, so rank 0 (the lower
    # contiguous vocabulary shard) preserves full-vocabulary argmax behavior.
    assert result.tolist() == [3, 1]


def test_reduce_local_argmax_rejects_inexact_fp32_token_ids() -> None:
    with pytest.raises(ValueError, match="exact FP32 token-ID range"):
        reduce_local_argmax(
            torch.ones(1, 1),
            vocab_start=0,
            vocab_end=1,
            vocab_size=MAX_EXACT_FP32_TOKEN_ID + 1,
            tp_size=1,
            all_gather=MagicMock(),
        )


def test_deepseek_v4_local_logits_normalize_and_skip_gather() -> None:
    model = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    hidden_states = torch.arange(6, dtype=torch.float32).view(2, 3)
    normalized = hidden_states + 1
    local_logits = torch.arange(8, dtype=torch.float32).view(2, 4)
    norm = MagicMock(return_value=normalized)
    logits_processor = MagicMock(return_value=local_logits)
    lm_head = object()
    model.model = SimpleNamespace(norm=norm)
    model.lm_head = lm_head
    model.logits_processor = logits_processor

    result = model.compute_local_draft_logits(hidden_states)

    assert result is local_logits
    norm.assert_called_once_with(hidden_states)
    logits_processor.assert_called_once_with(
        lm_head,
        normalized,
        skip_gather=True,
    )


def test_model_with_context_forwards_local_logits() -> None:
    class DraftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compute = MagicMock(return_value=torch.tensor([[1.0, 2.0]]))

        def compute_local_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.compute(hidden_states)

    original_model = DraftModel()
    wrapped_model = ModelWithContext(original_model)
    hidden_states = torch.ones(1, 3)

    result = wrapped_model.compute_local_draft_logits(hidden_states)

    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0]]))
    original_model.compute.assert_called_once_with(hidden_states)
