# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.model_executor.warmup.v2_rejection_sampler_warmup import (
    warmup_v2_model_dtype_rejection_sampler,
)


def _runner(dtype=torch.bfloat16, draft_dtype=torch.float32, draft_vocab_size=32, synthetic_rates=None):
    draft_logits = None if draft_dtype is None else torch.empty(1, 2, draft_vocab_size, dtype=draft_dtype)
    return SimpleNamespace(
        is_last_pp_rank=True,
        num_speculative_steps=2,
        model_config=SimpleNamespace(dtype=dtype, get_vocab_size=lambda: 32),
        speculator=SimpleNamespace(draft_logits=draft_logits),
        rejection_sampler=SimpleNamespace(synthetic_conditional_rates=synthetic_rates),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("draft_dtype", [None, torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("draft_vocab_size", [16, 32])
def test_warms_model_dtype_logits(dtype, draft_dtype, draft_vocab_size):
    runner = _runner(dtype, draft_dtype, draft_vocab_size)
    with patch("vllm_ascend.model_executor.warmup.v2_rejection_sampler_warmup.rejection_sample") as sample:
        warmup_v2_model_dtype_rejection_sampler(runner)

    sample.assert_called_once()
    args = sample.call_args.args
    assert args[0].dtype == dtype
    assert args[0].shape == (3, 32)
    if draft_dtype is None:
        assert args[1] is None
    else:
        assert args[1].dtype == draft_dtype
        assert args[1].shape == (1, 2, draft_vocab_size)
        assert args[1].data_ptr() != runner.speculator.draft_logits.data_ptr()
    assert args[2].dtype == torch.int32
    assert args[2].tolist() == [0, 0, 0]
    assert args[3].dtype == torch.int32
    assert args[3].tolist() == [0, 3]
    assert args[4].dtype == torch.int64
    assert args[4].tolist() == [0, 1, 2]
    assert args[5].dtype == torch.int64
    assert args[5].tolist() == [0]
    assert args[6].dtype == torch.int64
    assert args[6].tolist() == [0, 0, 0]
    assert args[7].dtype == torch.int32
    assert args[7].tolist() == [0, 1, 2]
    assert args[8].dtype == torch.float32
    assert args[8].tolist() == [1.0]
    assert args[9].dtype == torch.int64
    assert args[9].tolist() == [0]
    assert args[10] == 2
    assert sample.call_args.kwargs["synthetic_conditional_rates"] is None
    assert all(t.device.type == "cpu" for t in args if isinstance(t, torch.Tensor))


@pytest.mark.parametrize("draft_dtype", [None, torch.float32])
def test_warms_configured_synthetic_acceptance(draft_dtype):
    rates = torch.tensor([0.5, 0.25], dtype=torch.float32)
    with patch("vllm_ascend.model_executor.warmup.v2_rejection_sampler_warmup.rejection_sample") as sample:
        warmup_v2_model_dtype_rejection_sampler(_runner(draft_dtype=draft_dtype, synthetic_rates=rates))
    assert sample.call_args.kwargs["synthetic_conditional_rates"] is rates


@pytest.mark.parametrize(
    "runner",
    [
        SimpleNamespace(is_last_pp_rank=False),
        SimpleNamespace(is_last_pp_rank=True, num_speculative_steps=0),
        SimpleNamespace(
            is_last_pp_rank=True,
            num_speculative_steps=2,
            model_config=SimpleNamespace(dtype=torch.float32),
        ),
        SimpleNamespace(
            is_last_pp_rank=True,
            num_speculative_steps=2,
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            speculator=None,
        ),
    ],
    ids=["non_sampling_rank", "no_speculation", "fp32", "no_speculator"],
)
def test_skips_unused_variants(runner):
    with patch("vllm_ascend.model_executor.warmup.v2_rejection_sampler_warmup.rejection_sample") as sample:
        warmup_v2_model_dtype_rejection_sampler(runner)
    sample.assert_not_called()
