from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.model_executor.warmup.v2_rejection_sampler_warmup import (
    warmup_v2_model_dtype_rejection_sampler,
)


def _runner(dtype=torch.bfloat16, draft_logits=None, last_pp_rank=True):
    return SimpleNamespace(
        is_last_pp_rank=last_pp_rank,
        num_speculative_steps=2,
        model_config=SimpleNamespace(dtype=dtype, get_vocab_size=lambda: 32),
        speculator=SimpleNamespace(
            draft_logits=draft_logits if draft_logits is not None else torch.empty(1, 2, 32, dtype=torch.bfloat16)
        ),
        device=torch.device("cpu"),
    )


def test_warms_model_dtype_logits():
    with patch(
        "vllm_ascend.model_executor.warmup.v2_rejection_sampler_warmup.rejection_sample"
    ) as sample:
        warmup_v2_model_dtype_rejection_sampler(_runner())

    sample.assert_called_once()
    args = sample.call_args.args
    assert args[0].dtype == torch.bfloat16
    assert args[0].shape == (3, 32)
    assert args[1].dtype == torch.bfloat16
    assert args[1].shape == (1, 2, 32)
    assert args[2].dtype == torch.int32
    assert args[3].tolist() == [0, 3]
    assert args[-1] == 2


def test_skips_fp32_and_non_sampling_rank():
    with patch(
        "vllm_ascend.model_executor.warmup.v2_rejection_sampler_warmup.rejection_sample"
    ) as sample:
        warmup_v2_model_dtype_rejection_sampler(_runner(dtype=torch.float32))
        warmup_v2_model_dtype_rejection_sampler(_runner(last_pp_rank=False))
    sample.assert_not_called()
