# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for vllm_ascend.worker.v2.sample helpers.

Most v2 sample kernels (bad_words, min_p, the gumbel/topk-logprob kernels) are
Triton-Ascend kernels validated under ``tests/e2e/.../triton/`` on NPU. This
file covers the CPU-reachable surface only: the pure-PyTorch top-k/top-p
wrapper (whose both-None early return is a documented OOM-avoidance branch)
and the guard paths that raise *before* any Triton kernel is launched.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu


def test_apply_top_k_top_p_returns_logits_unchanged_when_both_none():
    logits = torch.randn(2, 8)
    out = apply_top_k_top_p_npu(logits, None, None)
    # Early return (warmup path): same object, contents untouched.
    assert out is logits


def test_apply_top_k_top_p_delegates_to_pytorch_when_k_given():
    logits = torch.randn(2, 8)
    k = torch.tensor([1, 2])
    p = None

    sentinel = torch.zeros(2, 8)
    pytorch_impl = Mock(return_value=sentinel)
    with patch(
        "vllm_ascend.worker.v2.sample.apply_top_k_top_p._apply_top_k_top_p_pytorch",
        new=pytorch_impl,
    ):
        out = apply_top_k_top_p_npu(logits, k, p)

    assert out is sentinel
    pytorch_impl.assert_called_once_with(logits, k, p)


def test_apply_top_k_top_p_delegates_when_only_p_given():
    logits = torch.randn(2, 8)
    p = torch.tensor([0.9, 0.5])

    sentinel = torch.zeros(2, 8)
    pytorch_impl = Mock(return_value=sentinel)
    with patch(
        "vllm_ascend.worker.v2.sample.apply_top_k_top_p._apply_top_k_top_p_pytorch",
        new=pytorch_impl,
    ):
        out = apply_top_k_top_p_npu(logits, None, p)

    assert out is sentinel
    pytorch_impl.assert_called_once_with(logits, None, p)


def test_gumbel_sample_rejects_fp64_before_kernel_launch():
    # use_fp64 is rejected before any tensor/triton work, so no NPU is needed.
    pytest.importorskip("vllm_ascend.worker.v2.sample.gumbel")
    from vllm_ascend.worker.v2.sample.gumbel import _gumbel_sample

    logits = torch.zeros(1, 4)
    mapping = torch.zeros(1, dtype=torch.int64)
    temperature = torch.tensor([1.0])
    seed = torch.tensor([0])
    pos = torch.tensor([0])
    with pytest.raises(NotImplementedError, match="FP64"):
        _gumbel_sample(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            apply_temperature=False,
            use_fp64=True,
        )


def test_compute_topk_logprobs_rejects_negative_num_logprobs():
    # The assert guards the kernel launch and runs first, so no NPU is needed.
    pytest.importorskip("vllm_ascend.worker.v2.sample.logprob")
    from vllm_ascend.worker.v2.sample.logprob import compute_topk_logprobs

    logits = torch.zeros(1, 4)
    sampled = torch.tensor([[0]])
    with pytest.raises(AssertionError):
        compute_topk_logprobs(logits, num_logprobs=-1, sampled_token_ids=sampled)

