from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p_pytorch
from vllm.v1.sample.sampler import Sampler

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import (
    AscendSampler,
    AscendTopKTopPSampler,
    random_sample,
)
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import _apply_top_k_top_p_pytorch


@contextmanager
def _cpu_safe_npu_sample():
    with (
        patch("vllm_ascend.sample.sampler.npu_stream_switch", return_value=nullcontext()),
        patch("vllm_ascend.sample.sampler.global_stream", return_value=MagicMock()),
        patch("vllm_ascend.sample.sampler.torch.npu.current_stream", return_value=MagicMock()),
        patch.object(torch.Tensor, "record_stream", lambda self, _stream: None),
    ):
        yield


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)


def test_penalties_greedy_prepare_and_random_sample():
    logits = torch.tensor([[0.1, 0.8, 0.1], [0.4, 0.1, 0.5]])
    fallback_logits = logits.clone()

    with (
        patch("vllm_ascend.sample.sampler.HAS_TRITON", False),
        patch.object(Sampler, "apply_penalties", return_value=fallback_logits) as fallback,
    ):
        assert AscendSampler.apply_penalties(logits, SimpleNamespace(), []) is fallback_logits
        fallback.assert_called_once()

    no_penalty_meta = SimpleNamespace(no_penalties=True, prompt_token_ids=None)
    with patch("vllm_ascend.sample.sampler.HAS_TRITON", True):
        assert AscendSampler.apply_penalties(logits, no_penalty_meta, [[]]) is logits

    penalty_meta = SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=torch.tensor([[1, 2]]),
        presence_penalties=torch.zeros(2),
        frequency_penalties=torch.zeros(2),
        repetition_penalties=torch.ones(2),
    )
    with (
        patch("vllm_ascend.sample.sampler.HAS_TRITON", True),
        patch("vllm_ascend.sample.sampler.apply_all_penalties", return_value=fallback_logits) as apply_penalties,
    ):
        assert AscendSampler.apply_penalties(logits, penalty_meta, [[3], [4]]) is fallback_logits
        apply_penalties.assert_called_once()

    greedy = AscendSampler.greedy_sample(logits)
    assert greedy.tolist() == [1, 2]

    probs = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32)
    generators = {0: torch.Generator().manual_seed(0)}
    with _cpu_safe_npu_sample():
        partial = random_sample(probs.clone(), generators)
        full = random_sample(
            probs.clone(),
            {0: torch.Generator().manual_seed(1), 1: torch.Generator().manual_seed(2)},
        )
        empty = random_sample(probs.clone(), {})
    assert partial.shape == full.shape == empty.shape == (2,)


def test_topk_topp_forward_and_apply_helpers():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    k = torch.tensor([1, 4])
    p = torch.tensor([0.9, 1.0])
    sampler = AscendTopKTopPSampler(logprobs_mode="raw_logprobs")

    with (
        patch("vllm_ascend.sample.sampler.envs.VLLM_BATCH_INVARIANT", True),
        patch.object(TopKTopPSampler, "forward_native", return_value=("fallback", None)) as native,
    ):
        assert sampler.forward_native(logits, {}, k, p) == ("fallback", None)
        native.assert_called_once()

    def identity_apply(values, _k, _p):
        return values

    sampler.apply_top_k_top_p = identity_apply
    with _cpu_safe_npu_sample():
        for mode in ("processed_logits", "processed_logprobs", "raw_logprobs"):
            sampler.logprobs_mode = mode
            tokens, processed = sampler.forward_native(logits, {}, k, p)
            assert tokens.shape == (2,)
            if mode == "raw_logprobs":
                assert processed is None
            else:
                assert processed.shape == logits.shape

    unchanged = _apply_top_k_top_p_pytorch(logits.clone(), None, None)
    assert unchanged.shape == logits.shape
    filtered = _apply_top_k_top_p_pytorch(logits.clone(), k, p)
    assert filtered.shape == logits.shape
    assert torch.isfinite(filtered).any()
    top_k_only = _apply_top_k_top_p_pytorch(logits.clone(), torch.tensor([1, 1]), None)
    assert top_k_only.shape == logits.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k_value", [1, 10, 1024, 2000])
@pytest.mark.parametrize("p_value", [0.5, 0.9, 0.99, 1.0, None])
def test_apply_top_k_top_p_matches_upstream(dtype, k_value, p_value):
    """top-p must be applied to the distribution truncated by top-k.

    A pre-cut (full-vocabulary) distribution keeps extra tokens and, when the
    mass retained by top-k is <= p, collapses top-p into a no-op.
    """
    torch.manual_seed(0)
    vocab_size = 2000
    logits = torch.randn(4, vocab_size, dtype=torch.float32).to(dtype)
    k = torch.full((4,), k_value, dtype=torch.int64)
    p = None if p_value is None else torch.full((4,), p_value)

    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    actual = _apply_top_k_top_p_pytorch(logits.clone(), k, p)

    assert torch.equal(actual, expected)


def test_apply_top_k_top_p_mixed_and_tied_rows():
    """Per-row k/p and tied logits must match upstream row by row."""
    vocab_size = 2000
    torch.manual_seed(0)
    logits = torch.randn(4, vocab_size)
    k = torch.tensor([1, 10, 1024, vocab_size], dtype=torch.int64)
    p = torch.tensor([0.5, 0.9, 0.99, 1.0])

    assert torch.equal(
        _apply_top_k_top_p_pytorch(logits.clone(), k, p),
        apply_top_k_top_p_pytorch(logits.clone(), k, p),
    )

    # Every 20 tokens share one logit, so top-k alone cannot separate them and
    # only a re-normalized top-p can cut inside the tie group.
    tied = torch.floor(torch.arange(vocab_size, dtype=torch.float32) / 20.0).repeat(2, 1)
    tied_k = torch.tensor([10, 10], dtype=torch.int64)
    tied_p = torch.tensor([0.5, 0.9])
    assert torch.equal(
        _apply_top_k_top_p_pytorch(tied.clone(), tied_k, tied_p),
        apply_top_k_top_p_pytorch(tied.clone(), tied_k, tied_p),
    )


def test_apply_top_k_top_p_skips_softmax_without_top_p():
    """The pure top-k path must not pay for a full-vocabulary softmax."""
    logits = torch.randn(2, 128)
    k = torch.tensor([5, 50], dtype=torch.int64)

    calls = []
    original_softmax = torch.Tensor.softmax

    def counting_softmax(self, *args, **kwargs):
        calls.append(1)
        return original_softmax(self, *args, **kwargs)

    with patch.object(torch.Tensor, "softmax", counting_softmax):
        _apply_top_k_top_p_pytorch(logits.clone(), k, None)

    assert calls == []
