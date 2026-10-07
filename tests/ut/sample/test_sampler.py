from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import (
    AscendSampler,
    AscendTopKTopPSampler,
    _apply_top_k_top_p_pytorch,
    _apply_top_k_top_p_torch_npu,
    random_sample,
)


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
    npu_unchanged = _apply_top_k_top_p_torch_npu(logits.clone(), None, None)
    assert npu_unchanged.shape == logits.shape
    npu_filtered = _apply_top_k_top_p_torch_npu(logits.clone(), k, p)
    assert npu_filtered.shape == logits.shape
