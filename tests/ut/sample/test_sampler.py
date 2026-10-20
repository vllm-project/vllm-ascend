import math
import unittest
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
        patch("vllm_ascend.sample.sampler.npu_stream_switch", return_value=nullcontext(), create=True),
        patch("vllm_ascend.sample.sampler.global_stream", return_value=MagicMock(), create=True),
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


class TestRandomSampleNumerics(unittest.TestCase):
    """CPU numerical contracts; stream mocks do not validate NPU execution."""

    def sample(self, probabilities, generators=None):
        with _cpu_safe_npu_sample():
            return random_sample(probabilities, generators or {})

    def test_zero_mass_tokens_are_never_selected(self):
        probabilities = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).repeat(256, 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(18754)
            actual = self.sample(probabilities)
        expected = torch.tensor([1, 0]).repeat(256)
        torch.testing.assert_close(actual, expected)

    def test_empty_batch(self):
        actual = self.sample(torch.empty((0, 3)))
        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, torch.int64)

    def test_global_seed_reproduces_unseeded_sampling(self):
        probabilities = torch.tensor([[0.1, 0.3, 0.6]]).repeat(128, 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(18754)
            first = self.sample(probabilities.clone())
            torch.manual_seed(18754)
            second = self.sample(probabilities.clone())
        torch.testing.assert_close(first, second)

    def test_seeded_rows_do_not_depend_on_the_global_seed(self):
        probabilities = torch.tensor([[0.1, 0.3, 0.6]]).repeat(16, 1)
        for rows in ([0, 7, 15], list(range(16))):
            with self.subTest(rows=rows), torch.random.fork_rng(devices=[]):
                torch.manual_seed(10)
                first = self.sample(
                    probabilities.clone(),
                    {row: torch.Generator().manual_seed(100 + row) for row in rows},
                )
                torch.manual_seed(20)
                second = self.sample(
                    probabilities.clone(),
                    {row: torch.Generator().manual_seed(100 + row) for row in rows},
                )
            torch.testing.assert_close(first[rows], second[rows])

    def test_request_generator_state_advances(self):
        generator = torch.Generator().manual_seed(18754)
        original = generator.get_state().clone()
        self.sample(torch.tensor([[0.2, 0.3, 0.5]]), {0: generator})
        self.assertFalse(torch.equal(original, generator.get_state()))

    def test_distribution_matches_categorical_probabilities(self):
        count = 32768
        expected = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        probabilities = expected.to(torch.float32).expand(count, -1).clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(18754)
            tokens = self.sample(probabilities)
        observed = torch.bincount(tokens, minlength=3).to(torch.float64) / count
        for actual, probability in zip(observed.tolist(), expected.tolist()):
            # A fixed-seed six-sigma multinomial check, independent of the
            # implementation's exponential-race construction.
            tolerance = 6 * math.sqrt(probability * (1 - probability) / count)
            self.assertLessEqual(abs(actual - probability), tolerance)

    def test_forward_native_preserves_one_hot_and_logprob_modes(self):
        logits = torch.tensor([[-float("inf"), 0.0], [0.0, -float("inf")]])
        for mode in ("raw_logprobs", "processed_logits", "processed_logprobs"):
            with self.subTest(mode=mode):
                sampler = AscendTopKTopPSampler(logprobs_mode=mode)
                with _cpu_safe_npu_sample():
                    tokens, processed = sampler.forward_native(logits.clone(), {}, None, None)
                torch.testing.assert_close(tokens, torch.tensor([1, 0]))
                if mode == "raw_logprobs":
                    self.assertIsNone(processed)
                else:
                    self.assertEqual(processed.shape, logits.shape)

    def test_noise_generation_stays_on_the_calling_stream(self):
        # This white-box placement contract does not reproduce a device race.
        with patch(
            "vllm_ascend.sample.sampler.global_stream",
            side_effect=AssertionError("random_sample selected an auxiliary stream"),
            create=True,
        ):
            tokens = random_sample(torch.tensor([[0.0, 1.0]]), {})
        torch.testing.assert_close(tokens, torch.tensor([1]))
