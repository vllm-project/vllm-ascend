# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Tests for vllm_ascend.worker.v2.sample.gumbel on Ascend NPU.
# Validates gumbel_sample and apply_temperature against PyTorch references.

import math

import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.worker.v2.sample.gumbel import apply_temperature, gumbel_sample
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _npu_gumbel_block_argmax

DEVICE = "npu"


def _ref_apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Pure-Python reference for temperature scaling."""
    out = logits.clone().float()
    for tok in range(logits.shape[0]):
        req = expanded_idx_mapping[tok].item()
        temp = temperature[req].item()
        if temp == 0.0 or temp == 1.0:
            continue
        out[tok] = out[tok] / temp
    return out


class TestGumbelSampling:
    @pytest.mark.parametrize(
        "num_tokens,vocab_size",
        [
            (1, 32000),
            (8, 32000),
            (48, 102400),
            (64, 151936),
        ],
    )
    def test_apply_temperature(self, num_tokens, vocab_size):
        """Temperature kernel matches PyTorch reference for various vocab sizes."""
        torch.manual_seed(0)
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.randint(0, num_tokens, (num_tokens,), dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_tokens, dtype=torch.float32, device=DEVICE) * 1.8 + 0.2
        # inject edge cases
        temperature[0] = 0.0
        if num_tokens > 1:
            temperature[1] = 1.0

        logits_triton = logits.clone()
        apply_temperature(logits_triton, expanded_idx_mapping, temperature)
        torch.npu.synchronize()

        logits_ref = _ref_apply_temperature(logits, expanded_idx_mapping, temperature)

        assert torch.allclose(logits_triton.float(), logits_ref, atol=1e-4, rtol=1e-5), (
            f"apply_temperature mismatch: max_diff={(logits_triton.float() - logits_ref).abs().max().item():.6f}"
        )

    def test_apply_temperature_skip_zero_and_one(self):
        """Logits should be unchanged for temp=0.0 and temp=1.0."""
        torch.manual_seed(10)
        num_tokens = 4
        vocab_size = 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32, device=DEVICE)

        original = logits.clone()
        apply_temperature(logits, expanded_idx_mapping, temperature)
        torch.npu.synchronize()

        assert torch.equal(logits, original), "Logits changed for temp=0.0 or temp=1.0"

    @pytest.mark.parametrize(
        "num_tokens,num_reqs,vocab_size",
        [
            (1, 1, 32000),
            (4, 4, 32000),
            (8, 4, 32000),  # expanded: multiple tokens per request
            (16, 8, 102400),
        ],
    )
    def test_gumbel_sample_greedy(self, num_tokens, num_reqs, vocab_size):
        """temperature=0 must return argmax (greedy)."""
        torch.manual_seed(42)
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE)
        temperature = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()

        expected = logits.argmax(dim=-1)
        assert torch.equal(sampled, expected), (
            f"Greedy mismatch: sampled={sampled.tolist()} expected={expected.tolist()}"
        )

    def test_gumbel_sample_greedy_apply_temp_flag_irrelevant(self):
        """With temp=0, apply_temperature flag should not affect result (both greedy)."""
        torch.manual_seed(55)
        num_tokens, num_reqs, vocab_size = 4, 4, 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        s_false = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        s_true = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True, is_drafting=False
        )
        torch.npu.synchronize()

        expected = logits.argmax(dim=-1)
        assert torch.equal(s_false, expected)
        assert torch.equal(s_true, expected)

    @pytest.mark.parametrize(
        "num_tokens,num_reqs,vocab_size",
        [
            (4, 4, 32000),
            (8, 4, 32000),
            (16, 8, 102400),
        ],
    )
    def test_gumbel_sample_deterministic(self, num_tokens, num_reqs, vocab_size):
        """Same seed must produce identical results across runs."""
        torch.manual_seed(7)
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        r1 = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()
        r2 = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()

        assert torch.equal(r1, r2), "gumbel_sample is non-deterministic with same seed"

    def test_gumbel_sample_different_seeds(self):
        """Different seeds must (almost surely) produce different results."""
        torch.manual_seed(8)
        num_tokens, num_reqs, vocab_size = 16, 16, 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE) * 1.0
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        seed1 = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        seed2 = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        # Ensure seeds differ
        seed2[0] = seed1[0] + 1

        r1 = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed1, pos, apply_temperature=False, is_drafting=False
        )
        r2 = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed2, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()

        # With 16 tokens and vocab 32000 at temp=1.0, identical results are astronomically unlikely
        assert not torch.equal(r1, r2), "Different seeds produced identical results"

    @pytest.mark.parametrize(
        "num_tokens,num_reqs,vocab_size",
        [
            (4, 4, 32000),
            (8, 4, 32000),
            (16, 8, 102400),
        ],
    )
    def test_gumbel_sample_valid_token_ids(self, num_tokens, num_reqs, vocab_size):
        """Sampled token IDs must be in [0, vocab_size)."""
        torch.manual_seed(3)
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.randint(0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) + 0.1
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()

        assert sampled.shape == (num_tokens,)
        assert (sampled >= 0).all() and (sampled < vocab_size).all(), (
            f"Out-of-range token IDs: min={sampled.min()}, max={sampled.max()}"
        )

    def test_gumbel_sample_temperature_affects_distribution(self):
        """Higher temperature should increase sampling entropy (less concentrated).

        Strategy: create logits with a clear winner. At low temp the winner should
        be sampled most often. At high temp other tokens get more probability.
        """
        vocab_size = 100
        num_trials = 256
        logits_base = torch.zeros(1, vocab_size, dtype=torch.float32, device=DEVICE)
        logits_base[0, 0] = 10.0  # strong signal at token 0

        expanded_idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)

        low_temp = torch.tensor([0.1], dtype=torch.float32, device=DEVICE)
        high_temp = torch.tensor([5.0], dtype=torch.float32, device=DEVICE)

        low_temp_winner_count = 0
        high_temp_winner_count = 0

        for i in range(num_trials):
            seed = torch.tensor([i * 1000 + 42], dtype=torch.int64, device=DEVICE)
            pos = torch.tensor([i], dtype=torch.int32, device=DEVICE)

            s_low = gumbel_sample(
                logits_base.clone(),
                expanded_idx_mapping,
                low_temp,
                seed,
                pos,
                apply_temperature=True,
                is_drafting=False,
            )
            s_high = gumbel_sample(
                logits_base.clone(),
                expanded_idx_mapping,
                high_temp,
                seed,
                pos,
                apply_temperature=True,
                is_drafting=False,
            )
            if s_low.item() == 0:
                low_temp_winner_count += 1
            if s_high.item() == 0:
                high_temp_winner_count += 1

        torch.npu.synchronize()
        # Low temp should pick the winner much more often than high temp
        assert low_temp_winner_count > high_temp_winner_count, (
            f"Low temp winner count ({low_temp_winner_count}) should be > "
            f"high temp winner count ({high_temp_winner_count})"
        )
        # Low temp with such a strong signal should almost always pick token 0
        assert low_temp_winner_count > num_trials * 0.9, (
            f"Low temp winner count ({low_temp_winner_count}/{num_trials}) should be >90%"
        )

    @pytest.mark.parametrize(
        "num_tokens,num_reqs,vocab_size",
        [
            (4, 4, 32000),
            (8, 4, 32000),
        ],
    )
    def test_gumbel_sample_mixed_temperature(self, num_tokens, num_reqs, vocab_size):
        """Mix of temp=0 and temp>0: temp=0 tokens must be greedy."""
        torch.manual_seed(11)
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        # identity mapping: token i -> request i (for simplicity)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_tokens, dtype=torch.float32, device=DEVICE) + 0.5
        # force first half to greedy
        temperature[: num_tokens // 2] = 0.0
        seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()

        greedy = logits.argmax(dim=-1)
        for tok in range(num_tokens // 2):
            assert sampled[tok].item() == greedy[tok].item(), (
                f"Token {tok} (temp=0) should be greedy: got {sampled[tok].item()}, expected {greedy[tok].item()}"
            )

    def test_gumbel_sample_expanded_idx_mapping(self):
        """Multiple tokens mapping to the same request must work correctly."""
        torch.manual_seed(99)
        num_tokens = 6
        num_reqs = 2
        vocab_size = 32000

        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        # tokens 0,1,2 -> req 0; tokens 3,4,5 -> req 1
        expanded_idx_mapping = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=DEVICE)
        temperature = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()

        expected = logits.argmax(dim=-1)
        assert torch.equal(sampled, expected), (
            f"Expanded mapping greedy mismatch: {sampled.tolist()} vs {expected.tolist()}"
        )

    def test_gumbel_sample_shared_seed_same_request(self):
        """Tokens mapping to the same request share seed, so with same pos they
        should produce the same Gumbel noise and therefore the same sample (given
        same logits)."""
        torch.manual_seed(42)
        vocab_size = 32000
        num_reqs = 1

        # Two tokens with identical logits, same request, same position
        logits_row = torch.randn(1, vocab_size, dtype=torch.float32, device=DEVICE)
        logits = logits_row.repeat(2, 1)
        expanded_idx_mapping = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
        temperature = torch.tensor([0.8], dtype=torch.float32, device=DEVICE)
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        # Same pos -> same Gumbel noise
        pos = torch.tensor([5, 5], dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True, is_drafting=False
        )
        torch.npu.synchronize()

        assert sampled[0].item() == sampled[1].item(), (
            f"Tokens with same logits, seed, and pos should sample the same token: "
            f"got {sampled[0].item()} vs {sampled[1].item()}"
        )

    def test_gumbel_sample_apply_temperature_true_nonzero(self):
        """apply_temperature=True with temp>0 must divide logits by temperature
        before adding Gumbel noise. The logits cache stores the input logits
        *before* temperature; consumers (the rejection sampler) divide by the
        same temperature on load."""
        torch.manual_seed(77)
        num_tokens, num_reqs, vocab_size = 4, 4, 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        out_logits = torch.zeros(num_reqs, vocab_size, dtype=torch.float32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=False,
            logits_cache=out_logits,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            # The cache stores the pre-temperature logits.
            expected = logits[tok].float()
            assert torch.allclose(out_logits[req].float(), expected, atol=1e-4, rtol=1e-4), (
                f"logits_cache mismatch at token {tok} (req {req}): "
                f"max_diff={(out_logits[req].float() - expected).abs().max().item():.6f}"
            )

    def test_gumbel_sample_apply_temperature_false_nonzero(self):
        """apply_temperature=False with temp>0: the logits cache must contain
        raw logits (no temperature division), but Gumbel noise is still added
        to sampling."""
        torch.manual_seed(78)
        num_tokens, num_reqs, vocab_size = 4, 4, 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        out_logits = torch.zeros(num_reqs, vocab_size, dtype=torch.float32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=False,
            is_drafting=False,
            logits_cache=out_logits,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            # Without temperature application, stored logits should match raw logits
            expected = logits[tok].float()
            assert torch.allclose(out_logits[req].float(), expected, atol=1e-4, rtol=1e-4), (
                f"logits_cache should be raw logits when apply_temperature=False: "
                f"max_diff={(out_logits[req].float() - expected).abs().max().item():.6f}"
            )

    def test_gumbel_sample_logits_cache_req_state_idx(self):
        """Cached logits must be stored at req_state_idx position, not token_idx.

        This tests the EAGLE speculative decoding scenario where the idx_mapping
        is non-contiguous (e.g., active requests [2,5,7,0] out of 8 slots).
        The buffer is shaped [max_num_reqs, vocab_size] and the kernel must store
        at the correct request slot.
        """
        torch.manual_seed(200)
        num_tokens = 4
        max_num_reqs = 8
        vocab_size = 4096

        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        # Non-contiguous mapping: tokens 0-3 map to requests 2,5,7,0
        expanded_idx_mapping = torch.tensor([2, 5, 7, 0], dtype=torch.int32, device=DEVICE)
        temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=DEVICE) * 0.8
        seed = torch.randint(0, 2**31, (max_num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        out_logits = torch.zeros(max_num_reqs, vocab_size, dtype=torch.float32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=False,
            logits_cache=out_logits,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            # The cache stores the pre-temperature logits.
            expected = logits[tok].float()
            actual = out_logits[req]
            assert torch.allclose(actual.float(), expected, atol=1e-4, rtol=1e-4), (
                f"Req {req} (tok={tok}): max_diff={(actual.float() - expected).abs().max().item():.6f}"
            )

        # Also verify that unused request slots remain zero
        used_reqs = set(expanded_idx_mapping.tolist())
        for req in range(max_num_reqs):
            if req not in used_reqs:
                assert (out_logits[req] == 0).all(), f"Unused request slot {req} should be all zeros"

    def test_gumbel_sample_logits_cache_col(self):
        """logits_cache_col selects which column (draft step) to write.

        Simulates EAGLE with buffer [max_num_reqs, num_steps, vocab_size].
        """
        torch.manual_seed(201)
        num_tokens = 3
        max_num_reqs = 4
        vocab_size = 2048
        num_steps = 3

        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=DEVICE) * 0.9
        seed = torch.randint(0, 2**31, (max_num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        # Buffer: [max_num_reqs, num_steps, vocab_size]
        draft_logits = torch.zeros(max_num_reqs, num_steps, vocab_size, dtype=torch.float32, device=DEVICE)

        # Write to column (step) 1
        col_tensor = torch.tensor(1, dtype=torch.int32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=False,
            logits_cache=draft_logits,
            logits_cache_col=col_tensor,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            # The cache stores the pre-temperature logits.
            expected = logits[tok].float()
            # Data should be at draft_logits[req, 1, :]  (column 1)
            actual = draft_logits[req, 1, :]
            assert torch.allclose(actual.float(), expected, atol=1e-4, rtol=1e-4), (
                f"Token {tok} at col=1: mismatch, max_diff={(actual.float() - expected).abs().max().item():.6f}"
            )
            # Column 0 and 2 should be untouched (zeros)
            assert (draft_logits[req, 0, :] == 0).all(), f"Col 0 should be zeros for req {req}"
            assert (draft_logits[req, 2, :] == 0).all(), f"Col 2 should be zeros for req {req}"

    def test_gumbel_sample_logits_cache_mixed_temp(self):
        """Cached logits with mixed temperature (1:1 token-to-request mapping):
        the cache always stores the raw input logits regardless of temperature.

        Note: In practice, the logits cache is only used by EAGLE
        speculative decoding, which always has 1:1 token-to-request mapping.
        Multiple tokens per request would cause a write race (undefined order).
        """
        torch.manual_seed(88)
        num_tokens = 4
        num_reqs = 4
        vocab_size = 4096

        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        # 1:1 mapping: token i -> request i (matches EAGLE usage)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.tensor([0.0, 0.8, 1.5, 0.0], dtype=torch.float32, device=DEVICE)
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        out_logits = torch.zeros(num_reqs, vocab_size, dtype=torch.float32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=False,
            logits_cache=out_logits,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            # The cache stores the pre-temperature logits.
            expected = logits[tok].float()
            actual = out_logits[req]
            assert torch.allclose(actual.float(), expected, atol=1e-4, rtol=1e-4), (
                f"Req {req} (tok={tok}): max_diff={(actual.float() - expected).abs().max().item():.6f}"
            )

    def test_gumbel_sample_single_token(self):
        """Single token with temperature > 0 should work."""
        torch.manual_seed(42)
        logits = torch.randn(1, 32000, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        temperature = torch.tensor([0.7], dtype=torch.float32, device=DEVICE)
        seed = torch.tensor([12345], dtype=torch.int64, device=DEVICE)
        pos = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True, is_drafting=False
        )
        torch.npu.synchronize()

        assert sampled.shape == (1,)
        assert 0 <= sampled.item() < 32000

    def test_gumbel_sample_large_vocab(self):
        """Large vocabulary (151936 = Qwen2) should work correctly."""
        torch.manual_seed(401)
        vocab_size = 151936
        num_tokens = 4
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE)
        seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(
            logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False
        )
        torch.npu.synchronize()

        expected = logits.argmax(dim=-1)
        assert torch.equal(sampled, expected), "Large vocab greedy mismatch"

    def test_gumbel_sample_extreme_temperatures(self):
        """Very low and very high temperatures should not crash."""
        torch.manual_seed(42)
        num_tokens, vocab_size = 4, 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        # Very low temperature (near-greedy)
        low_temp = torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=torch.float32, device=DEVICE)
        s1 = gumbel_sample(logits, expanded_idx_mapping, low_temp, seed, pos, apply_temperature=True, is_drafting=False)
        torch.npu.synchronize()
        assert (s1 >= 0).all() and (s1 < vocab_size).all()

        # Very high temperature (near-uniform)
        high_temp = torch.tensor([100.0, 100.0, 100.0, 100.0], dtype=torch.float32, device=DEVICE)
        s2 = gumbel_sample(
            logits, expanded_idx_mapping, high_temp, seed, pos, apply_temperature=True, is_drafting=False
        )
        torch.npu.synchronize()
        assert (s2 >= 0).all() and (s2 < vocab_size).all()


def _float_bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret floats as same-width ints, so equality is truly bitwise."""
    int_dtype = torch.int32 if t.dtype == torch.float32 else torch.int16
    return t.view(int_dtype)


class TestGumbelSampleLogitsCache:
    """Regression tests for the draft logits cache (upstream #50910)."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("per_token_col", [False, True])
    def test_logits_cache_stores_input_logits_bitwise(self, dtype: torch.dtype, per_token_col: bool):
        """`logits_cache` must receive the input logits, pre-temperature and bit-exact.

        The cache is kept in the head's dtype, which is only lossless because the
        stored value is the input logit itself. Storing `logit / temp` instead would
        generally not be representable there, and the rejection sampler -- which
        divides by the same temperature on load -- would then verify against a `q`
        the draft never sampled from. Temperatures 0.0 and 1.0 are included because
        both make the divide a no-op and would mask such a bug.

        Both column-addressing modes are covered: a 0-d column (one draft step per
        call, as MTP/EAGLE do) and a [num_tokens] column (as DFlash does, sampling
        every step in one call).
        """
        torch.manual_seed(0)
        num_reqs, vocab_size, num_steps = 8, 4099, 3
        logits = torch.randn(num_reqs, vocab_size, device=DEVICE, dtype=dtype)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        temp = torch.tensor([0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 0.7, 1.0], dtype=torch.float32, device=DEVICE)
        seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
        # NPU: pos is cast to int32 inside the kernel.
        pos = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)

        cache = torch.zeros(num_reqs, num_steps, vocab_size, device=DEVICE, dtype=dtype)
        if per_token_col:
            # Each token lands in its own column, cycling through the steps.
            cols = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE) % num_steps
        else:
            cols = torch.tensor(1, dtype=torch.int32, device=DEVICE)

        gumbel_sample(
            logits,
            idx_mapping,
            temp,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=cache,
            logits_cache_col=cols,
        )
        torch.npu.synchronize()

        if per_token_col:
            stored = cache[torch.arange(num_reqs, device=DEVICE), cols.long()]
        else:
            stored = cache[:, 1]
            # Untouched columns must stay untouched.
            assert not cache[:, 0].any() and not cache[:, 2].any()
        assert torch.equal(_float_bits(stored), _float_bits(logits)), "cached logits differ from the input logits"

    @pytest.mark.parametrize("extra_cache_cols", [0, 1])
    def test_logits_cache_columns_stay_separate_across_steps(self, extra_cache_cols: int):
        """Each drafting step must land in its own cache column.

        `extra_cache_cols=1` is the shape a draft produces when it adds an
        input-only mask/noise row to its embedding table but keeps a full-width
        output head: N-wide logits cached into N+1-wide rows. Striding cache
        columns by the logits width instead of the cache's own row width leaves
        step 0 correct and silently misaligns every step after it.
        """
        torch.manual_seed(0)
        num_reqs, vocab_size, num_steps = 4, 1031, 3
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        temp = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
        seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
        # NPU: pos is cast to int32 inside the kernel.
        pos = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)

        cache = torch.zeros(num_reqs, num_steps, vocab_size + extra_cache_cols, device=DEVICE)
        cols = torch.arange(num_steps, dtype=torch.int32, device=DEVICE)
        per_step = [torch.randn(num_reqs, vocab_size, device=DEVICE) for _ in range(num_steps)]

        for step, logits in enumerate(per_step):
            gumbel_sample(
                logits,
                idx_mapping,
                temp,
                seed,
                pos,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=cache,
                logits_cache_col=cols[step],
            )
        torch.npu.synchronize()

        for step, logits in enumerate(per_step):
            stored = cache[:, step, :vocab_size]
            assert torch.equal(_float_bits(stored), _float_bits(logits)), f"step {step} was overwritten by a later step"
        # Columns past the sampled width belong to no step and stay untouched.
        assert not cache[:, :, vocab_size:].any()

    def test_logits_cache_narrower_than_logits_is_rejected(self):
        """A cache too narrow to hold a step would silently drop its tail."""
        num_reqs, vocab_size, num_steps = 2, 64, 3
        logits = torch.randn(num_reqs, vocab_size, device=DEVICE)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        temp = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
        seed = torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        cache = torch.zeros(num_reqs, num_steps, vocab_size - 1, device=DEVICE)

        with pytest.raises(AssertionError, match="narrower"):
            gumbel_sample(
                logits,
                idx_mapping,
                temp,
                seed,
                pos,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=cache,
                logits_cache_col=torch.tensor(0, dtype=torch.int32, device=DEVICE),
            )


@triton.jit
def _npu_gumbel_block_argmax_wrapper_kernel(
    # [num_tokens, V]
    logits_ptr,
    # [num_tokens]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_tokens]
    pos_ptr,
    # [max_num_reqs, num_cols, V]
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    # [] (shared column) or [num_tokens] (per-token column)
    cache_col_ptr,
    vocab_size,
    # [num_tokens]
    value_out_ptr,
    idx_out_ptr,
    APPLY_TEMPERATURE: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Minimal launcher so _npu_gumbel_block_argmax's cache path can be tested
    # directly: it is a device function, so a host-side call is impossible.
    token_idx = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * vocab_size + block, mask=mask, other=float("-inf"))
    value, idx = _npu_gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        cache_ptr,
        cache_stride_0,
        cache_stride_1,
        cache_col_ptr,
        vocab_size,
        IS_DRAFTING=False,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=False,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    tl.store(value_out_ptr + token_idx, value)
    tl.store(idx_out_ptr + token_idx, idx)


class TestNpuGumbelBlockArgmaxCache:
    """Direct coverage for _npu_gumbel_block_argmax's logits-cache path.

    In production the rejection sampler's _resample_kernel calls the op with
    logits_cache_ptr=None, so the cache path only runs when the NPU op is
    patched into the upstream kernels (draft logits caching, upstream #50910).
    These tests launch the op through a wrapper kernel (mirroring the
    TestGumbelSampleLogitsCache assertions for gumbel_sample's cache) so the
    fork-owned cache path keeps a regression guard.
    """

    BLOCK_SIZE = 8192

    def _run(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        temp: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        cache: torch.Tensor,
        cols: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, vocab_size = logits.shape
        value_out = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE)
        idx_out = torch.empty(num_tokens, dtype=torch.int64, device=DEVICE)
        _npu_gumbel_block_argmax_wrapper_kernel[(num_tokens,)](
            logits,
            idx_mapping,
            temp,
            seed,
            pos,
            cache,
            cache.stride(0),
            cache.stride(1),
            cols,
            vocab_size,
            value_out,
            idx_out,
            APPLY_TEMPERATURE=True,
            PER_TOKEN_COL=cols.ndim > 0,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )
        torch.npu.synchronize()
        return value_out, idx_out

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("per_token_col", [False, True])
    @torch.inference_mode()
    def test_cache_path_stores_input_logits_bitwise(self, dtype: torch.dtype, per_token_col: bool):
        """The op's cache store must be pre-temperature and bit-exact.

        Temps far from 0.0/1.0 make a post-temperature store fail the bitwise
        check in every dtype. Both column-addressing modes are covered (shared
        0-d column and per-token column), matching TestGumbelSampleLogitsCache.
        """
        torch.manual_seed(0)
        num_reqs, vocab_size, num_steps = 8, 4099, 3
        logits = torch.randn(num_reqs, vocab_size, device=DEVICE, dtype=dtype)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        temp = torch.tensor([0.1, 0.5, 1.5, 2.0, 0.7, 0.3, 0.9, 1.1], dtype=torch.float32, device=DEVICE)
        seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
        # NPU: pos is cast to int32 inside the kernel.
        pos = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)

        cache = torch.zeros(num_reqs, num_steps, vocab_size, device=DEVICE, dtype=dtype)
        if per_token_col:
            # Each token lands in its own column, cycling through the steps.
            cols = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE) % num_steps
        else:
            cols = torch.tensor(1, dtype=torch.int32, device=DEVICE)

        _, idx_out = self._run(logits, idx_mapping, temp, seed, pos, cache, cols)
        # Same (seed, pos) must reproduce the same noisy draw.
        _, idx_out_2 = self._run(logits, idx_mapping, temp, seed, pos, cache, cols)
        assert torch.equal(idx_out, idx_out_2), "same (seed, pos) produced different draws"

        if per_token_col:
            stored = cache[torch.arange(num_reqs, device=DEVICE), cols.long()]
        else:
            stored = cache[:, 1]
            # Untouched columns must stay untouched.
            assert not cache[:, 0].any() and not cache[:, 2].any()
        assert torch.equal(_float_bits(stored), _float_bits(logits)), "cached logits differ from the input logits"
        assert ((idx_out >= 0) & (idx_out < vocab_size)).all()

    @torch.inference_mode()
    def test_cache_path_greedy_returns_argmax(self):
        """temp=0 applies neither temperature nor noise: argmax and raw max."""
        torch.manual_seed(1)
        num_reqs, vocab_size, num_steps = 4, 1031, 3
        logits = torch.randn(num_reqs, vocab_size, device=DEVICE, dtype=torch.float32)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        temp = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
        seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        cache = torch.zeros(num_reqs, num_steps, vocab_size, device=DEVICE, dtype=torch.float32)
        cols = torch.tensor(0, dtype=torch.int32, device=DEVICE)

        value_out, idx_out = self._run(logits, idx_mapping, temp, seed, pos, cache, cols)

        assert torch.equal(idx_out, logits.argmax(dim=-1))
        assert torch.equal(value_out, logits.max(dim=-1).values)
        assert torch.equal(cache[:, 0], logits)


# --------------------------- Noise streams ---------------------------------

# Adapted from upstream tests/v1/worker/test_gpu_gumbel_sample.py
# (upstream #54282: decouple the draft's Gumbel noise stream from the
# target's via a Philox offset salt).

# Dominant token is exp(HEAD_LOG_GAP)x larger than the unit-count tail, so the
# tail sits ~HEAD_LOG_GAP logits below the top. That tail is the sensitive
# part: the fp32 Gumbel noise must reach ~18 to ever sample it.
HEAD_LOG_GAP = 18.0
# 10-sigma band: a correct sampler effectively never trips it.
Z_TOLERANCE = 10.0
# NPU: upstream draws 500K samples in one call; the kernel's int64 argmax
# scratch is [num_tokens, num_blocks], so draw in chunks to bound memory.
NOISE_STREAM_VOCAB_SIZE = 200_000
NOISE_STREAM_CHUNK = 50_000
NOISE_STREAM_NUM_SAMPLES = 500_000


def _make_heavy_tailed_counts(seed: int = 1234) -> torch.Tensor:
    """Non-negative int64 counts of shape [vocab]; target prob = counts/N."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    counts = torch.randint(1, 4, (NOISE_STREAM_VOCAB_SIZE,), generator=gen, dtype=torch.int64, device=DEVICE)
    counts[0] = round(math.exp(HEAD_LOG_GAP))  # dominant token
    return counts


def _counts_to_logits(counts: torch.Tensor) -> torch.Tensor:
    # softmax(log(count)) == count / sum(count); count 0 -> logit -inf -> prob 0.
    return counts.double().log().to(torch.float32)


def _sample_noise_stream(
    logits_1d: torch.Tensor,
    *,
    is_drafting: bool,
) -> torch.Tensor:
    """Sample NOISE_STREAM_NUM_SAMPLES tokens from one logit vector.

    Fixed seed with a distinct `pos` per sample gives independent draws; the
    logits are broadcast with a 0-stride view to avoid materializing
    [num_samples, vocab_size]. NPU: pos is int32 (triton-ascend philox), and
    draws are chunked to bound the kernel's scratch memory.
    """
    vocab_size = logits_1d.shape[0]
    sampled_parts = []
    for start in range(0, NOISE_STREAM_NUM_SAMPLES, NOISE_STREAM_CHUNK):
        size = min(NOISE_STREAM_CHUNK, NOISE_STREAM_NUM_SAMPLES - start)
        logits = logits_1d.unsqueeze(0).expand(size, vocab_size)
        idx_mapping = torch.zeros(size, dtype=torch.int32, device=DEVICE)
        temp = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
        seed = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
        pos = torch.arange(start, start + size, dtype=torch.int32, device=DEVICE)
        sampled_parts.append(
            gumbel_sample(
                logits,
                idx_mapping,
                temp,
                seed,
                pos,
                apply_temperature=True,
                is_drafting=is_drafting,
            )
        )
    return torch.cat(sampled_parts)


def _z_score(observed: int, expected: float, num_trials: int) -> float:
    p = expected / num_trials
    return (observed - expected) / math.sqrt(num_trials * p * (1 - p))


class TestGumbelSampleNoiseStreams:
    """The draft's Gumbel noise stream must be disjoint from the target's."""

    @torch.inference_mode()
    def test_drafting_uses_a_separate_noise_stream(self):
        """is_drafting salts the Philox offset: same inputs, different draws.

        The draft proposal and the residual resample after a rejection must be
        independent. They key noise by the same (seed, pos), so only the salt
        keeps them apart -- without it the resample inherits the very noise
        vector that picked the rejected proposal. See
        test_gumbel_drafted_rejection_sample_is_unbiased in
        tests/e2e/nightly/single_node/ops/singlecard_ops/triton/
        test_rejection_sample_v2.py for the distributional consequence.

        Relocating the offset must not distort the draw either, so both streams
        are checked against the target's far-tail mass, which sits
        HEAD_LOG_GAP logits below the head -- the regime where fp32 Gumbel
        precision matters.
        """
        counts = _make_heavy_tailed_counts()
        total = counts.sum().item()
        logits = _counts_to_logits(counts)
        tail_prob = (total - counts[0].item()) / total

        target = _sample_noise_stream(logits, is_drafting=False)
        draft = _sample_noise_stream(logits, is_drafting=True)

        # The head dominates, so the streams agree on most draws by construction.
        # Compare instead which draws leave the head: shared noise makes that
        # identical, independent noise makes them differ on ~2p(1-p) of draws.
        target_tail = target != 0
        draft_tail = draft != 0
        disagree = (target_tail != draft_tail).double().mean().item()
        assert disagree > tail_prob, (
            f"streams leave the head on the same draws ({disagree:.3e} disagreement "
            f"vs tail mass {tail_prob:.3e}); the draft salt is not taking effect"
        )

        # Both streams must still reproduce the target's tail mass.
        for name, tail in (("target", target_tail), ("draft", draft_tail)):
            tail_count = tail.sum().item()
            z = _z_score(tail_count, NOISE_STREAM_NUM_SAMPLES * tail_prob, NOISE_STREAM_NUM_SAMPLES)
            assert abs(z) < Z_TOLERANCE, (
                f"{name} tail mass {tail_count / NOISE_STREAM_NUM_SAMPLES:.3e} != {tail_prob:.3e} (z={z:.2f})"
            )

        # The draft stream is reproducible.
        assert torch.equal(draft, _sample_noise_stream(logits, is_drafting=True))

        torch.npu.synchronize()
