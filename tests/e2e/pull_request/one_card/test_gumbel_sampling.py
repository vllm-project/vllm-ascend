# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Tests for vllm_ascend.worker.v2.sample.gumbel on Ascend NPU.
# Validates gumbel_sample and apply_temperature against PyTorch references.

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.worker.v2.sample.gumbel import apply_temperature, gumbel_sample
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

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

        sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
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

        s_false = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
        s_true = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)
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

        r1 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
        torch.npu.synchronize()
        r2 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
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

        r1 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed1, pos, apply_temperature=False)
        r2 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed2, pos, apply_temperature=False)
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

        sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
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
                logits_base.clone(), expanded_idx_mapping, low_temp, seed, pos, apply_temperature=True
            )
            s_high = gumbel_sample(
                logits_base.clone(), expanded_idx_mapping, high_temp, seed, pos, apply_temperature=True
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

        sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
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

        sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
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

        sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)
        torch.npu.synchronize()

        assert sampled[0].item() == sampled[1].item(), (
            f"Tokens with same logits, seed, and pos should sample the same token: "
            f"got {sampled[0].item()} vs {sampled[1].item()}"
        )

    def test_gumbel_sample_apply_temperature_true_nonzero(self):
        """apply_temperature=True with temp>0 must divide logits by temperature
        before adding Gumbel noise, while caching the unscaled logits."""
        torch.manual_seed(77)
        num_tokens, num_reqs, vocab_size = 4, 4, 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        # Cache writes must not change the temperature-scaled samples.
        out_logits = torch.zeros(num_reqs, 1, vocab_size, dtype=torch.float32, device=DEVICE)
        sampled = gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            logits_cache=out_logits,
        )
        expected_sampled = gumbel_sample(
            logits / temperature[:, None],
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=False,
        )
        torch.npu.synchronize()
        assert torch.equal(sampled, expected_sampled)

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            temp = temperature[req].item()
            expected = logits[tok].float()
            assert torch.allclose(out_logits[req, 0].float(), expected, atol=1e-4, rtol=1e-4), (
                f"logits_cache mismatch at token {tok} (req {req}, temp={temp:.3f}): "
                f"max_diff={(out_logits[req, 0].float() - expected).abs().max().item():.6f}"
            )

    def test_gumbel_sample_apply_temperature_false_nonzero(self):
        """apply_temperature=False with temp>0: logits_cache must contain
        raw logits (no temperature division), but Gumbel noise is still added
        to sampling."""
        torch.manual_seed(78)
        num_tokens, num_reqs, vocab_size = 4, 4, 32000
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
        seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

        out_logits = torch.zeros(num_reqs, 1, vocab_size, dtype=torch.float32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=False,
            logits_cache=out_logits,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            # Without temperature application, stored logits should match raw logits
            expected = logits[tok].float()
            assert torch.allclose(out_logits[req, 0].float(), expected, atol=1e-4, rtol=1e-4), (
                f"logits_cache should be raw logits when apply_temperature=False: "
                f"max_diff={(out_logits[req, 0].float() - expected).abs().max().item():.6f}"
            )

    def test_gumbel_sample_logits_cache_req_state_idx(self):
        """Cached logits must be stored at req_state_idx position, not token_idx.

        This tests the EAGLE speculative decoding scenario where the idx_mapping
        is non-contiguous (e.g., active requests [2,5,7,0] out of 8 slots).
        The buffer is shaped [max_num_reqs, 1, vocab_size] and the kernel must store
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

        out_logits = torch.zeros(max_num_reqs, 1, vocab_size, dtype=torch.float32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            logits_cache=out_logits,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            temp = temperature[req].item()
            expected = logits[tok].float()
            actual = out_logits[req, 0]
            assert torch.allclose(actual.float(), expected, atol=1e-4, rtol=1e-4), (
                f"Req {req} (tok={tok}, temp={temp:.3f}): max_diff={(actual.float() - expected).abs().max().item():.6f}"
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
            logits_cache=draft_logits,
            logits_cache_col=col_tensor,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
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
        """Cached logits must remain unscaled for both greedy and random requests.

        Note: In practice, logits_cache is only used by EAGLE
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

        out_logits = torch.zeros(num_reqs, 1, vocab_size, dtype=torch.float32, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            logits_cache=out_logits,
        )
        torch.npu.synchronize()

        for tok in range(num_tokens):
            req = expanded_idx_mapping[tok].item()
            temp = temperature[req].item()
            expected = logits[tok].float()
            actual = out_logits[req, 0]
            assert torch.allclose(actual.float(), expected, atol=1e-4, rtol=1e-4), (
                f"Req {req} (tok={tok}, temp={temp:.3f}): max_diff={(actual.float() - expected).abs().max().item():.6f}"
            )

    @pytest.mark.parametrize("per_token_col", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gumbel_sample_strided_logits_cache(self, per_token_col, dtype):
        """Cache raw logits at mapped request/step slots, preserving padding."""
        torch.manual_seed(202)
        vocab_size = 1031
        logits = torch.randn(4, vocab_size, dtype=dtype, device=DEVICE)
        mapping = torch.tensor([2, 0, 2, -1], dtype=torch.int32, device=DEVICE)
        if not per_token_col:
            mapping[2] = 1
        # Exercise non-contiguous index inputs as used by expanded batches.
        mapping = torch.stack((mapping, mapping), dim=1)[:, 0]
        pos = torch.arange(8, dtype=torch.int32, device=DEVICE)[::2]
        temperature = torch.tensor([0.0, 0.5, 1.5], device=DEVICE)
        seed = torch.arange(3, dtype=torch.int64, device=DEVICE)
        cols = torch.tensor([0, 0, 2, 0, 1, 0, 2, 0], dtype=torch.int32, device=DEVICE)[::2]
        if not per_token_col:
            cols = torch.tensor(1, dtype=torch.int32, device=DEVICE)
        storage = torch.full((3, 6, vocab_size + 17), 42, dtype=dtype, device=DEVICE)
        cache = storage[:, ::2, :]
        expected = storage.clone()
        for token, req in enumerate(mapping.cpu().tolist()):
            if req >= 0:
                col = cols[token].item() if per_token_col else cols.item()
                expected[req, 2 * col, :vocab_size] = logits[token]

        gumbel_sample(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=cache,
            logits_cache_col=cols,
        )
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)

    def test_gumbel_sample_draft_noise(self):
        """Draft sampling uses the upstream position salt for an independent stream."""
        logits = torch.zeros(16, 1031, device=DEVICE)
        mapping = torch.arange(16, dtype=torch.int32, device=DEVICE)
        temperature = torch.ones(16, device=DEVICE)
        seed = torch.arange(16, dtype=torch.int64, device=DEVICE)
        pos = torch.arange(16, dtype=torch.int32, device=DEVICE)
        draft_noise_salt = 1 << 30
        draft = gumbel_sample(logits, mapping, temperature, seed, pos, True, is_drafting=True)
        shifted = gumbel_sample(logits, mapping, temperature, seed, pos + draft_noise_salt, True, is_drafting=False)
        target = gumbel_sample(logits, mapping, temperature, seed, pos, True)
        assert torch.equal(draft, shifted)
        assert not torch.equal(draft, target)

    def test_gumbel_sample_rejects_narrow_cache(self):
        logits = torch.zeros(1, 32, device=DEVICE)
        mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        with pytest.raises(AssertionError, match="is narrower"):
            gumbel_sample(
                logits,
                mapping,
                torch.ones(1, device=DEVICE),
                mapping,
                mapping,
                True,
                logits_cache=torch.empty(1, 1, 31, device=DEVICE),
            )

    def test_dspark_uses_ascend_gumbel(self):
        """Exercise the inherited DSpark entry point with real NPU sampling."""
        assert DSparkSpeculator._sample_logits.__globals__["gumbel_sample"] is gumbel_sample
        speculator = DSparkSpeculator.__new__(DSparkSpeculator)
        speculator._d2t_scatter_index = None
        speculator.temperature = torch.tensor([0.5, 1.5], device=DEVICE)
        speculator.seeds = torch.tensor([3, 7], dtype=torch.int64, device=DEVICE)
        speculator._step_cols = torch.arange(2, dtype=torch.int32, device=DEVICE)
        speculator.draft_logits = torch.zeros(2, 2, 1031, device=DEVICE)
        speculator.use_fp64_gumbel = False
        logits = torch.randn(2, 1031, device=DEVICE)
        idx_mapping = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
        sample_pos = torch.tensor([8, 12], dtype=torch.int32, device=DEVICE)

        sampled = speculator._sample_logits(logits, idx_mapping, sample_pos, step=1)

        assert sampled.shape == (2,)
        assert ((sampled >= 0) & (sampled < logits.shape[-1])).all()
        torch.testing.assert_close(speculator.draft_logits[idx_mapping.long(), 1], logits, rtol=0, atol=0)
        assert (speculator.draft_logits[:, 0] == 0).all()

    @pytest.mark.parametrize("temp", [0.5, 2.0])
    @pytest.mark.parametrize("vocab_size", [31, 1031])
    def test_cached_draft_rejection_temperature(self, temp, vocab_size):
        """Raw cached drafts and pre-scaled drafts must verify/resample identically."""
        torch.manual_seed(203)
        num_reqs, num_steps = 32, 2
        mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        temperature = torch.full((num_reqs,), temp, device=DEVICE)
        seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
        draft_logits = torch.randn(num_reqs, vocab_size, device=DEVICE)
        cache = torch.empty(num_reqs, num_steps, vocab_size, device=DEVICE)
        draft_tokens = torch.zeros(num_reqs, num_steps + 1, dtype=torch.int64, device=DEVICE)
        for step in range(num_steps):
            draft_tokens[:, step + 1] = gumbel_sample(
                draft_logits,
                mapping,
                temperature,
                seed,
                mapping + step,
                True,
                is_drafting=True,
                logits_cache=cache,
                logits_cache_col=torch.tensor(step, dtype=torch.int32, device=DEVICE),
            )
        inputs = dict(
            target_logits=torch.randn(num_reqs * (num_steps + 1), vocab_size, device=DEVICE),
            draft_sampled=draft_tokens.flatten(),
            cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE) * (num_steps + 1),
            pos=torch.arange(num_reqs * (num_steps + 1), dtype=torch.int32, device=DEVICE),
            idx_mapping=mapping,
            expanded_idx_mapping=mapping.repeat_interleave(num_steps + 1),
            expanded_local_pos=torch.arange(num_steps + 1, dtype=torch.int32, device=DEVICE).repeat(num_reqs),
            seed=seed,
            num_speculative_steps=num_steps,
        )
        sampled, counts = rejection_sample(**inputs, draft_logits=cache, temperature=temperature)
        expected, expected_counts = rejection_sample(
            **inputs, draft_logits=cache / temp, temperature=torch.ones_like(temperature)
        )
        assert torch.equal(counts, expected_counts)
        # Output slots after the first rejection are unspecified.
        valid = torch.arange(num_steps + 1, device=DEVICE)[None, :] < counts[:, None]
        assert torch.equal(sampled[valid], expected[valid])
        assert (counts < num_steps + 1).any(), "Exercise the residual resampling path"

    def test_gumbel_sample_single_token(self):
        """Single token with temperature > 0 should work."""
        torch.manual_seed(42)
        logits = torch.randn(1, 32000, dtype=torch.float32, device=DEVICE)
        expanded_idx_mapping = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        temperature = torch.tensor([0.7], dtype=torch.float32, device=DEVICE)
        seed = torch.tensor([12345], dtype=torch.int64, device=DEVICE)
        pos = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)
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

        sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
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
        s1 = gumbel_sample(logits, expanded_idx_mapping, low_temp, seed, pos, apply_temperature=True)
        torch.npu.synchronize()
        assert (s1 >= 0).all() and (s1 < vocab_size).all()

        # Very high temperature (near-uniform)
        high_temp = torch.tensor([100.0, 100.0, 100.0, 100.0], dtype=torch.float32, device=DEVICE)
        s2 = gumbel_sample(logits, expanded_idx_mapping, high_temp, seed, pos, apply_temperature=True)
        torch.npu.synchronize()
        assert (s2 >= 0).all() and (s2 < vocab_size).all()
