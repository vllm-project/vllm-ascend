# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Tests for vllm_ascend.worker.v2.sample.gumbel on Ascend NPU.
# Validates gumbel_sample and apply_temperature against PyTorch references.

import pytest
import torch

from vllm_ascend.worker.v2.sample.gumbel import apply_temperature, gumbel_sample

DEVICE = "npu"


def _ref_gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
) -> torch.Tensor:
    """Pure-Python reference: greedy when temp==0, else argmax(logits/T + gumbel)."""
    num_tokens, vocab_size = logits.shape
    result = torch.empty(num_tokens, dtype=torch.int64, device=logits.device)
    for tok in range(num_tokens):
        req = expanded_idx_mapping[tok].item()
        temp = temperature[req].item()
        l = logits[tok].float().clone()
        if temp == 0.0:
            result[tok] = l.argmax()
        else:
            if apply_temperature:
                l = l / temp
            result[tok] = l.argmax()
    return result


def _ref_apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    out = logits.clone().float()
    for tok in range(logits.shape[0]):
        req = expanded_idx_mapping[tok].item()
        temp = temperature[req].item()
        if temp == 0.0 or temp == 1.0:
            continue
        out[tok] = out[tok] / temp
    return out


@pytest.mark.parametrize("num_tokens,vocab_size", [
    (1, 32000),
    (8, 32000),
    (48, 102400),
    (64, 151936),
])
def test_apply_temperature(num_tokens, vocab_size):
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.randint(
        0, num_tokens, (num_tokens,), dtype=torch.int32, device=DEVICE
    )
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
        f"apply_temperature mismatch: max_diff="
        f"{(logits_triton.float() - logits_ref).abs().max().item():.6f}"
    )


@pytest.mark.parametrize("num_tokens,num_reqs,vocab_size", [
    (4, 4, 32000),
    (8, 4, 32000),   # expanded: multiple tokens per request
    (16, 8, 102400),
    (1, 1, 32000),
])
def test_gumbel_sample_greedy(num_tokens, num_reqs, vocab_size):
    """temperature=0 must return argmax (greedy)."""
    torch.manual_seed(42)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE
    )
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                            apply_temperature=False)
    torch.npu.synchronize()

    expected = logits.argmax(dim=-1)
    assert torch.equal(sampled, expected), (
        f"Greedy mismatch: sampled={sampled.tolist()} expected={expected.tolist()}"
    )


@pytest.mark.parametrize("num_tokens,num_reqs,vocab_size", [
    (4, 4, 32000),
    (8, 4, 32000),
    (16, 8, 102400),
])
def test_gumbel_sample_deterministic(num_tokens, num_reqs, vocab_size):
    """Same seed must produce identical results across runs."""
    torch.manual_seed(7)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE
    )
    temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    r1 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                       apply_temperature=False)
    torch.npu.synchronize()
    r2 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                       apply_temperature=False)
    torch.npu.synchronize()

    assert torch.equal(r1, r2), "gumbel_sample is non-deterministic with same seed"


@pytest.mark.parametrize("num_tokens,num_reqs,vocab_size", [
    (4, 4, 32000),
    (8, 4, 32000),
    (16, 8, 102400),
])
def test_gumbel_sample_valid_token_ids(num_tokens, num_reqs, vocab_size):
    """Sampled token IDs must be in [0, vocab_size)."""
    torch.manual_seed(3)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE
    )
    temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) + 0.1
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                            apply_temperature=False)
    torch.npu.synchronize()

    assert sampled.shape == (num_tokens,)
    assert (sampled >= 0).all() and (sampled < vocab_size).all(), (
        f"Out-of-range token IDs: min={sampled.min()}, max={sampled.max()}"
    )


@pytest.mark.parametrize("num_tokens,num_reqs,vocab_size", [
    (4, 4, 32000),
    (8, 4, 32000),
])
def test_gumbel_sample_mixed_temperature(num_tokens, num_reqs, vocab_size):
    """Mix of temp=0 and temp>0: temp=0 tokens must be greedy."""
    torch.manual_seed(11)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    # identity mapping: token i -> request i (for simplicity)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.rand(num_tokens, dtype=torch.float32, device=DEVICE) + 0.5
    # force first half to greedy
    temperature[:num_tokens // 2] = 0.0
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                            apply_temperature=False)
    torch.npu.synchronize()

    greedy = logits.argmax(dim=-1)
    for tok in range(num_tokens // 2):
        assert sampled[tok].item() == greedy[tok].item(), (
            f"Token {tok} (temp=0) should be greedy: "
            f"got {sampled[tok].item()}, expected {greedy[tok].item()}"
        )


def test_gumbel_sample_expanded_idx_mapping():
    """Multiple tokens mapping to the same request must work correctly."""
    torch.manual_seed(99)
    num_tokens = 6
    num_reqs = 2
    vocab_size = 32000

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    # tokens 0,1,2 -> req 0; tokens 3,4,5 -> req 1
    expanded_idx_mapping = torch.tensor(
        [0, 0, 0, 1, 1, 1], dtype=torch.int32, device=DEVICE
    )
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                            apply_temperature=False)
    torch.npu.synchronize()

    expected = logits.argmax(dim=-1)
    assert torch.equal(sampled, expected), (
        f"Expanded mapping greedy mismatch: {sampled.tolist()} vs {expected.tolist()}"
    )


@pytest.mark.parametrize("num_tokens,num_reqs,vocab_size", [
    (4, 4, 32000),
    (8, 4, 102400),
])
def test_gumbel_sample_apply_temperature_flag(num_tokens, num_reqs, vocab_size):
    """apply_temperature=True must divide logits by temperature before sampling."""
    torch.manual_seed(55)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE)
    # all greedy so we can verify deterministically
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    # With temp=0, apply_temperature flag has no effect — both must be greedy
    s_false = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                            apply_temperature=False)
    s_true = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                           apply_temperature=True)
    torch.npu.synchronize()

    expected = logits.argmax(dim=-1)
    assert torch.equal(s_false, expected)
    assert torch.equal(s_true, expected)


@pytest.mark.parametrize("num_tokens,num_reqs,vocab_size", [
    (4, 4, 32000),   # num_tokens == num_reqs
    (8, 4, 32000),   # num_tokens > num_reqs (expanded)
])
def test_gumbel_sample_output_processed_logits(num_tokens, num_reqs, vocab_size):
    """output_processed_logits must contain logits/temperature per token."""
    torch.manual_seed(77)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE
    )
    temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    out_logits = torch.zeros(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                  apply_temperature=True, output_processed_logits=out_logits)
    torch.npu.synchronize()

    for tok in range(num_tokens):
        req = expanded_idx_mapping[tok].item()
        temp = temperature[req].item()
        expected = logits[tok].float() / temp
        assert torch.allclose(out_logits[tok].float(), expected, atol=1e-4, rtol=1e-4), (
            f"processed_logits mismatch at token {tok} (req {req}): "
            f"max_diff={(out_logits[tok].float() - expected).abs().max().item():.6f}"
        )


@pytest.mark.parametrize("num_tokens,num_reqs,vocab_size", [
    (6, 2, 32000),
    (8, 4, 32000),
])
def test_gumbel_sample_output_processed_logits_expanded(num_tokens, num_reqs, vocab_size):
    """output_processed_logits with expanded_idx_mapping: per-token, no race condition.

    Covers mixed temperature (temp=0 and temp>0) per the reviewer suggestion:
    - temp=0: processed logits must equal original logits (no scaling)
    - temp>0: processed logits must equal logits / temp
    Buffer is sized [num_tokens, vocab_size] to match per-token kernel indexing.
    """
    torch.manual_seed(88)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    # multiple tokens per request
    expanded_idx_mapping = torch.randint(
        0, num_reqs, (num_tokens,), dtype=torch.int32, device=DEVICE
    )
    temperature = torch.rand(num_reqs, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
    # force first request to temp=0 to cover the greedy branch
    temperature[0] = 0.0
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    # Buffer must be [num_tokens, vocab_size], not [num_reqs, vocab_size],
    # to avoid out-of-bounds writes when num_tokens > num_reqs.
    out_logits = torch.zeros(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                  apply_temperature=True, output_processed_logits=out_logits)
    torch.npu.synchronize()

    for tok_idx in range(num_tokens):
        req_idx = expanded_idx_mapping[tok_idx].item()
        temp = temperature[req_idx].item()
        if temp == 0.0:
            expected = logits[tok_idx].float()
        else:
            expected = logits[tok_idx].float() / temp
        assert torch.allclose(out_logits[tok_idx].float(), expected, atol=1e-4, rtol=1e-4), (
            f"tok {tok_idx} (req {req_idx}, temp={temp:.3f}): out_logits mismatch, "
            f"max_diff={torch.max(torch.abs(out_logits[tok_idx].float() - expected)).item():.6f}"
        )
