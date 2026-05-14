# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu",
                                reason="torch_npu not available, skipping NPU tests")

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="No NPU device available",
)

VOCAB_SIZES = [32000, 128256, 151936]


@pytest.mark.parametrize("num_tokens,vocab_size", [
    (1, 32000),
    (48, 32000),
    (96, 128256),
    (24, 151936),
])
def test_gumbel_sample_greedy(num_tokens, vocab_size):
    """temperature=0 must return the argmax of the raw logits (greedy)."""
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.zeros(num_tokens, dtype=torch.float32, device="npu")
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed,
                            pos, apply_temperature=False)
    torch.npu.synchronize()

    expected = logits.argmax(dim=-1).cpu()
    assert torch.equal(sampled.cpu(), expected), (
        f"Greedy mismatch: got {sampled.cpu()}, expected {expected}")


@pytest.mark.parametrize("num_tokens,vocab_size", [
    (1, 32000),
    (48, 128256),
])
def test_gumbel_sample_deterministic(num_tokens, vocab_size):
    """Same seed must produce the same sample (determinism)."""
    torch.manual_seed(42)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.ones(num_tokens, dtype=torch.float32, device="npu") * 0.8
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    s1 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                       apply_temperature=True)
    s2 = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                       apply_temperature=True)
    torch.npu.synchronize()
    assert torch.equal(s1.cpu(), s2.cpu()), "Same seed must yield identical samples"


@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_gumbel_sample_valid_token_id(vocab_size):
    """Sampled token ids must be in [0, vocab_size)."""
    num_tokens = 32
    torch.manual_seed(7)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.rand(num_tokens, device="npu") * 1.5 + 0.1
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed,
                            pos, apply_temperature=True)
    torch.npu.synchronize()
    assert sampled.shape == (num_tokens,)
    assert (sampled >= 0).all() and (sampled < vocab_size).all(), (
        f"Out-of-range token ids: min={sampled.min()}, max={sampled.max()}")


@pytest.mark.parametrize("num_tokens,vocab_size", [
    (64, 32000),
    (32, 151936),
])
def test_gumbel_sample_apply_temperature_flag(num_tokens, vocab_size):
    """apply_temperature=True must scale logits; verify via processed_logits_out."""
    torch.manual_seed(3)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.full((num_tokens,), 2.0, dtype=torch.float32, device="npu")
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    out_with = torch.zeros(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    out_without = torch.zeros(num_tokens, vocab_size, device="npu", dtype=torch.float32)

    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                  apply_temperature=True, output_processed_logits=out_with)
    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                  apply_temperature=False, output_processed_logits=out_without)
    torch.npu.synchronize()

    # apply_temperature=True: processed logits must equal logits / temp
    expected_scaled = logits.cpu().float() / 2.0
    assert torch.allclose(out_with.cpu(), expected_scaled, atol=1e-4, rtol=1e-4), \
        "apply_temperature=True: processed_logits must be logits/temp"

    # apply_temperature=False: processed logits must equal raw logits
    assert torch.allclose(out_without.cpu(), logits.cpu(), atol=1e-5), \
        "apply_temperature=False: processed_logits must equal raw logits"


@pytest.mark.parametrize("vocab_size", [32000, 128256])
def test_gumbel_sample_mixed_temperature(vocab_size):
    """Batch with mixed temp=0 and temp>0: greedy tokens must match argmax."""
    num_tokens = 16
    torch.manual_seed(99)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.rand(num_tokens, device="npu") * 1.5 + 0.5
    temperature[::2] = 0.0  # even indices are greedy
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed,
                            pos, apply_temperature=True)
    torch.npu.synchronize()

    greedy_expected = logits.argmax(dim=-1).cpu()
    for i in range(0, num_tokens, 2):
        assert sampled[i].item() == greedy_expected[i].item(), (
            f"req {i}: greedy expected {greedy_expected[i]}, got {sampled[i]}")


@pytest.mark.parametrize("vocab_size", [32000])
def test_gumbel_sample_distribution(vocab_size):
    """High-temperature sampling should spread across tokens (not always argmax)."""
    num_tokens = 512
    torch.manual_seed(5)
    logits = torch.zeros(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.ones(num_tokens, dtype=torch.float32, device="npu")
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed,
                            pos, apply_temperature=False)
    torch.npu.synchronize()

    unique = sampled.unique().numel()
    assert unique > num_tokens // 4, (
        f"Too few unique tokens ({unique}), sampling may be degenerate")


@pytest.mark.parametrize("num_tokens,vocab_size", [
    (8, 32000),
    (32, 128256),
])
def test_gumbel_sample_processed_logits_with_temperature(num_tokens, vocab_size):
    """output_processed_logits must contain logits divided by temperature."""
    torch.manual_seed(11)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.rand(num_tokens, device="npu") * 1.5 + 0.5
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    output_processed_logits = torch.zeros(num_tokens, vocab_size,
                                          device="npu", dtype=torch.float32)
    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                  apply_temperature=True,
                  output_processed_logits=output_processed_logits)
    torch.npu.synchronize()

    for i in range(num_tokens):
        expected = logits[i].float() / temperature[i].item()
        actual = output_processed_logits[i].float()
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4), (
            f"req {i}: processed_logits mismatch, "
            f"max_diff={torch.max(torch.abs(actual - expected)).item():.6f}")


@pytest.mark.parametrize("num_tokens,vocab_size", [
    (8, 32000),
])
def test_gumbel_sample_processed_logits_no_temperature(num_tokens, vocab_size):
    """With apply_temperature=False, output_processed_logits must contain raw logits."""
    torch.manual_seed(0)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.rand(num_tokens, device="npu") * 1.5 + 0.5
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    output_processed_logits = torch.zeros(num_tokens, vocab_size,
                                          device="npu", dtype=torch.float32)
    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                  apply_temperature=False,
                  output_processed_logits=output_processed_logits)
    torch.npu.synchronize()

    assert torch.allclose(output_processed_logits, logits, atol=1e-5), (
        f"apply_temperature=False: output_processed_logits should equal raw logits, "
        f"max_diff={torch.max(torch.abs(output_processed_logits - logits)).item():.6f}")


def test_gumbel_sample_expanded_idx_mapping():
    """Multiple tokens mapped to the same request via expanded_idx_mapping."""
    torch.manual_seed(0)
    vocab_size = 32000
    num_tokens, num_reqs = 5, 3
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    # req0->tok0,tok1  req1->tok2,tok3  req2->tok4
    expanded_idx_mapping = torch.tensor([0, 0, 1, 1, 2],
                                        dtype=torch.int32, device="npu")
    temperature = torch.tensor([0.5, 1.5, 0.0], device="npu", dtype=torch.float32)
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    out_logits = torch.zeros(num_reqs, vocab_size, device="npu", dtype=torch.float32)
    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed,
                            pos, apply_temperature=True,
                            output_processed_logits=out_logits)
    torch.npu.synchronize()

    assert sampled.shape == (num_tokens,)
    assert (sampled >= 0).all() and (sampled < vocab_size).all()

    # tok4 belongs to req2 with temp=0 -> must be greedy argmax
    assert sampled[4].item() == logits[4].argmax().item(), (
        f"greedy token mismatch: expected {logits[4].argmax().item()}, "
        f"got {sampled[4].item()}")

    # req0 (temp=0.5): out_logits[0] must equal logits of last writer (tok1) / 0.5
    # req1 (temp=1.5): out_logits[1] must equal logits of last writer (tok3) / 1.5
    # req2 (temp=0.0): out_logits[2] must equal raw logits of tok4
    for req_idx, tok_idx, temp in [(0, 1, 0.5), (1, 3, 1.5), (2, 4, 0.0)]:
        if temp == 0.0:
            expected = logits[tok_idx].float()
        else:
            expected = logits[tok_idx].float() / temp
        assert torch.allclose(out_logits[req_idx].float(), expected,
                               atol=1e-4, rtol=1e-4), (
            f"req {req_idx}: out_logits mismatch, "
            f"max_diff={torch.max(torch.abs(out_logits[req_idx].float() - expected)).item():.6f}")


@pytest.mark.parametrize("num_tokens,vocab_size", [
    (8, 32000),
    (16, 128256),
])
def test_gumbel_sample_processed_logits_col(num_tokens, vocab_size):
    """output_processed_logits_col must write to the correct column slice."""
    torch.manual_seed(77)
    logits = torch.randn(num_tokens, vocab_size, device="npu", dtype=torch.float32)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device="npu")
    temperature = torch.rand(num_tokens, device="npu") * 1.5 + 0.5
    seed = torch.randint(0, 2**31, (num_tokens,), dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int64, device="npu")

    # 2-column output buffer: [num_tokens, 2, vocab_size] flattened as [num_tokens, 2*vocab_size]
    num_cols = 2
    output_buf = torch.zeros(num_tokens, num_cols * vocab_size,
                             device="npu", dtype=torch.float32)
    col_idx = torch.tensor(1, dtype=torch.int32, device="npu")  # write to col=1

    gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos,
                  apply_temperature=True,
                  output_processed_logits=output_buf,
                  output_processed_logits_col=col_idx)
    torch.npu.synchronize()

    # col=0 must be all zeros (untouched)
    col0 = output_buf[:, :vocab_size]
    assert torch.all(col0 == 0.0), "col=0 must be untouched"

    # col=1 must equal logits / temperature
    col1 = output_buf[:, vocab_size:]
    for i in range(num_tokens):
        expected = logits[i].float() / temperature[i].item()
        assert torch.allclose(col1[i].float(), expected, atol=1e-4, rtol=1e-4), (
            f"req {i}: col=1 mismatch, "
            f"max_diff={torch.max(torch.abs(col1[i].float() - expected)).item():.6f}")
