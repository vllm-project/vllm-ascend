# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#
# Unit-level tests for the Triton apply_top_k_top_p op
# (vllm_ascend.ops.triton.apply_top_k_top_p, CANN npu_top_k_top_p semantics
# + optional fused softmax) and its wiring into
# AscendTopKTopPSampler.forward_native
# (additional_config enable_triton_apply_topk_topp).

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.apply_top_k_top_p import (
    apply_top_k_top_p,
    fused_topk_topp_softmax,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def cann_masked(logits, k_t, p_t):
    """CANN golden: torch_npu.npu_top_k_top_p (p dtype must match logits)."""
    B, V = logits.shape
    if k_t is None:
        k_t = torch.full((B,), V, dtype=torch.int32, device=logits.device)
    if p_t is not None:
        p_t = p_t.to(logits.dtype)
    return torch_npu.npu_top_k_top_p(logits, p_t, k_t)


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


# -----------------------------------------------------------------------------
# Kernel-level correctness (vs CANN npu_top_k_top_p)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("B,V", [(4, 1000), (8, 4096), (4, 4097), (8, 32000)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_logits_bit_exact_cann(B, V, dtype):
    """Masked logits are bit-exact vs CANN npu_top_k_top_p, including
    batches with per-request mixed k/p."""
    init_device_properties_triton()
    logits = (torch.randn(B, V, device="npu") * 4).to(dtype)
    k = torch.randint(1, V + 1, (B,), dtype=torch.int32, device="npu")
    k[0] = V  # row with top-k disabled
    p = torch.rand(B, device="npu") * 0.9 + 0.05
    p[1] = 1.0  # row with top-p disabled

    # CANN requires p to match the logits dtype (p is rounded at low
    # precision); the op computes in fp32 internally, so pass the aligned
    # effective p value to keep both inputs identical.
    out = apply_top_k_top_p(logits, k, p.to(dtype).float())
    ref = cann_masked(logits, k, p)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("B,V", [(4, 1000), (8, 4096), (8, 32000), (8, 151936)])
def test_fused_probs_vs_cann_softmax(B, V):
    """Fused-softmax output vs CANN masked logits + torch.softmax(fp32):
    the kept set (zero positions) is bit-exact, probabilities match at the
    ulp level, and each row sums to 1."""
    init_device_properties_triton()
    logits = torch.randn(B, V, device="npu") * 4.0
    k = torch.randint(1, min(V, 2000), (B,), dtype=torch.int32, device="npu")
    p = torch.rand(B, device="npu") * 0.9 + 0.05

    probs = fused_topk_topp_softmax(logits, k, p)
    ref = cann_masked(logits, k, p).softmax(dim=-1, dtype=torch.float32)

    assert probs.dtype == torch.float32
    assert torch.equal(probs == 0, ref == 0)  # exact kept set
    torch.testing.assert_close(probs, ref, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(probs.sum(-1), torch.ones(B, device="npu"), atol=1e-6, rtol=0)


def test_scalar_and_none_variants():
    """Scalar k/p combined with None (disabled)."""
    init_device_properties_triton()
    B, V = 4, 8192
    logits = torch.randn(B, V, device="npu") * 4.0
    for k, p in [(None, None), (1, None), (50, 0.9), (V, 0.5), (200, 1.0)]:
        probs = fused_topk_topp_softmax(logits, k, p)
        k_t = None if k is None or k >= V else torch.full((B,), k, dtype=torch.int32, device="npu")
        p_t = None if p is None or p >= 1.0 else torch.full((B,), p, device="npu")
        ref = cann_masked(logits, k_t, p_t).softmax(dim=-1, dtype=torch.float32)
        assert torch.equal(probs == 0, ref == 0)
        torch.testing.assert_close(probs, ref, atol=1e-7, rtol=1e-5)


# -----------------------------------------------------------------------------
# Sampler wiring (additional_config enable_triton_apply_topk_topp)
# -----------------------------------------------------------------------------


class _StubAscendConfig:
    """Minimal AscendConfig stand-in for unit tests (no engine required)."""

    enable_reduce_sample = False
    enable_async_exponential = False
    enable_triton_apply_topk_topp = True


class _StubAscendConfigStock(_StubAscendConfig):
    """Same stub with the Triton apply path disabled (stock CANN path)."""

    enable_triton_apply_topk_topp = False


@pytest.fixture
def sampler_module(monkeypatch):
    """vllm_ascend.sample.sampler with the apply_top_k_top_p path enabled."""
    from vllm_ascend.sample import sampler as module

    monkeypatch.setattr(module, "get_ascend_config", lambda: _StubAscendConfig())
    return module


def test_sampler_apply_path_topk1(sampler_module):
    """k=1 keeps only the argmax token, so the sampled token must equal
    the row argmax regardless of the exponential noise."""
    init_device_properties_triton()
    B, V = 8, 1024
    logits = torch.randn(B, V, dtype=torch.float32, device="npu")
    k = torch.ones(B, dtype=torch.int32, device="npu")
    p = None

    sampler = sampler_module.AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
    next_tokens, logits_to_return = sampler.forward_native(logits, {}, k, p)
    torch.npu.synchronize()

    assert logits_to_return is None
    assert torch.equal(next_tokens, logits.argmax(dim=-1))


def test_sampler_apply_path_processed_logits_matches_stock(sampler_module):
    """processed_logits mode returns masked logits, which must be bit-exact
    vs the stock (CANN) path, including per-request mixed k/p."""
    init_device_properties_triton()
    B, V = 4, 8192
    logits = torch.randn(B, V, dtype=torch.float32, device="npu") * 4.0
    k = torch.tensor([50, V, 1, 200], dtype=torch.int32, device="npu")
    p = torch.tensor([0.9, 1.0, 0.5, 0.8], dtype=torch.float32, device="npu")

    sampler = sampler_module.AscendTopKTopPSampler(logprobs_mode="processed_logits")
    _, logits_to_return = sampler.forward_native(logits.clone(), {}, k, p)
    torch.npu.synchronize()

    ref = cann_masked(logits, k, p)
    assert torch.equal(logits_to_return, ref)


def test_sampler_apply_path_mixed_kp_sampling(sampler_module, monkeypatch):
    """raw mode + per-request mixed k/p: sampled tokens always fall inside
    the kept set (zero-probability positions are never sampled), and the
    result matches the stock path under the same RNG seed."""
    init_device_properties_triton()
    B, V = 4, 32000
    logits = torch.randn(B, V, dtype=torch.float32, device="npu") * 4.0
    k = torch.tensor([50, V, 1, 200], dtype=torch.int32, device="npu")
    p = torch.tensor([0.9, 1.0, 0.5, 0.8], dtype=torch.float32, device="npu")

    sampler = sampler_module.AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
    torch.manual_seed(7)
    next_tokens, _ = sampler.forward_native(logits.clone(), {}, k, p)
    torch.npu.synchronize()

    # The kept set is the finite positions of the CANN masked logits.
    ref_masked = cann_masked(logits, k, p)
    kept = torch.isfinite(ref_masked)
    assert kept.gather(1, next_tokens.unsqueeze(1)).all()

    # The stock path with the same seed must give the same sampling result.
    monkeypatch.setattr(sampler_module, "get_ascend_config", lambda: _StubAscendConfigStock())
    stock_sampler = sampler_module.AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
    torch.manual_seed(7)
    stock_tokens, _ = stock_sampler.forward_native(logits.clone(), {}, k, p)
    torch.npu.synchronize()
    assert torch.equal(next_tokens, stock_tokens)


def test_sampler_stock_still_works(monkeypatch):
    """Regression: the stock path is unaffected when the switch is off
    (default)."""
    init_device_properties_triton()
    from vllm_ascend.sample import sampler as module

    monkeypatch.setattr(module, "get_ascend_config", lambda: _StubAscendConfigStock())

    B, V = 8, 1024
    logits = torch.randn(B, V, dtype=torch.float32, device="npu")
    k = torch.ones(B, dtype=torch.int32, device="npu")

    stock = module.AscendTopKTopPSampler(logprobs_mode="raw_logprobs")
    next_tokens, _ = stock.forward_native(logits.clone(), {}, k, None)
    assert torch.equal(next_tokens, logits.argmax(dim=-1))
