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
# NPU tests for the Triton apply_top_k_top_p op
# (vllm_ascend.ops.triton.sample.apply_top_k_top_p).
#
# Two semantic modes are covered:
#   - top_p_mode="topk_renorm": bit-exact vs CANN npu_top_k_top_p
#   - top_p_mode="full_vocab": bit-exact vs the legacy 0.29/0.32 reference
#     (full-vocabulary top-p cutoff), ties kept

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.sample.apply_top_k_top_p import apply_top_k_top_p
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


@pytest.fixture
def pytorch_baseline():
    """Legacy 0.29/0.32 full-vocab reference (pre-top-k softmax)."""

    def _ref(logits, k, p):
        if p is None and k is None:
            return logits
        probs = logits.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)
        if k is not None:
            top_k_count = probs_sort.size(1) - k.to(torch.long)
            top_k_cutoff = probs_sort.gather(-1, top_k_count.unsqueeze(dim=1))
            no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))
            logits.masked_fill_(probs < top_k_cutoff, -float("inf"))
        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False
            top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            logits.masked_fill_(probs < top_p_cutoff, -float("inf"))
        return logits

    return _ref


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


# -----------------------------------------------------------------------------
# topk_renorm mode: bit-exact vs CANN npu_top_k_top_p
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("B,V", [(4, 1000), (8, 4096), (4, 4097), (8, 32000)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_logits_bit_exact_cann(B, V, dtype):
    """topk_renorm masked logits are bit-exact vs CANN npu_top_k_top_p,
    including batches with per-request mixed k/p and the unified tiled
    kernel at every V (small-row adaptive block + multi-tile path)."""
    init_device_properties_triton()
    logits = (torch.randn(B, V, device="npu") * 4).to(dtype)
    k = torch.randint(1, V + 1, (B,), dtype=torch.int32, device="npu")
    k[0] = V  # row with top-k disabled
    p = torch.rand(B, device="npu") * 0.9 + 0.05
    p[1] = 1.0  # row with top-p disabled

    # CANN requires p to match the logits dtype (p is rounded at low
    # precision); the op computes in fp32 internally, so pass the aligned
    # effective p value to keep both inputs identical.
    out = apply_top_k_top_p(logits, k, p.to(dtype).float(), top_p_mode="topk_renorm")
    ref = cann_masked(logits, k, p)
    assert torch.equal(out, ref)


def test_scalar_and_none_variants():
    """Scalar k/p combined with None (disabled), topk_renorm vs CANN."""
    init_device_properties_triton()
    B, V = 4, 8192
    logits = torch.randn(B, V, device="npu") * 4.0
    for k, p in [(None, None), (1, None), (50, 0.9), (V, 0.5), (200, 1.0)]:
        out = apply_top_k_top_p(logits.clone(), k, p, top_p_mode="topk_renorm")
        k_t = None if k is None or k >= V else torch.full((B,), k, dtype=torch.int32, device="npu")
        p_t = None if p is None or p >= 1.0 else torch.full((B,), p, dtype=torch.float32, device="npu")
        ref = cann_masked(logits, k_t, p_t)
        assert torch.equal(out, ref)


# -----------------------------------------------------------------------------
# full_vocab mode (default): equivalence to the current default PyTorch path
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("B,V", [(1, 32000), (8, 32000), (8, 151936), (32, 4097)])
def test_full_vocab_bit_exact_pytorch_fp32(B, V, pytorch_baseline):
    """Default mode must be a drop-in of _apply_top_k_top_p_pytorch:
    fp32 bit-exact with k and p both enabled (AE-3 regression)."""
    init_device_properties_triton()
    logits = torch.randn(B, V, dtype=torch.float32, device="npu") * 4.0
    k = torch.randint(1, 200, (B,), dtype=torch.int32, device="npu")
    p = torch.rand(B, device="npu") * 0.09 + 0.90

    out = apply_top_k_top_p(logits.clone(), k, p)
    ref = pytorch_baseline(logits.clone(), k, p)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bucket", [0.5, 1.0, 2.0])
def test_full_vocab_ties_kept(dtype, bucket, pytorch_baseline):
    """Repeated logit values at the cutoff must all survive (value
    comparison, not a positional suffix): bf16 quantization creates ties."""
    init_device_properties_triton()
    B, V = 4, 32000
    logits = ((torch.randn(B, V) * 4.0 / bucket).round() * bucket).to(dtype).npu()
    k = torch.full((B,), 20, dtype=torch.int32, device="npu")
    p = torch.full((B,), 0.95, dtype=torch.float32, device="npu")

    out = apply_top_k_top_p(logits.clone(), k, p)
    ref = pytorch_baseline(logits.clone(), k, p)
    if dtype == torch.float32:
        assert torch.equal(out, ref)
    else:
        diff = (out != float("-inf")).sum(-1) - (ref != float("-inf")).sum(-1)
        assert diff.abs().max() <= 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_full_vocab_low_precision_kept_tolerance(dtype, pytorch_baseline):
    """bf16/fp16: kept count may differ by <=3 tokens (low-precision
    softmax/cumsum rounding in the PyTorch baseline)."""
    init_device_properties_triton()
    B, V = 8, 32000
    logits = (torch.randn(B, V) * 4.0).to(dtype).npu()
    k = torch.randint(1, 200, (B,), dtype=torch.int32, device="npu")
    p = torch.rand(B, device="npu") * 0.09 + 0.90

    out = apply_top_k_top_p(logits.clone(), k, p)
    ref = pytorch_baseline(logits.clone(), k, p)
    diff = (out != float("-inf")).sum(-1) - (ref != float("-inf")).sum(-1)
    assert diff.abs().max() <= 3


# -----------------------------------------------------------------------------
# Degenerate rows (all -inf / many -inf / NaN)
# -----------------------------------------------------------------------------


def test_degenerate_rows():
    init_device_properties_triton()
    B, V = 4, 32000
    logits = torch.randn(B, V, dtype=torch.float32, device="npu")
    logits[0] = float("-inf")
    logits[1, :1000] = float("-inf")
    logits[2, 0] = float("nan")
    k = torch.full((B,), 20, dtype=torch.int32, device="npu")
    p = torch.full((B,), 0.95, dtype=torch.float32, device="npu")

    out = apply_top_k_top_p(logits, k, p)

    # all -inf row: masked logits stay all -inf
    assert torch.isinf(out[0]).all()
    # many -inf / normal rows: finite behavior
    assert not torch.isnan(out[1]).any()
    assert torch.isnan(out[2]).any()  # NaN input stays visible (fail loud)


def test_invalid_top_p_mode_raises():
    init_device_properties_triton()
    logits = torch.randn(2, 4096, device="npu")
    with pytest.raises(ValueError, match="top_p_mode"):
        apply_top_k_top_p(logits, None, None, top_p_mode="bogus")
