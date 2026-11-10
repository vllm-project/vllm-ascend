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
# Semantics: topk_renorm (CANN/upstream), bit-exact vs CANN npu_top_k_top_p.

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
    out = apply_top_k_top_p(logits, k, p.to(dtype).float())
    ref = cann_masked(logits, k, p)
    assert torch.equal(out, ref)


def test_scalar_and_none_variants():
    """Scalar k/p combined with None (disabled), topk_renorm vs CANN."""
    init_device_properties_triton()
    B, V = 4, 8192
    logits = torch.randn(B, V, device="npu") * 4.0
    for k, p in [(None, None), (1, None), (50, 0.9), (V, 0.5), (200, 1.0)]:
        out = apply_top_k_top_p(logits.clone(), k, p)
        k_t = None if k is None or k >= V else torch.full((B,), k, dtype=torch.int32, device="npu")
        p_t = None if p is None or p >= 1.0 else torch.full((B,), p, dtype=torch.float32, device="npu")
        ref = cann_masked(logits, k_t, p_t)
        assert torch.equal(out, ref)


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
