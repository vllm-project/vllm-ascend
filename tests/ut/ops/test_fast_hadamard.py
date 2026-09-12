#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""CPU-side correctness of the fast Hadamard transform (FHT) against the
dense matrix Hadamard used by the DeepSeek-V4 DSA indexer.

The PTO FHT computes ``x @ H`` up to a fixed output permutation P, which
cancels in the indexer score ``q . k^T``, so equivalence is checked on
permutation structure and inner products rather than element order.
"""

import math

import torch
from scipy.linalg import hadamard as scipy_hadamard

from vllm_ascend.ops.fast_hadamard import (
    _fast_hadamard_dynamic_quant_int8_ref,
    fast_hadamard_pto_ref_inplace,
)

N = 128  # V4-Flash index_head_dim
SCALE = N**-0.5
TOPK = 512  # V4-Flash index_topk

torch.manual_seed(0)


def dense_h() -> torch.Tensor:
    return torch.tensor(scipy_hadamard(N, dtype=float), dtype=torch.float32)


def rotate_activation_ref(x: torch.Tensor) -> torch.Tensor:
    # mirrors vllm_ascend/attention/dsa_v1.py::rotate_activation
    return torch.nn.functional.linear(x, dense_h()) * SCALE


def fht(x: torch.Tensor) -> torch.Tensor:
    return fast_hadamard_pto_ref_inplace(x) * SCALE


def test_fht_is_permuted_hadamard():
    f = fast_hadamard_pto_ref_inplace(torch.eye(N))
    assert torch.equal(f.abs(), torch.ones(N, N))
    assert torch.allclose(f @ f.T, N * torch.eye(N))
    match = (f @ dense_h().T / N).round()
    assert torch.equal(match.abs().sum(0), torch.ones(N))  # bijective perm
    assert torch.equal(match.abs().sum(1), torch.ones(N))


def test_indexer_scores_invariant():
    q, k = torch.randn(1024, N), torch.randn(2048, N)
    s_ref = rotate_activation_ref(q) @ rotate_activation_ref(k).T
    s_fht = fht(q) @ fht(k).T
    torch.testing.assert_close(s_fht, s_ref, rtol=1e-4, atol=1e-3)


def test_int8_quantized_score_topk_overlap():
    q, k = torch.randn(64, N), torch.randn(2048, N)
    s_ref = rotate_activation_ref(q) @ rotate_activation_ref(k).T
    qq, qs = _fast_hadamard_dynamic_quant_int8_ref(q)
    kq, ks = _fast_hadamard_dynamic_quant_int8_ref(k)
    s = (qq.float() * qs) @ (kq.float() * ks).T
    assert ((s - s_ref).norm() / s_ref.norm()).item() < 0.02
    top_ref = s_ref.topk(TOPK, dim=-1).indices
    top = s.topk(TOPK, dim=-1).indices
    overlap = (top.unsqueeze(-1) == top_ref.unsqueeze(-2)).any(-1).float().mean()
    assert overlap.item() >= 0.98


def test_fht_matches_log2_structure():
    x = torch.randn(8, N)
    out = fast_hadamard_pto_ref_inplace(x)
    assert out.shape == x.shape
    assert int(math.log2(N)) == 7
    torch.testing.assert_close(fast_hadamard_pto_ref_inplace(out) / N, x, rtol=1e-4, atol=1e-4)
