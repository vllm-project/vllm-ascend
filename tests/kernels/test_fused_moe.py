#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
# Copyright 2023 The vLLM team.
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
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_fused_moe.py`.
"""
import pytest
import torch

from vllm_ascend.ops.fused_moe  import fused_moe # noqa
from tests.kernels.utils import torch_moe


NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]


@pytest.mark.parametrize("m", [1, 33, 64, 222, 1024 * 128])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device="npu", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="npu", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="npu", dtype=dtype) / 10
    e_score_correction_bias = torch.randn(e, dtype=dtype, device="npu")
    score = torch.randn((m, e), device="npu", dtype=dtype)
    ascend_output = fused_moe(a, w1, w2, score, topk, renormalize=True,num_expert_group=1,topk_group = 1,scoring_func='sigmoid',e_score_correction_bias=e_score_correction_bias)
    torch_output = torch_moe(a, w1, w2, score, topk, renormalize=True,num_expert_group=1,topk_group = 1,scoring_func='sigmoid',e_score_correction_bias=e_score_correction_bias)
    torch.testing.assert_close(ascend_output, torch_output, atol=2e-2, rtol=0)