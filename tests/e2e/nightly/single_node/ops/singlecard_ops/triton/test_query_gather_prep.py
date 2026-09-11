# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,heads", [(1, 1), (7, 8), (32, 16), (0, 8)])
@torch.inference_mode()
def test_query_gather_prep_matches_torch(dtype, tokens, heads):
    ql_nope = torch.randn(tokens, heads, 512, dtype=dtype, device="npu")
    q_pe = torch.randn(tokens, heads, 64, dtype=dtype, device="npu")
    actual = prep_query_head_major(ql_nope, q_pe)
    expected = torch.cat((ql_nope, q_pe), dim=-1).permute(1, 0, 2).contiguous()
    assert actual is not None
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_query_gather_prep_falls_back_for_unsafe_tail():
    ql_nope = torch.randn(3, 2, 512, dtype=torch.float16, device="npu")
    q_pe = torch.randn(3, 2, 48, dtype=torch.float16, device="npu")
    assert prep_query_head_major(ql_nope, q_pe) is None
