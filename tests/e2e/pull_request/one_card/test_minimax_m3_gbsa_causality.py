# SPDX-License-Identifier: Apache-2.0
"""Sparse block score order must not expose future or padded KV rows."""

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import minimax_m3_sparse_attn_decode

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an Ascend NPU")


@pytest.mark.parametrize("query_length", [1, 4, 8])
@pytest.mark.parametrize("padding", [0, 2])
@torch.inference_mode()
def test_score_ordered_blocks_preserve_causality(query_length, padding):
    total = 157
    query_index = 0 if query_length == 1 else query_length // 2 - 1
    position = total - query_length + query_index
    query = torch.zeros(query_length, 16, 128, dtype=torch.bfloat16, device="npu")
    key = torch.zeros(2, 128, 1, 128, dtype=torch.bfloat16, device="npu")
    value = torch.zeros_like(key)
    # Score order puts the current partial block before an older full block.
    blocks = torch.tensor([1, 0] + [-1] * padding, dtype=torch.int32, device="npu").repeat(1, query_length, 1)
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="npu")
    lengths = torch.tensor([total], dtype=torch.int32, device="npu")

    def attention():
        output = torch.empty_like(query)
        minimax_m3_sparse_attn_decode(query, (key, value), blocks, table, lengths, 1, 128**-0.5, output, query_length)
        return output[query_index].clone()

    baseline = attention()
    # These rows are invisible to this query, including when they lie beyond
    # seqused_kv for a single-token decode or in the same selected sparse block.
    value.view(256, 1, 128)[position + 1 : position + 6].fill_(100)
    changed = attention()
    torch.testing.assert_close(baseline, torch.zeros_like(baseline), atol=0, rtol=0)
    torch.testing.assert_close(changed, baseline, atol=0, rtol=0)
