# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.minimax_m3.ops.msa_m3_index_postprocess import index_decode_topk_fused


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 0), (1, 2)])
def test_decode_topk_varying_runtime_parameters(strided: bool, init_blocks: int, local_blocks: int) -> None:
    # Keep model configuration fixed while varying batch shape, widths and MTP.
    heads, topk, block_size = 2, 8, 128
    for requests, query_len, logical_blocks, width in [(1, 1, 3, 4), (3, 4, 17, 32), (2, 1, 8, 16)]:
        tokens = requests * query_len
        reference = torch.arange(heads * tokens * width, dtype=torch.float32).view(heads, tokens, width)
        context = torch.arange(requests, dtype=torch.int32) * block_size + block_size - 1
        score = reference.to("npu")
        context_npu = context.to("npu")
        if strided:
            score_storage = torch.empty((heads, tokens, width * 2), dtype=torch.float32, device="npu")
            score = score_storage[..., ::2]
            score.copy_(reference)
            context_storage = torch.empty(requests * 2, dtype=torch.int32, device="npu")
            context_npu = context_storage[::2]
            context_npu.copy_(context)

        indices, counts = index_decode_topk_fused(
            score, context_npu, query_len, topk, logical_blocks, init_blocks, local_blocks
        )
        indices, counts = indices.cpu(), counts.cpu()
        assert indices.dtype == torch.int32
        assert counts.dtype == torch.int32
        for query in range(tokens):
            visible = min((int(context[query // query_len]) + query % query_len) // block_size + 1, logical_blocks)
            expected_scores = reference[:, query, :visible].clone()
            expected_scores[:, :init_blocks] = 1.0e30
            if local_blocks:
                expected_scores[:, max(visible - local_blocks, 0) :] = 1.0e29
            valid = indices[:, query] >= 0
            assert torch.all(counts[:, query] == min(visible, topk))
            assert torch.equal(valid.sum(-1).to(torch.int32), counts[:, query])
            assert torch.all(indices[:, query][~valid] == -1)
            for head in range(heads):
                selected = indices[head, query][valid[head]].long()
                assert torch.all(selected < visible)
                assert selected.unique().numel() == selected.numel()
                # Compare scores to allow any ordering among forced-score ties.
                actual = expected_scores[head, selected].sort(descending=True).values
                expected = expected_scores[head].topk(min(visible, topk)).values
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
