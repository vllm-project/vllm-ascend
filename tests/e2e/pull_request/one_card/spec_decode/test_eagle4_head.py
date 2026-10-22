# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU candidate kernel accuracy and ACL graph replay; no model weights needed."""

import pytest
import torch
import torch_npu

pytest.importorskip("eagle_harness")

from vllm_ascend.spec_decode.eagle4_head import rescore_candidates  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,candidates,hidden_size", [(1, 128, 5120), (7, 65, 160), (0, 128, 5120)])
def test_candidate_rescore_and_graph_replay(dtype, rows, candidates, hidden_size):
    generator = torch.Generator().manual_seed(71)
    weight_cpu = torch.randn(257, hidden_size, generator=generator).to(dtype) * 0.02
    hidden_cpu = torch.randn(rows, hidden_size, generator=generator).to(dtype)
    ids_cpu = torch.randint(0, 257, (rows, candidates), generator=generator)
    weight, hidden, ids = weight_cpu.npu(), hidden_cpu.npu(), ids_cpu.npu()
    result = rescore_candidates(weight, hidden, ids)
    expected = torch.einsum("bkh,bh->bk", weight_cpu[ids_cpu].float(), hidden_cpu.float()).to(dtype).float()
    torch.testing.assert_close(result.cpu(), expected, rtol=2 * torch.finfo(dtype).eps, atol=1e-5)
    if not rows:
        return
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = rescore_candidates(weight, hidden, ids)
    hidden.copy_(hidden + 0.125)
    ids.copy_(ids.flip(1))
    graph.replay()
    expected_replay = rescore_candidates(weight, hidden, ids)
    torch.npu.synchronize()
    torch.testing.assert_close(captured, expected_replay, rtol=0, atol=0)


@pytest.mark.parametrize("window", [1, 3, 8])
def test_draft_window_includes_current_token(window):
    """Check FIA's left boundary for both a chunk and a one-token decode."""
    generator = torch.Generator().manual_seed(19)
    key = torch.randn(7, 2, 64, generator=generator).to(torch.bfloat16)
    value = torch.randn(7, 2, 64, generator=generator).to(torch.bfloat16)
    mask = torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu_(1)
    for query_len in (1, 2):
        query = torch.randn(query_len, 2, 64, generator=generator).to(torch.bfloat16)
        positions = torch.arange(7 - query_len, 7)[:, None]
        keys = torch.arange(7)[None, :]
        visible = (keys <= positions) & (keys >= positions - window + 1)
        scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) / 8
        scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
        expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value.float())
        actual, _ = torch_npu.npu_fused_infer_attention_score(
            query.npu(),
            key.npu(),
            value.npu(),
            num_heads=2,
            num_key_value_heads=2,
            input_layout="TND",
            scale=1 / 8,
            atten_mask=mask,
            sparse_mode=4,
            pre_tokens=window - 1,
            next_tokens=0,
            actual_seq_lengths=[query_len],
            actual_seq_lengths_kv=[7],
        )
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.02, atol=0.02)
