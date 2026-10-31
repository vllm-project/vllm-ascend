# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import mixed_quant_attention


@pytest.mark.parametrize("rows", [1, 5, 72])
@pytest.mark.parametrize("compressed", ["none", "explicit", "inferred"])
def test_qsmla_plans_from_current_layer_lengths(monkeypatch, rows, compressed):
    """A stale shared plan must never replace the current layer's real lengths."""
    has_cmp = compressed != "none"
    ori_lengths = torch.full((rows, 1), 6, dtype=torch.int32)
    ori_indices = torch.zeros((rows, 1, 128), dtype=torch.int32)
    cmp_indices = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32).repeat(rows, 1)
    cmp_lengths = torch.full((rows, 1), 1, dtype=torch.int32) if compressed == "explicit" else None
    metadata = SimpleNamespace(
        swa=SimpleNamespace(
            ori_sparse_indices=ori_indices,
            ori_topk_length=ori_lengths,
            smla_metadata=torch.full((1024,), 999999, dtype=torch.int32),
            query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            seq_lens=torch.tensor([6], dtype=torch.int32),
        ),
        attention=SimpleNamespace(
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            cache_seq_lens=torch.tensor([2], dtype=torch.int32),
        ),
    )
    plans = []
    events = []

    def plan(ori, cmp, **kwargs):
        events.append("plan")
        assert kwargs["has_cmp_kv"] == has_cmp
        assert kwargs["cu_seqlens_q"] is metadata.swa.query_start_loc
        expected_cmp = cmp_lengths if compressed == "explicit" else (cmp_indices >= 0).sum(-1, keepdim=True)
        if not has_cmp:
            expected_cmp = torch.zeros_like(ori_lengths)
        torch.testing.assert_close(ori, ori_lengths)
        torch.testing.assert_close(cmp, expected_cmp.to(torch.int32))
        task = torch.tensor([int(ori.sum()), int(cmp.sum())], dtype=torch.int32)
        plans.append(task)
        return task

    def attention(query, **kwargs):
        events.append("attention")
        assert plans, "MLA must be preceded by planning from actual lengths"
        assert kwargs["metadata"] is plans[-1]
        assert kwargs["metadata"] is not metadata.swa.smla_metadata
        assert (kwargs["cmp_kv"] is not None) == has_cmp
        return query, None

    monkeypatch.setattr(mixed_quant_attention.ops, "mixed_quant_sparse_flash_mla_metadata", plan)
    monkeypatch.setattr(mixed_quant_attention.ops, "mixed_quant_sparse_flash_mla", attention)
    monkeypatch.setattr(
        mixed_quant_attention,
        "wait_for_device_metadata",
        lambda *args: pytest.fail("must not wait on stale shared MLA metadata"),
    )
    q = torch.zeros((rows, 64, 512), dtype=torch.bfloat16)
    for step in range(2):
        result = mixed_quant_attention.qsmla(
            q,
            torch.empty(1),
            torch.empty(1) if has_cmp else None,
            metadata,
            cmp_indices if has_cmp else None,
            window_size=128,
            sinks=torch.zeros(64),
            softmax_scale=512**-0.5,
            compressed_lengths=cmp_lengths,
        )
        assert result is q
        assert plans[-1][0] == rows * (6 if step == 0 else 3)
        ori_lengths.fill_(3)
        if cmp_lengths is not None:
            cmp_lengths.zero_()
        cmp_indices.fill_(-1)
    assert events == ["plan", "attention", "plan", "attention"]
    assert len(plans) == 2
