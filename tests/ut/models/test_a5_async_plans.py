# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import mixed_quant_attention as mqa


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("tokens", [1, 5, 72, 256])
def test_attention_only_consumes_correct_builder_plan(monkeypatch, ratio, tokens):
    plans = {r: torch.full((1024,), r, dtype=torch.int32) for r in (0, 2, 1)}
    metadata = SimpleNamespace(
        swa=SimpleNamespace(
            a5_smla_plans=plans,
            ori_sparse_indices=torch.zeros((tokens, 1, 128), dtype=torch.int32),
            ori_topk_length=torch.full((tokens, 1), 6, dtype=torch.int32),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
            seq_lens=torch.tensor([6], dtype=torch.int32),
        ),
        attention=SimpleNamespace(compress_ratio=ratio, block_table=None, cache_seq_lens=None),
    )
    calls = []
    monkeypatch.setattr(mqa, "wait_for_device_metadata", lambda stage, group: calls.append(group))
    monkeypatch.setattr(
        mqa.ops,
        "mixed_quant_sparse_flash_mla_metadata",
        lambda *a, **k: pytest.fail("MLA must not build metadata"),
    )

    def mla(query, **kwargs):
        assert kwargs["metadata"] is plans[ratio]
        assert calls[-1] == id(plans[ratio])
        return query, None

    monkeypatch.setattr(mqa.ops, "mixed_quant_sparse_flash_mla", mla)
    q = torch.zeros((tokens, 64, 512))
    for _ in range(3):
        assert (
            mqa.qsmla(
                q,
                torch.empty(1),
                torch.empty(1) if ratio else None,
                metadata,
                torch.zeros((tokens, 512), dtype=torch.int32) if ratio else None,
                window_size=128,
                sinks=torch.zeros(64),
                softmax_scale=1,
            )
            is q
        )


@pytest.mark.parametrize("plans", [None, {}, {0: torch.zeros(1, dtype=torch.int32)}])
def test_attention_requires_its_own_build_plan(monkeypatch, plans):
    metadata = SimpleNamespace(
        swa=SimpleNamespace(a5_smla_plans=plans),
        attention=SimpleNamespace(compress_ratio=2),
    )
    monkeypatch.setattr(mqa, "_resolve_window_indices", lambda *args: (None, None))
    monkeypatch.setattr(
        mqa.ops, "mixed_quant_sparse_flash_mla_metadata", lambda *args, **kwargs: pytest.fail("inline planner")
    )
    monkeypatch.setattr(mqa.ops, "mixed_quant_sparse_flash_mla", lambda *args, **kwargs: pytest.fail("MLA"))
    with pytest.raises(RuntimeError, match="A5 MLA plan c2 was not prepared"):
        mqa.qsmla(
            torch.zeros((1, 64, 512)),
            torch.empty(1),
            torch.empty(1),
            metadata,
            torch.zeros((1, 512), dtype=torch.int32),
            window_size=128,
            sinks=torch.zeros(64),
            softmax_scale=1,
        )
