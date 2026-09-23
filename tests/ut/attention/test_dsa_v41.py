# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for an empty DSA index during FULL graph capture."""

from types import SimpleNamespace

import torch

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl, DeepseekV41PreparedIndexer

TOKENS, TOPK = 4, 512


def _impl(role):
    impl = AscendDSAV41Impl.__new__(AscendDSAV41Impl)
    impl.role = role
    impl.index_k_source_prefix = "model.layers.2.attn"
    impl.topology = SimpleNamespace(candidate_topk_blocks=8, candidate_block_size=8)
    return impl


def _attn(selected):
    shared = SimpleNamespace(
        topk_indices=torch.full((TOKENS, TOPK), 7, dtype=torch.int32),
        candidates=torch.full((TOKENS, 1, 8), 7, dtype=torch.int32),
    )
    indexer = SimpleNamespace(select_projected=lambda *args, **kwargs: (selected, None))
    return SimpleNamespace(shared_state=shared, indexer=indexer), shared


def test_empty_cache_selection_publishes_no_slot(monkeypatch):
    prefix = "model.layers.2.attn"
    monkeypatch.setattr(
        dsa_v41,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={prefix: SimpleNamespace(kv_cache=[None])}),
    )
    selected = torch.full((TOKENS, 0), -1, dtype=torch.int32)
    attn, shared = _attn(selected)
    impl = _impl(
        SimpleNamespace(
            has_long_context=True,
            is_index_source=True,
            is_candidate_source=False,
            uses_candidate_filter=False,
        )
    )

    out = impl._select_sparse_indices(
        attn,
        torch.zeros(TOKENS, 8),
        torch.zeros(TOKENS, 8),
        torch.arange(TOKENS),
        None,
        None,
        SimpleNamespace(
            swa=SimpleNamespace(num_actual_tokens=TOKENS),
            indexer=SimpleNamespace(cache=object()),
        ),
        DeepseekV41PreparedIndexer(
            query=torch.zeros(TOKENS, 1, 8),
            weights=torch.zeros(TOKENS, 1),
        ),
    )

    assert shared.topk_indices.shape == (TOKENS, TOPK)
    assert torch.all(shared.topk_indices == -1)
    assert out is not None and out.shape == (TOKENS, TOPK)
