# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp QSA has one fused Ascend execution path."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.models.qwen4_exp import lightning_indexer
from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa
from vllm_ascend.utils import vllm_version_is


def _make_indexer() -> ascend_qsa.AscendQSAIndexer:
    indexer = object.__new__(ascend_qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.compressed_key_cache = SimpleNamespace(kv_cache=torch.zeros(1))
    indexer.token_topk = 2048
    indexer.compress_ratio = 4
    return indexer


def _select_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        token_to_req=torch.zeros(1, dtype=torch.int64),
        logical_positions=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int32),
        query_start_loc=torch.arange(2, dtype=torch.int32),
    )


def test_select_always_uses_cann_lightning_v2(monkeypatch) -> None:
    sentinel = torch.zeros(1, 2051, dtype=torch.int32)
    calls: list[tuple] = []

    def lightning(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(ascend_qsa, "qsa_select_paged_tokens_lightning", lightning)
    result = _make_indexer()._select(torch.zeros(1, 4, 128), _select_metadata(), None)

    assert result is sentinel
    assert len(calls) == 1


@pytest.mark.parametrize("page_size", [192, 384])
def test_lightning_indexer_accepts_runtime_page_sizes(page_size: int) -> None:
    output = lightning_indexer.qsa_select_paged_tokens_lightning(
        query=torch.empty(0, 8, 128, dtype=torch.bfloat16),
        compressed_key_cache=torch.empty(1, page_size, 1, 128, dtype=torch.bfloat16),
        block_table=torch.empty(0, 1, dtype=torch.int32),
        token_to_req=torch.empty(0, dtype=torch.int64),
        query_positions=torch.empty(0, dtype=torch.int64),
        sequence_lengths=torch.empty(0, dtype=torch.int32),
        query_start_loc=torch.zeros(1, dtype=torch.int32),
        token_topk=2048,
        compress_ratio=4,
    )
    assert output.shape == (0, 2051)


def _make_impl() -> ascend_qsa.AscendQSAImpl:
    impl = object.__new__(ascend_qsa.AscendQSAImpl)
    impl.head_size = 2
    return impl


def test_impl_always_uses_cann_gbsa(monkeypatch) -> None:
    sentinel = torch.zeros(1, 2, 2)
    calls: list[tuple] = []
    logical_positions = torch.zeros(1, dtype=torch.int64)
    seq_lens = torch.ones(1, dtype=torch.int32)
    layer = SimpleNamespace(
        topk_indices_buffer=torch.zeros(1, 515, dtype=torch.int32),
        indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
    )
    kv_cache = (torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    output = torch.zeros(1, 2, 2)

    def gbsa(*args, **kwargs):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(ascend_qsa, "qsa_gbsa_attention", gbsa)
    impl: Any = _make_impl()
    if vllm_version_is("0.32.0"):
        result = impl.forward_qsa(
            layer=layer,
            query=torch.zeros(1, 2, 2),
            key=torch.zeros(0),
            value=torch.zeros(0),
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            token_to_req=torch.zeros(1, dtype=torch.int64),
            logical_positions=logical_positions,
            seq_lens=seq_lens,
        )
    else:
        result = impl.forward(
            layer=layer,
            query=torch.zeros(1, 2, 2),
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            token_to_req=torch.zeros(1, dtype=torch.int64),
            logical_positions=logical_positions,
            seq_lens=seq_lens,
        )

    assert result is sentinel
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][6], logical_positions)
    assert calls[0][7] is seq_lens
