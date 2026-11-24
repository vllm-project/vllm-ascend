# SPDX-License-Identifier: Apache-2.0
"""Cover Qwen4Exp QSA backend selection on Ascend."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa
from vllm_ascend.utils import vllm_version_is


def _make_indexer() -> ascend_qsa.AscendQSAIndexer:
    indexer = object.__new__(ascend_qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.c8_enabled = False
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
        num_actual_tokens=1,
    )


@pytest.mark.parametrize(
    "c8_enabled,use_bf16_indexer_v1,backend",
    [(False, False, "v2"), (False, True, "v1"), (True, False, "int8")],
)
def test_select_dispatches_to_matching_indexer(monkeypatch, c8_enabled, use_bf16_indexer_v1, backend) -> None:
    sentinel = torch.zeros(1, 2051, dtype=torch.int32)
    metadata = _select_metadata()
    query = torch.zeros(1, 4, 128)
    indexer = _make_indexer()
    indexer.c8_enabled = c8_enabled
    indexer.use_bf16_indexer_v1 = use_bf16_indexer_v1
    cache = indexer.compressed_key_cache.kv_cache
    selectors = {
        "v1": Mock(return_value=sentinel),
        "v2": Mock(return_value=sentinel),
        "int8": Mock(return_value=sentinel),
    }
    monkeypatch.setattr(ascend_qsa, "qsa_select_paged_tokens_lightning_v1", selectors["v1"])
    monkeypatch.setattr(ascend_qsa, "qsa_select_paged_tokens_lightning_v2", selectors["v2"])
    monkeypatch.setattr(ascend_qsa, "qsa_select_paged_tokens_quantized", selectors["int8"])

    assert indexer._select(query, metadata, None) is sentinel
    if backend == "int8":
        selectors[backend].assert_called_once_with(
            query,
            cache,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.seq_lens,
            metadata.query_start_loc,
            2048,
            4,
            None,
        )
    else:
        selectors[backend].assert_called_once_with(query, cache, metadata, 2048, 4, None)
    for name, selector in selectors.items():
        if name != backend:
            selector.assert_not_called()


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
    if vllm_version_is("0.33.0"):
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
