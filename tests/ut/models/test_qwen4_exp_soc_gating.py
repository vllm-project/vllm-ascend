# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp QSA has one fused Ascend execution path."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import lightning_indexer
from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa


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
    impl.scale = 1.0
    impl.num_kv_heads = 1
    return impl


def _metadata_for_shortseq(num_actual_tokens=1, max_seq_len=None, seq_len=1, logical=None):
    meta = SimpleNamespace(
        num_actual_tokens=num_actual_tokens,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.full((1,), seq_len, dtype=torch.int32),
        logical_positions=torch.zeros(1, dtype=torch.int64)
        if logical is None
        else torch.tensor(logical, dtype=torch.int64),
        max_seq_len=max_seq_len,
    )
    return meta


def test_forward_qsa_gbsa_when_shortseq_fia_disabled(monkeypatch) -> None:
    """Env switch off must keep the legacy GBSA behavior."""
    monkeypatch.setenv("VLLM_ASCEND_QSA_SHORTSEQ_FIA", "0")
    sentinel = torch.zeros(1, 2, 2)
    calls: list[tuple] = []

    def gbsa(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(ascend_qsa, "qsa_gbsa_attention", gbsa)
    impl = _make_impl()
    result = impl.forward_qsa(
        layer=SimpleNamespace(
            topk_indices_buffer=torch.zeros(1, 515, dtype=torch.int32),
            indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
        ),
        query=torch.zeros(1, 2, 2),
        key=torch.zeros(0),
        value=torch.zeros(0),
        kv_cache=(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),
        attn_metadata=_metadata_for_shortseq(max_seq_len=5),
        output=torch.zeros(1, 2, 2),
        token_to_req=torch.zeros(1, dtype=torch.int64),
    )
    assert result is sentinel
    assert len(calls) == 1


def test_forward_qsa_shortseq_fia_branch(monkeypatch) -> None:
    """Short sequences route the dense attention to the FIA backend."""
    monkeypatch.setenv("VLLM_ASCEND_QSA_SHORTSEQ_FIA", "1")
    sentinel = torch.zeros(1, 2, 2)
    calls: list[tuple] = []

    def fia(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(ascend_qsa, "qsa_fia_shortseq_attention", fia)
    impl = _make_impl()
    impl.shortseq_fia = True
    result = impl.forward_qsa(
        layer=SimpleNamespace(
            topk_indices_buffer=torch.zeros(1, 515, dtype=torch.int32),
            indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
        ),
        query=torch.zeros(1, 2, 2),
        key=torch.zeros(0),
        value=torch.zeros(0),
        kv_cache=(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),
        attn_metadata=_metadata_for_shortseq(max_seq_len=5),
        output=torch.zeros(1, 2, 2),
        token_to_req=torch.zeros(1, dtype=torch.int64),
    )
    assert result is sentinel
    assert len(calls) == 1


def test_forward_qsa_long_seq_falls_back_to_gbsa(monkeypatch) -> None:
    """Sequences beyond the indexer budget keep the GBSA sparse path."""
    monkeypatch.setenv("VLLM_ASCEND_QSA_SHORTSEQ_FIA", "1")
    sentinel = torch.zeros(1, 2, 2)
    calls: list[tuple] = []

    def gbsa(*args):
        calls.append(args)
        return sentinel

    def fia(*args, **kwargs):
        raise AssertionError("shortseq FIA must not run for long sequences")

    monkeypatch.setattr(ascend_qsa, "qsa_gbsa_attention", gbsa)
    monkeypatch.setattr(ascend_qsa, "qsa_fia_shortseq_attention", fia)
    impl = _make_impl()
    impl.shortseq_fia = True
    result = impl.forward_qsa(
        layer=SimpleNamespace(
            topk_indices_buffer=torch.zeros(1, 515, dtype=torch.int32),
            indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
        ),
        query=torch.zeros(1, 2, 2),
        key=torch.zeros(0),
        value=torch.zeros(0),
        kv_cache=(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),
        attn_metadata=_metadata_for_shortseq(max_seq_len=4096),
        output=torch.zeros(1, 2, 2),
        token_to_req=torch.zeros(1, dtype=torch.int64),
    )
    assert result is sentinel
    assert len(calls) == 1


def test_forward_qsa_always_uses_cann_gbsa(monkeypatch) -> None:
    sentinel = torch.zeros(1, 2, 2)
    calls: list[tuple] = []

    def gbsa(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(ascend_qsa, "qsa_gbsa_attention", gbsa)
    result = _make_impl().forward_qsa(
        layer=SimpleNamespace(
            topk_indices_buffer=torch.zeros(1, 515, dtype=torch.int32),
            indexer=SimpleNamespace(token_topk=2048, compress_ratio=4),
        ),
        query=torch.zeros(1, 2, 2),
        key=torch.zeros(0),
        value=torch.zeros(0),
        kv_cache=(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),
        attn_metadata=SimpleNamespace(
            num_actual_tokens=1,
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
        ),
        output=torch.zeros(1, 2, 2),
        token_to_req=torch.zeros(1, dtype=torch.int64),
    )

    assert result is sentinel
    assert len(calls) == 1
