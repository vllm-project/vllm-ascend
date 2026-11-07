# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import pytest
import torch

from vllm_ascend.attention.sparse_flash_mla import (
    _ensure_sinks,
    a3_sparse_flash_mla,
    a3_sparse_flash_mla_metadata,
    sparse_flash_mla,
    sparse_flash_mla_metadata,
)


def test_adapter_enforces_bf16_paged_layout():
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        sparse_flash_mla_metadata(layout_kv="PA_ND")
        sparse_flash_mla(torch.empty(0), layout_kv="PA_ND", sinks=torch.ones(1))

    assert metadata_op.call_args.kwargs["layout_kv"] == "PA_BBND"
    assert attention_op.call_args.kwargs["layout_kv"] == "PA_BBND"


def test_adapter_drops_kv_cu_seqlens_for_paged_layout():
    """SparseFlashMla only accepts KV cu_seqlens when layout_kv is TND."""
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    cu_ori = torch.tensor([0, 4], dtype=torch.int32)
    cu_cmp = torch.tensor([0, 1], dtype=torch.int32)
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        sparse_flash_mla_metadata(cu_seqlens_ori_kv=cu_ori, cu_seqlens_cmp_kv=cu_cmp)
        sparse_flash_mla(torch.empty(0), cu_seqlens_ori_kv=cu_ori, cu_seqlens_cmp_kv=cu_cmp, sinks=torch.ones(1))

    assert "cu_seqlens_ori_kv" not in metadata_op.call_args.kwargs
    assert "cu_seqlens_cmp_kv" not in metadata_op.call_args.kwargs
    assert "cu_seqlens_ori_kv" not in attention_op.call_args.kwargs
    assert "cu_seqlens_cmp_kv" not in attention_op.call_args.kwargs


def test_adapter_rejects_missing_sinks():
    """A sinks tensor allocated inside the adapter would be freed as soon as
    the call returns, while a captured ACL Graph still holds its address."""
    with pytest.raises(ValueError, match="caller-owned per-head sinks"):
        _ensure_sinks({})


def test_adapter_keeps_caller_owned_sinks():
    sinks = torch.ones(4, dtype=torch.float32)
    kwargs = {"sinks": sinks}
    _ensure_sinks(kwargs)
    assert kwargs["sinks"] is sinks


def test_adapter_drops_dspark_draft_indices_but_keeps_vision():
    """DSpark draft (decode) passes borrow-mode indices that A5 BF16
    SparseFlashMla cannot execute (kernel trap 507015); the adapter drops
    them so attention is derived from the caller's window and the non-causal
    metadata, like the FP8 mixed-quant adapter. Vision prefill passes indices
    alongside the prefill-only cu_seqlens_ori_kv and must keep them for its
    span-level bidirectional masking."""
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    indices = torch.zeros((2, 8), dtype=torch.int32)
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        sparse_flash_mla(torch.empty(0), ori_sparse_indices=indices, sinks=torch.ones(1))
        assert "ori_sparse_indices" not in attention_op.call_args.kwargs

        sparse_flash_mla(
            torch.empty(0),
            ori_sparse_indices=indices,
            cu_seqlens_ori_kv=torch.tensor([0, 4], dtype=torch.int32),
            sinks=torch.ones(1),
        )
        assert attention_op.call_args.kwargs["ori_sparse_indices"] is indices


def test_adapter_passes_caller_window_and_widens_on_full_coverage():
    """Window assignment mirrors the FP8 mixed-quant adapter of #18808: the
    caller's win_left/win_right pass through unchanged; a full-coverage index
    set (eager mode, max_seqlen_kv supplied) widens the right edge to the
    batch total so the window alone covers the same KV."""
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    indices = torch.zeros((2, 8), dtype=torch.int32)
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        # caller's window passes through untouched (no max_seqlen_kv)
        sparse_flash_mla(
            torch.empty(0),
            ori_sparse_indices=indices,
            ori_win_left=132,
            ori_win_right=0,
            sinks=torch.ones(1),
        )
        kw = attention_op.call_args.kwargs
        assert kw["ori_win_left"] == 132
        assert kw["ori_win_right"] == 0
        assert "ori_sparse_indices" not in kw

        # full-coverage indices widen the right edge; the adapter treats
        # a missing or raising stream-capture check as eager (not capturing),
        # so the widening runs identically with and without an NPU device
        full = torch.arange(8, dtype=torch.int32).repeat(2)
        sparse_flash_mla(
            torch.empty((10, 4, 8)),
            ori_sparse_indices=full,
            max_seqlen_kv=8,
            sinks=torch.ones(1),
        )
        kw = attention_op.call_args.kwargs
        assert kw["ori_win_left"] == 0
        assert kw["ori_win_right"] == 10 - 1
        assert "ori_sparse_indices" not in kw


def test_a3_adapters_route_to_compiled_binding(monkeypatch):
    """The A3 adapters translate DSA kwargs to the compiled binding (v41 style)."""
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=(torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", attention_op, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", metadata_op, raising=False)

    result = a3_sparse_flash_mla_metadata(
        device="npu:0",
        num_heads_q=8,
        num_heads_kv=1,
        head_dim=64,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int64),
        seqused_q=torch.tensor([1, 1], dtype=torch.int32),
        seqused_kv=torch.tensor([8, 12], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_kv=12,
        batch_size=1,
        ori_topk_length=torch.zeros((2, 1), dtype=torch.int32),
        ori_topk=16,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=0,
        ori_win_right=0,
    )
    assert result is metadata_op.return_value
    kwargs = metadata_op.call_args.kwargs
    assert kwargs["seqused_ori_kv"].tolist() == [8, 12]
    assert kwargs["max_seqlen_ori_kv"] == 12
    assert kwargs["cu_seqlens_q"].dtype == torch.int32
    assert "seqused_q" not in kwargs
    assert "device" not in kwargs
    assert kwargs["layout_kv"] == "PA_BBND"

    out = a3_sparse_flash_mla(
        torch.empty(2, 8, 64),
        ori_kv=torch.empty(1, 128, 1, 64, dtype=torch.bfloat16),
        ori_block_table=torch.zeros((1, 4), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        seqused_kv=torch.tensor([8, 12], dtype=torch.int32),
        sinks=torch.ones(8),
        metadata=torch.zeros(1024, dtype=torch.int32),
        softmax_scale=0.5,
        ori_sparse_indices=torch.zeros((2, 1, 16), dtype=torch.int32),
        ori_topk_length=torch.zeros((2, 1), dtype=torch.int32),
        topk_value_mode=1,
        layout_kv="PA_ND",
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=0,
        ori_win_right=0,
    )
    assert out is attention_op.return_value
    kwargs = attention_op.call_args.kwargs
    assert kwargs["seqused_ori_kv"].tolist() == [8, 12]
    assert kwargs["sinks"] is not None
    assert kwargs["layout_kv"] == "PA_BBND"
    assert kwargs["topk_value_mode"] == 1
    assert kwargs["softmax_scale"] == 0.5


def test_a3_adapters_build_topk_from_borrow_mode(monkeypatch):
    """A borrow-mode draft call (physical indices, no ori_topk_length, causal
    mask) arrives from dsa_v1; the A3 adapter builds logical indices and
    per-query lengths from the caller's kwargs and switches to top-k mode."""
    attention_op = mock.Mock(return_value=(torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", attention_op, raising=False)

    # borrow-mode draft call: indices + causal window, no topk_length
    physical = torch.zeros((2, 1, 4), dtype=torch.int32)
    a3_sparse_flash_mla(
        torch.empty(2, 4, 64),
        ori_kv=torch.empty(1, 128, 1, 64, dtype=torch.bfloat16),
        ori_block_table=torch.zeros((1, 4), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        seqused_kv=torch.tensor([6, 8], dtype=torch.int32),
        sinks=torch.ones(4),
        metadata=torch.zeros(1024, dtype=torch.int32),
        softmax_scale=0.5,
        ori_sparse_indices=physical,
        ori_mask_mode=4,
        ori_win_left=5,
        ori_win_right=0,
    )
    kw = attention_op.call_args.kwargs
    # adapter switched to top-k mode
    assert kw["ori_mask_mode"] == 0
    assert kw["ori_win_left"] == 0
    assert kw["ori_win_right"] == 0
    # built indices + lengths from the kwargs
    assert kw["ori_sparse_indices"] is not physical
    assert kw["ori_topk_length"] is not None
    assert kw["ori_topk_length"].shape[1] == 1
    # lengths are uniform per request: min(seqused, query_len + win_left)
    # req0: min(6, 1+5)=6, req1: min(8, 1+5)=6
    assert kw["ori_topk_length"].flatten().tolist() == [6, 6]


def test_a3_adapters_clamp_zero_length_padding_requests(monkeypatch):
    """Cudagraph padding adds seqused=0 requests; the AICPU tiler crashes on
    zero-length sequences, so the adapter clamps them to 1."""
    attention_op = mock.Mock(return_value=(torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", attention_op, raising=False)

    a3_sparse_flash_mla(
        torch.empty(2, 4, 64),
        ori_kv=torch.empty(1, 128, 1, 64, dtype=torch.bfloat16),
        ori_block_table=torch.zeros((1, 4), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        seqused_kv=torch.tensor([0, 8], dtype=torch.int32),
        sinks=torch.ones(4),
        metadata=torch.zeros(1024, dtype=torch.int32),
        softmax_scale=0.5,
        ori_sparse_indices=torch.zeros((2, 1, 4), dtype=torch.int32),
        ori_topk_length=torch.ones((2, 1), dtype=torch.int32),
        topk_value_mode=1,
        layout_kv="PA_ND",
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=0,
        ori_win_right=0,
    )
    kw = attention_op.call_args.kwargs
    assert kw["seqused_ori_kv"].tolist() == [1, 8]


def test_a3_adapters_reject_unexpected_kwargs(monkeypatch):
    """Silently dropping new DSA kwargs would hide routing bugs; the adapter
    rejects them so additions fail loudly in CI."""
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", mock.Mock(), raising=False)
    with pytest.raises(ValueError, match="Unexpected a3 attention kwargs"):
        a3_sparse_flash_mla(
            torch.empty(2, 4, 64),
            ori_kv=torch.empty(1, 128, 1, 64, dtype=torch.bfloat16),
            ori_block_table=torch.zeros((1, 4), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            seqused_kv=torch.tensor([8], dtype=torch.int32),
            sinks=torch.ones(4),
            metadata=torch.zeros(1024, dtype=torch.int32),
            ori_sparse_indices=torch.zeros((2, 1, 4), dtype=torch.int32),
            ori_topk_length=torch.ones((2, 1), dtype=torch.int32),
            layout_kv="PA_ND",
            ori_mask_mode=0,
            cmp_mask_mode=0,
            ori_win_left=0,
            ori_win_right=0,
            mystery_kwarg=1,
        )
