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


def test_a3_adapters_clamp_zero_length_padding_requests(monkeypatch):
    """Cudagraph padding adds dummy requests with no queries and no cached KV.

    The compiled binding's AICPU tiler rejects zero-length sequences (aicpu
    exception in SparseFlashMlaMetadata), while the CANN toolkit op tolerates
    them; the adapters clamp ``seqused`` to one token, which the query-less
    dummy never consumes.
    """
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=(torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", attention_op, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", metadata_op, raising=False)

    a3_sparse_flash_mla_metadata(
        num_heads_q=8,
        num_heads_kv=1,
        head_dim=64,
        cu_seqlens_q=torch.tensor([0, 5, 5], dtype=torch.int64),
        seqused_q=torch.tensor([5, 0], dtype=torch.int32),
        seqused_kv=torch.tensor([10, 0], dtype=torch.int32),
        max_seqlen_q=5,
        max_seqlen_kv=10,
        batch_size=2,
        ori_topk_length=torch.zeros((5, 1), dtype=torch.int32),
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
    kwargs = metadata_op.call_args.kwargs
    assert kwargs["seqused_ori_kv"].tolist() == [10, 1]

    a3_sparse_flash_mla(
        torch.empty(5, 8, 64),
        ori_kv=torch.empty(1, 128, 1, 64, dtype=torch.bfloat16),
        ori_block_table=torch.zeros((2, 4), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 5, 5], dtype=torch.int32),
        seqused_kv=torch.tensor([10, 0], dtype=torch.int32),
        sinks=torch.ones(8),
        metadata=torch.zeros(1024, dtype=torch.int32),
        softmax_scale=0.5,
        ori_sparse_indices=torch.zeros((5, 1, 16), dtype=torch.int32),
        ori_topk_length=torch.zeros((5, 1), dtype=torch.int32),
        topk_value_mode=1,
        layout_kv="PA_ND",
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=0,
        ori_win_right=0,
    )
    kwargs = attention_op.call_args.kwargs
    assert kwargs["seqused_ori_kv"].tolist() == [10, 1]


def test_a3_adapters_reject_unexpected_kwargs(monkeypatch):
    """Leftover kwargs must fail loud instead of being silently dropped."""
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", mock.Mock(return_value=torch.empty(0)), raising=False
    )
    with pytest.raises(ValueError, match="unexpected_kwarg"):
        a3_sparse_flash_mla_metadata(
            num_heads_q=8,
            num_heads_kv=1,
            head_dim=64,
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            batch_size=1,
            max_seqlen_q=2,
            ori_topk_length=torch.zeros((2, 1), dtype=torch.int32),
            ori_topk=16,
            ori_mask_mode=0,
            cmp_mask_mode=0,
            ori_win_left=0,
            ori_win_right=0,
            unexpected_kwarg=1,
        )
