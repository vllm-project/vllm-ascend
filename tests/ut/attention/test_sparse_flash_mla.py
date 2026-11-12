# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import pytest
import torch

from vllm_ascend.attention.sparse_flash_mla import (
    _ensure_sinks,
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
