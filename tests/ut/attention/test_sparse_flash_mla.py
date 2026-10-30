# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import pytest
import torch

from vllm_ascend.attention.sparse_flash_mla import (
    _ensure_sinks,
    build_dspark_sparse_flash_mla_metadata,
    sparse_flash_mla,
    sparse_flash_mla_metadata,
    write_dspark_topk_lengths,
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


def test_toolkit_adapters_own_dspark_draft_contract():
    """Draft calls (per-query top-k visibility) get the draft contract applied
    here; target-style calls keep the caller's window untouched."""
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    topk = torch.ones(2, 1, dtype=torch.int32)
    seqused = torch.tensor([4, 6], dtype=torch.int32)
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        # Draft-style calls carry ori_topk_length; the caller's window and
        # ratio must be overridden by the adapter contract.
        sparse_flash_mla_metadata(
            seqused_kv=seqused, ori_topk_length=topk, ori_win_left=7, ori_win_right=7, cmp_ratio=4
        )
        sparse_flash_mla(
            torch.empty(0),
            seqused_kv=seqused,
            ori_topk_length=topk,
            ori_win_left=7,
            ori_win_right=7,
            cmp_ratio=4,
            sinks=torch.ones(1),
        )
        # Target-style calls never carry ori_topk_length.
        sparse_flash_mla_metadata(seqused_kv=seqused, ori_win_left=7, ori_win_right=7)
        sparse_flash_mla(torch.empty(0), seqused_kv=seqused, ori_win_left=7, ori_win_right=7, sinks=torch.ones(1))

    draft_meta = metadata_op.call_args_list[0].kwargs
    assert draft_meta["ori_topk_length"] is topk
    assert draft_meta["cmp_ratio"] == 1
    assert draft_meta["ori_mask_mode"] == draft_meta["cmp_mask_mode"] == 0
    assert draft_meta["ori_win_left"] == draft_meta["ori_win_right"] == -1

    draft_attn = attention_op.call_args_list[0].kwargs
    assert draft_attn["ori_topk_length"] is topk
    assert draft_attn["cmp_ratio"] == 1
    assert draft_attn["ori_mask_mode"] == draft_attn["cmp_mask_mode"] == 0
    assert draft_attn["ori_win_left"] == draft_attn["ori_win_right"] == -1
    assert draft_attn["topk_value_mode"] == 1

    target_meta = metadata_op.call_args_list[1].kwargs
    assert "ori_topk_length" not in target_meta
    assert target_meta["ori_win_left"] == target_meta["ori_win_right"] == 7

    target_attn = attention_op.call_args_list[1].kwargs
    assert "ori_topk_length" not in target_attn
    assert target_attn["ori_win_left"] == target_attn["ori_win_right"] == 7
    assert "topk_value_mode" not in target_attn


def test_build_dspark_sparse_flash_mla_metadata_assembles_draft_call():
    """The moved dsa_v1 builder routes through the adapter: renames and the
    draft contract are applied before the toolkit op sees the kwargs."""
    metadata_op = mock.Mock(return_value=torch.empty(0))
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(mock.Mock(), metadata_op),
    ):
        result = build_dspark_sparse_flash_mla_metadata(
            num_heads_q=8,
            head_dim=576,
            seqused_q=torch.zeros(2, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            seq_lens=torch.tensor([10, 14], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_kv=14,
            ori_topk_length=torch.ones(6, 1, dtype=torch.int32),
            ori_topk=14,
        )
    assert result is metadata_op.return_value
    kwargs = metadata_op.call_args.kwargs
    assert kwargs["num_heads_q"] == 8
    assert kwargs["num_heads_kv"] == 1
    assert kwargs["head_dim"] == 576
    assert kwargs["batch_size"] == 2
    assert kwargs["max_seqlen_ori_kv"] == 14
    assert kwargs["layout_kv"] == "PA_BBND"
    assert kwargs["cmp_ratio"] == 1
    assert kwargs["ori_mask_mode"] == kwargs["cmp_mask_mode"] == 0
    assert kwargs["ori_win_left"] == kwargs["ori_win_right"] == -1
    assert "seqused_kv" not in kwargs
    assert "device" not in kwargs


def test_write_dspark_topk_lengths_adapts_shape():
    lens = torch.tensor([6, 6, 6], dtype=torch.int32)
    out = torch.zeros((3, 1), dtype=torch.int32)
    assert write_dspark_topk_lengths(lens, out) is out
    assert out.tolist() == [[6], [6], [6]]
    with pytest.raises(ValueError, match=r"num_tokens, 1"):
        write_dspark_topk_lengths(lens, torch.zeros((2, 1), dtype=torch.int32))
