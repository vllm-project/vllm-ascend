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
