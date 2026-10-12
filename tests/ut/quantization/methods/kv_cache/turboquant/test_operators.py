# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.attention.mixed_quant_sparse_flash_mla as adapter
from vllm_ascend.attention.mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)


def test_metadata_maps_paged_lengths_and_quant_mode():
    lengths = torch.tensor([7, 128, 513], dtype=torch.int32)
    op = Mock()
    with patch.object(adapter, "_get_mixed_quant_sparse_flash_mla_ops", return_value=(None, op)):
        mixed_quant_sparse_flash_mla_metadata(
            seqused_kv=lengths,
            max_seqlen_kv=513,
            cmp_ratio=4,
            cu_seqlens_ori_kv=torch.tensor([0, 1]),
            cu_seqlens_cmp_kv=torch.tensor([0, 1]),
            layout_kv="PA_ND",
            device="npu:0",
        )
    kwargs = op.call_args.kwargs
    torch.testing.assert_close(kwargs["seqused_cmp_kv"], torch.tensor([1, 32, 128], dtype=torch.int32))
    torch.testing.assert_close(kwargs["cmp_residual_kv"], torch.tensor([3, 0, 1], dtype=torch.int32))
    assert kwargs["max_seqlen_cmp_kv"] == 128
    assert kwargs["layout_kv"] == "PA_BBND"
    assert kwargs["quant_mode"] == 3
    assert kwargs["rope_head_dim"] == 64
    assert not {"device", "cu_seqlens_cmp_kv", "cu_seqlens_ori_kv", "seqused_kv"}.intersection(kwargs)


def test_attention_retains_caller_owned_cache_and_sinks():
    q = torch.zeros(1, 4, 512, dtype=torch.bfloat16)
    cache = torch.zeros(2, 32, 1, 258, dtype=torch.uint8)
    sinks = torch.zeros(4)
    op = Mock()
    with patch.object(adapter, "_get_mixed_quant_sparse_flash_mla_ops", return_value=(op, None)):
        mixed_quant_sparse_flash_mla(
            q, cmp_kv=cache, sinks=sinks, cmp_ratio=4, seqused_kv=torch.tensor([132], dtype=torch.int32)
        )
    assert "seqused_cmp_kv" not in op.call_args.kwargs
    assert "cmp_residual_kv" not in op.call_args.kwargs
    assert op.call_args.args[0] is q
    assert op.call_args.kwargs["cmp_kv"] is cache
    assert op.call_args.kwargs["sinks"] is sinks
    with pytest.raises(ValueError, match="caller-owned"):
        mixed_quant_sparse_flash_mla(q, sinks=None)


def test_attention_defaults_to_c4_compression_ratio():
    q = torch.zeros(1, 4, 512, dtype=torch.bfloat16)
    op = Mock()
    with patch.object(adapter, "_get_mixed_quant_sparse_flash_mla_ops", return_value=(op, None)):
        mixed_quant_sparse_flash_mla(q, cmp_kv=torch.zeros(1, 1, 1, 258, dtype=torch.uint8), sinks=torch.zeros(4))
    assert op.call_args.kwargs["cmp_ratio"] == 4
