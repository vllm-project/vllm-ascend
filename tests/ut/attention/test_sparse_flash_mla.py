# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import torch

from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata


def test_adapter_enforces_bf16_paged_layout():
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        seq_lens = torch.tensor([8, 12], dtype=torch.int32)
        query_offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
        kwargs = dict(
            layout_kv="PA_ND",
            cu_seqlens_q=query_offsets,
            cu_seqlens_ori_kv=query_offsets,
            cu_seqlens_cmp_kv=query_offsets,
            seqused_kv=seq_lens,
            cmp_ratio=4,
        )
        sparse_flash_mla_metadata(**kwargs)
        sparse_flash_mla(torch.empty(0), **kwargs)

    for op in (metadata_op, attention_op):
        actual = op.call_args.kwargs
        assert actual["layout_kv"] == "PA_BBND"
        assert "cu_seqlens_ori_kv" not in actual
        assert "cu_seqlens_cmp_kv" not in actual
        assert actual["cu_seqlens_q"] is query_offsets
        assert actual["seqused_ori_kv"] is seq_lens
        assert actual["seqused_cmp_kv"].tolist() == [2, 3]
        assert actual["cmp_residual_kv"].tolist() == [0, 0]
