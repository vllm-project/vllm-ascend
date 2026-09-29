# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


def test_band_visible_lengths_require_one_row_per_query():
    enable_custom_op()
    num_tokens, num_heads, head_dim, block_size = 2, 16, 512, 128
    band_mask_mode = 4
    query = torch.zeros((num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device="npu")
    kv = torch.ones((1, block_size, 1, head_dim), dtype=torch.bfloat16, device="npu")
    starts = torch.tensor([0, num_tokens], dtype=torch.int32, device="npu")
    lengths = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    visible = torch.ones((num_tokens, 1), dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        num_heads,
        1,
        head_dim,
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        batch_size=1,
        max_seqlen_q=num_tokens,
        max_seqlen_ori_kv=num_tokens,
        ori_mask_mode=band_mask_mode,
        ori_win_left=block_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        sinks=torch.zeros(num_heads, dtype=torch.float32, device="npu"),
        metadata=metadata,
        softmax_scale=head_dim**-0.5,
        ori_mask_mode=band_mask_mode,
        ori_win_left=block_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(query, ori_topk_length=visible, **kwargs)
    torch.npu.synchronize()
    assert output.shape == query.shape
    assert torch.isfinite(output).all()

    with pytest.raises(RuntimeError, match="TND band ori_topk_length shape must be"):
        torch.ops._C_ascend.npu_sparse_flash_mla(query, ori_topk_length=visible[:1], **kwargs)
        torch.npu.synchronize()
