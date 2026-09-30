import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sfa_split_kv import sparse_flash_attention_split_kv
from vllm_ascend.utils import enable_custom_op


@pytest.mark.skipif("950" not in torch.npu.get_device_name(0), reason="A5-only SFA Split-KV regression")
@pytest.mark.parametrize("query_count", [1, 4])
@pytest.mark.parametrize("sort_indices", [False, True])
def test_sfa_split_kv_matches_unsplit(query_count, sort_indices):
    enable_custom_op()
    torch.manual_seed(20260923 + query_count + int(sort_indices))
    heads, kv_length, capacity = 8, 8192, 2048
    query = torch.randn(query_count, heads, 512, dtype=torch.bfloat16).npu()
    query_rope = torch.randn(query_count, heads, 64, dtype=torch.bfloat16).npu()
    key = torch.randn(kv_length, 512, dtype=torch.bfloat16)
    key_rope = torch.randn(kv_length, 64, dtype=torch.bfloat16)
    pages = key.reshape(64, 128, 1, 512).npu()
    rope_pages = key_rope.reshape(64, 128, 1, 64).npu()
    indices = torch.stack([torch.randperm(kv_length)[:capacity] for _ in range(query_count)])
    if sort_indices:
        indices = indices.sort(dim=-1).values
    indices = indices.unsqueeze(1).to(torch.int32).npu()
    block_table = torch.arange(64, dtype=torch.int32).reshape(1, -1).npu()
    query_lengths = torch.tensor([query_count], dtype=torch.int32).npu()
    kv_lengths = torch.tensor([kv_length], dtype=torch.int32).npu()
    scale = 1 / math.sqrt(576)

    expected = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query,
        key=pages,
        value=pages,
        sparse_indices=indices,
        scale_value=scale,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=query_lengths,
        actual_seq_lengths_kv=kv_lengths,
        query_rope=query_rope,
        key_rope=rope_pages,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    actual = sparse_flash_attention_split_kv(
        query=query,
        key=pages,
        value=pages,
        sparse_indices=indices,
        scale_value=scale,
        block_table=block_table,
        actual_seq_lengths_query=query_lengths,
        actual_seq_lengths_kv=kv_lengths,
        query_rope=query_rope,
        key_rope=rope_pages,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(actual[0], expected[0], atol=0.003, rtol=0.01)
    actual_lse = actual[1] + actual[2].log()
    expected_lse = expected[1] + expected[2].log()
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.005, rtol=0.001)
