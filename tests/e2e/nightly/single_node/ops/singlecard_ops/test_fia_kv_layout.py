"""Check FIA's zero-copy block-strided cache views on real NPU kernels."""

import pytest
import torch
import torch_npu

from vllm_ascend.attention.fia_kv_layout import fia_kv_cache_views


@pytest.mark.parametrize("gap", [2, 4])
@pytest.mark.parametrize("query_lengths", [[1, 1], [3, 2]])
def test_fia_strided_cache_matches_contiguous_cache(gap, query_lengths):
    torch.manual_seed(17)
    device = "npu:0"
    blocks, block_size, head_size, query_heads = 6, 128, 256, 4
    backing = torch.randn(blocks, gap, block_size, head_size, device=device, dtype=torch.bfloat16)
    key, value = backing[:, 0], backing[:, 1]
    dense_key, dense_value, multiplier = fia_kv_cache_views(key, value)
    assert dense_key.is_contiguous() and dense_value.is_contiguous()
    assert dense_key.data_ptr() == key.data_ptr()
    query = torch.randn(sum(query_lengths), query_heads, head_size, device=device, dtype=torch.bfloat16)
    table = torch.tensor([[5, 1, 3], [2, 4, 0]], dtype=torch.int32, device=device)
    seq_lens = [257, 141]
    q_ends = [query_lengths[0], sum(query_lengths)]
    kwargs = dict(
        num_heads=query_heads,
        num_key_value_heads=1,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=q_ends,
        actual_seq_lengths_kv=seq_lens,
        scale=head_size**-0.5,
        sparse_mode=0,
    )
    expected, _ = torch_npu.npu_fused_infer_attention_score(
        query, key.contiguous(), value.contiguous(), block_table=table, **kwargs
    )
    actual, _ = torch_npu.npu_fused_infer_attention_score(
        query, dense_key, dense_value, block_table=table * multiplier, **kwargs
    )
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    # Update the final page after constructing the aliases, including a short
    # last sequence block; no stale contiguous copy may survive the update.
    key[5].fill_(0.25)
    value[5].fill_(0.75)
    expected, _ = torch_npu.npu_fused_infer_attention_score(
        query, key.contiguous(), value.contiguous(), block_table=table, **kwargs
    )
    actual, _ = torch_npu.npu_fused_infer_attention_score(
        query, dense_key, dense_value, block_table=table * multiplier, **kwargs
    )
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
