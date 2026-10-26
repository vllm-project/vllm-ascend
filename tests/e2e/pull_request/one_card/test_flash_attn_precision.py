# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Compare eager DFlash2 FlashAttn with explicit matmul/softmax."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.fa_v1 import AscendFlashAttentionImpl, AscendFlashAttentionMetadataBuilder


def golden(query, key, value, block_table, lengths, window, query_len, causal):
    results = []
    offset = 0
    ratio = query.shape[1] // key.shape[2]
    for req, length in enumerate(lengths):
        size = query_len
        indices = block_table[req].long()
        k = key[indices].flatten(0, 1)[:length].float().repeat_interleave(ratio, dim=1)
        v = value[indices].flatten(0, 1)[:length].float().repeat_interleave(ratio, dim=1)
        q = query[offset : offset + size].float()
        offset += size
        scores = torch.matmul(q.transpose(0, 1), k.permute(1, 2, 0)) * query.shape[-1] ** -0.5
        if window is not None or causal:
            q_pos = torch.arange(size, device=query.device) + length - size
            kv_pos = torch.arange(length, device=query.device)
            delta = kv_pos[None, :] - q_pos[:, None]
            mask = delta.abs() >= window if window is not None else torch.zeros_like(delta, dtype=torch.bool)
            if causal:
                mask = mask | (delta > 0)
            scores = scores.masked_fill(mask, float("-inf"))
        results.append(torch.matmul(scores.softmax(-1), v.transpose(0, 1)).transpose(0, 1))
    return torch.cat(results).to(query.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [None, 2048])
@pytest.mark.parametrize("layout", ["PA_BBND", "PA_BNBD"])
@pytest.mark.parametrize("causal", [False, True])
def test_strided_paged_flash_attn_matches_golden(dtype, window, layout, causal):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    device = "npu"
    block_size, head_dim, query_len = 128, 128, 6
    num_blocks = 56
    page_stride = 2 * block_size * head_dim + 128
    backing = torch.full((num_blocks, page_stride), float("nan"), dtype=dtype, device=device)
    key = backing.as_strided((num_blocks, block_size, 1, head_dim), (page_stride, head_dim, head_dim, 1))
    value = backing.as_strided(key.shape, key.stride(), storage_offset=block_size * head_dim)
    key.copy_(torch.randn(key.shape, dtype=key.dtype, device=device))
    value.copy_(torch.randn(value.shape, dtype=value.dtype, device=device))
    query = torch.randn(3 * query_len, 8, head_dim, dtype=key.dtype, device=device)
    permutation = torch.randperm(num_blocks, device=device).to(torch.int32)
    block_table = torch.zeros(3, 32, dtype=torch.int32, device=device)
    block_table[0].copy_(permutation[:32])
    block_table[1, :24].copy_(permutation[32:])
    optimistic = torch.tensor([4096, 3072, 0], dtype=torch.int32, device=device)
    rejected = torch.tensor([0, 2, 0], dtype=torch.int32, device=device)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1)
    builder = AscendFlashAttentionMetadataBuilder.__new__(AscendFlashAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(runner_type="draft")
    builder.attn_mask_builder = SimpleNamespace(
        get_splitfuse_attn_mask=lambda: mask,
    )
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=query.shape[0],
        query_start_loc=torch.tensor([0, 6, 12, 12], dtype=torch.int32, device=device),
        block_table_tensor=block_table,
        slot_mapping=torch.arange(query.shape[0], dtype=torch.int32, device=device),
        max_query_len=query_len,
        causal=causal,
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    k_view, v_view = (key, value) if layout == "PA_BBND" else (key.transpose(1, 2), value.transpose(1, 2))
    impl = AscendFlashAttentionImpl.__new__(AscendFlashAttentionImpl)
    impl.key_cache, impl.value_cache = k_view, v_view
    impl.num_heads, impl.num_kv_heads = 8, 1
    impl.scale, impl.sliding_window = head_dim**-0.5, window
    impl.use_bnsd_kv_cache, impl.sinks = layout == "PA_BNBD", None
    output = torch.empty_like(query)

    def run():
        common.seq_lens = optimistic - rejected
        metadata = builder.build(0, common)
        for _ in range(6):
            result = impl.forward_impl(query, None, None, (k_view, v_view), metadata, output)
        return result

    result = run()
    reference = golden(query, key, value, block_table, [4096, 3070], window, query_len, causal)
    torch.testing.assert_close(result[:12], reference, rtol=1e-2, atol=1e-3)
    rejected.copy_(torch.tensor([3, 5, 0], dtype=torch.int32, device=device))
    result = run()
    reference = golden(query, key, value, block_table, [4093, 3067], window, query_len, causal)
    torch.testing.assert_close(result[:12], reference, rtol=1e-2, atol=1e-3)
