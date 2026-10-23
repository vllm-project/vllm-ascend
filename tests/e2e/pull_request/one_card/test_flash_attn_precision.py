# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Compare eager and captured draft FlashAttn with explicit matmul/softmax."""

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
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "window", "query_len"),
    [(8, 1, None, 6), (8, 1, 2048, 6), (8, 2, 1024, 8), (4, 1, 1024, 8)],
)
@pytest.mark.parametrize("layout", ["PA_BBND", "PA_BNBD"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_strided_paged_flash_attn_matches_golden(
    dtype, num_heads, num_kv_heads, window, query_len, layout, causal, capture
):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    device = "npu"
    block_size, head_dim = 128, 128
    num_blocks = 56
    page_stride = 2 * block_size * num_kv_heads * head_dim + 128
    backing = torch.full((num_blocks, page_stride), float("nan"), dtype=dtype, device=device)
    # Keep the inner two dimensions contiguous in each native CANN layout.
    strides = (
        (page_stride, num_kv_heads * head_dim, head_dim, 1)
        if layout == "PA_BBND"
        else (page_stride, head_dim, block_size * head_dim, 1)
    )
    key = backing.as_strided((num_blocks, block_size, num_kv_heads, head_dim), strides)
    value = backing.as_strided(key.shape, key.stride(), storage_offset=block_size * num_kv_heads * head_dim)
    key.copy_(torch.randn(key.shape, dtype=key.dtype, device=device))
    value.copy_(torch.randn(value.shape, dtype=value.dtype, device=device))
    query = torch.randn(3 * query_len, num_heads, head_dim, dtype=key.dtype, device=device)
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
        seq_lens=optimistic - rejected,
        num_reqs=3,
        num_actual_tokens=query.shape[0],
        query_start_loc=torch.tensor([0, query_len, 2 * query_len, 2 * query_len], dtype=torch.int32, device=device),
        block_table_tensor=block_table,
        slot_mapping=torch.arange(query.shape[0], dtype=torch.int32, device=device),
        max_query_len=query_len,
        causal=causal,
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    k_view, v_view = (key, value) if layout == "PA_BBND" else (key.transpose(1, 2), value.transpose(1, 2))
    impl = AscendFlashAttentionImpl.__new__(AscendFlashAttentionImpl)
    impl.key_cache, impl.value_cache = k_view, v_view
    impl.num_heads, impl.num_kv_heads = num_heads, num_kv_heads
    impl.scale, impl.sliding_window = head_dim**-0.5, window
    impl.use_bnsd_kv_cache, impl.sinks = layout == "PA_BNBD", None
    output = torch.empty_like(query)

    def run():
        metadata = builder.build(0, common)
        for _ in range(6):
            result = impl.forward_impl(query, None, None, (k_view, v_view), metadata, output)
        return result

    replay = run
    if capture:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured_result = run()

        def replay():
            graph.replay()
            return captured_result

    result = replay()
    reference = golden(query, key, value, block_table, [4096, 3070], window, query_len, causal)
    torch.testing.assert_close(result[: 2 * query_len], reference, rtol=1e-2, atol=1e-3)
    rejected.copy_(torch.tensor([3, 5, 0], dtype=torch.int32, device=device))
    common.seq_lens.copy_(optimistic - rejected)
    result = replay()
    reference = golden(query, key, value, block_table, [4093, 3067], window, query_len, causal)
    torch.testing.assert_close(result[: 2 * query_len], reference, rtol=1e-2, atol=1e-3)

    # Reuse the graph while changing page assignments, Q/K/V and active rows.
    block_table[0].copy_(block_table[0].roll(1))
    block_table[2, :4].copy_(block_table[1, :4])
    common.query_start_loc.copy_(torch.arange(4, dtype=torch.int32, device=device) * query_len)
    common.seq_lens.copy_(torch.tensor([3073, 2049, 385], dtype=torch.int32, device=device))
    query.copy_(torch.randn_like(query))
    key.mul_(0.5)
    value.mul_(0.5)
    result = replay()
    reference = golden(query, key, value, block_table, [3073, 2049, 385], window, query_len, causal)
    torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-3)

    common.query_start_loc.copy_(torch.tensor([0, query_len, query_len, query_len], dtype=torch.int32, device=device))
    common.seq_lens.copy_(torch.tensor([1025, 0, 0], dtype=torch.int32, device=device))
    result = replay()
    reference = golden(query, key, value, block_table, [1025], window, query_len, causal)
    torch.testing.assert_close(result[:query_len], reference, rtol=1e-2, atol=1e-3)
