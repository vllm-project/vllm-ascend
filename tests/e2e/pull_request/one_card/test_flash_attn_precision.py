# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Compare draft FlashAttn with explicit matmul/softmax on A5."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.fa_v1 import (
    AscendFlashAttentionBackend,
    AscendFlashAttentionImpl,
    AscendFlashAttentionMetadataBuilder,
)


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
@pytest.mark.parametrize("head_dim", [128, 192])
def test_strided_paged_flash_attn_matches_golden(dtype, window, layout, causal, head_dim):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    device = "npu"
    block_size, query_len = 128, 6
    physical_dim = AscendFlashAttentionBackend.get_padded_head_size(head_dim)
    num_blocks = 56
    page_stride = 2 * block_size * physical_dim + 128
    backing = torch.full((num_blocks, page_stride), float("nan"), dtype=dtype, device=device)
    key = backing.as_strided((num_blocks, block_size, 1, physical_dim), (page_stride, physical_dim, physical_dim, 1))
    value = backing.as_strided(key.shape, key.stride(), storage_offset=block_size * physical_dim)
    key.copy_(torch.randn(key.shape, dtype=key.dtype, device=device))
    value.copy_(torch.randn(value.shape, dtype=value.dtype, device=device))
    key[..., head_dim:] = 0
    value[..., head_dim:] = 0
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
    reference = golden(
        query, key[..., :head_dim], value[..., :head_dim], block_table, [4096, 3070], window, query_len, causal
    )
    torch.testing.assert_close(result[:12], reference, rtol=1e-2, atol=1e-3)
    rejected.copy_(torch.tensor([3, 5, 0], dtype=torch.int32, device=device))
    result = run()
    reference = golden(
        query, key[..., :head_dim], value[..., :head_dim], block_table, [4093, 3067], window, query_len, causal
    )
    torch.testing.assert_close(result[:12], reference, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("full_graph", [False, True])
def test_kimi_gqa_flash_attn_uses_updated_device_lengths(full_graph):
    """Check SWA boundaries and rejection updates with Kimi's GQA geometry."""
    torch.npu.set_device(0)
    torch.manual_seed(42)
    device, dtype = "npu", torch.bfloat16
    block_size, query_len, head_dim = 128, 3, 64
    num_heads, num_kv_heads, window = 12, 2, 2048
    # Three requests straddle the SWA boundary; one exercises 128K context.
    # The final row is graph padding with no query tokens.
    lengths = [2047, 2048, 2049, 131072]
    num_reqs = len(lengths)
    max_blocks = (max(lengths) + block_size - 1) // block_size
    num_blocks = sum((length + block_size - 1) // block_size for length in lengths)
    key = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)
    query = torch.randn((num_reqs + 1) * query_len, num_heads, head_dim, dtype=dtype, device=device)
    block_table = torch.zeros(num_reqs + 1, max_blocks, dtype=torch.int32, device=device)
    permutation = torch.randperm(num_blocks, device=device).to(torch.int32)
    offset = 0
    for req, length in enumerate(lengths):
        blocks = (length + block_size - 1) // block_size
        indices = permutation[offset : offset + blocks]
        block_table[req, :blocks].copy_(indices)
        if length == window + 1:
            # Uniform scores with large values at positions 0 and 1 expose a
            # one-token window error: the last query must exclude only pos 0.
            query[req * query_len : (req + 1) * query_len].zero_()
            key[indices.long()] = 0
            value[indices.long()] = 0
            value[indices[:1].long(), :, 0, 0] = 2 * window
            value[indices[:1].long(), :, 1, 1] = 2 * window
        offset += blocks
    optimistic = torch.tensor([*lengths, 0], dtype=torch.int32, device=device)
    rejected = torch.zeros_like(optimistic)
    mask = torch.triu(torch.ones(window, window, dtype=torch.int8, device=device), diagonal=1)
    builder = AscendFlashAttentionMetadataBuilder.__new__(AscendFlashAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(runner_type="draft")
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: mask)
    common = SimpleNamespace(
        num_reqs=num_reqs + 1,
        num_actual_tokens=query.shape[0],
        query_start_loc=torch.tensor([0, 3, 6, 9, 12, 12], dtype=torch.int32, device=device),
        block_table_tensor=block_table,
        slot_mapping=torch.arange(query.shape[0], dtype=torch.int32, device=device),
        max_query_len=query_len,
        causal=True,
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    impl = AscendFlashAttentionImpl.__new__(AscendFlashAttentionImpl)
    impl.key_cache, impl.value_cache = key, value
    impl.num_heads, impl.num_kv_heads = num_heads, num_kv_heads
    impl.scale, impl.sliding_window = head_dim**-0.5, window
    impl.use_bnsd_kv_cache, impl.sinks = True, None
    output = torch.empty_like(query)

    def run():
        # Capture preprocessing and metadata generation, not just FlashAttn.
        common.seq_lens = optimistic - rejected
        metadata = builder.build(0, common)
        for _ in range(5):
            impl.forward_impl(query, None, None, (key, value), metadata, output)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        run()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph() if full_graph else None
    if graph is not None:
        with torch.npu.graph(graph, stream=stream):
            run()

    for rejections in ([0, 0, 0, 0], [3, 2, 1, 3], [1, 0, 2, 1]):
        rejected.copy_(torch.tensor([*rejections, 0], dtype=torch.int32, device=device))
        if graph is None:
            run()
        else:
            graph.replay()
        reference = golden(
            query,
            key.transpose(1, 2),
            value.transpose(1, 2),
            block_table,
            [length - rejection for length, rejection in zip(lengths, rejections)],
            window,
            query_len,
            True,
        )
        torch.testing.assert_close(output[: num_reqs * query_len], reference, rtol=1e-2, atol=1e-3)
