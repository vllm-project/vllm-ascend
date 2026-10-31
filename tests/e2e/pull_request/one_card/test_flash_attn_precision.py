# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Compare eager DFlash2 FlashAttn with explicit matmul/softmax."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

import vllm_ascend.attention.attention_v1 as attn_module
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState, _get_fia_sparse_params
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [4, 16])
def test_noncausal_fia_band_eager_and_graph_match_reference(dtype, window):
    """Keep future draft keys, but exclude KV outside the trained window."""
    torch.npu.set_device(0)
    torch.manual_seed(2026)
    head_dim, query_len, kv_len, block_size = 128, 9, 65, 128
    query = torch.randn(query_len, 2, head_dim, device="npu", dtype=dtype)
    key = torch.randn(1, block_size, 1, head_dim, device="npu", dtype=dtype)
    value = torch.randn_like(key)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device="npu")
    mask = AttentionMaskBuilder(torch.device("npu")).get_splitfuse_attn_mask()
    sparse_mode, pre_tokens, next_tokens = _get_fia_sparse_params(window, False)
    output = torch.empty_like(query)

    def forward():
        result, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            block_table=block_table,
            atten_mask=mask,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=[query_len],
            actual_seq_lengths_kv=[kv_len],
            num_heads=2,
            num_key_value_heads=1,
            scale=head_dim**-0.5,
        )
        output.copy_(result)

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    for _ in range(3):
        query.copy_(torch.randn_like(query))
        key.copy_(torch.randn_like(key))
        value.copy_(torch.randn_like(value))
        expected = golden(query, key, value, block_table, [kv_len], window, query_len, False)
        for execute in (forward, graph.replay):
            execute()
            torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("window", [None, 16])
def test_fia_graph_shared_band_mask_matches_layer_reference(window):
    """Exercise the actual graph backend with a merged group's shared mask."""
    torch.npu.set_device(0)
    torch.manual_seed(2026)
    query_len, kv_len, head_dim = 9, 65, 128
    query = torch.randn(query_len, 2, head_dim, device="npu", dtype=torch.bfloat16)
    key = torch.randn(1, 128, 1, head_dim, device="npu", dtype=query.dtype)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device="npu")
    shared_mask = AttentionMaskBuilder(torch.device("npu")).get_splitfuse_attn_mask()
    metadata = SimpleNamespace(
        causal=False,
        attn_mask=shared_mask,
        actual_seq_lengths_q=[query_len],
        seq_lens_list=[kv_len],
        block_tables=block_table,
        num_actual_tokens=query_len,
        attn_state=AscendAttentionState.DecodeOnly,
    )
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 2, 1, head_dim
    impl.scale = head_dim**-0.5
    impl.sliding_window = window
    impl.key_cache, impl.value_cache = key, value
    impl.use_bnsd_kv_cache = False
    impl.enable_c8_quant = False
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "draft.test"

    def forward():
        with patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=True)):
            impl.full_graph_fia(query, None, None, metadata, output)

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()
    for _ in range(3):
        query.copy_(torch.randn_like(query))
        key.copy_(torch.randn_like(key))
        value.copy_(torch.randn_like(value))
        expected = golden(query, key, value, block_table, [kv_len], window, query_len, False)
        for execute in (forward, graph.replay):
            execute()
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
            assert metadata.attn_mask is shared_mask
