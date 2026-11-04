# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Compare eager and captured DFlash2 FlashAttn with explicit matmul/softmax."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.fa_v1 import (
    AscendFlashAttentionBackend,
    AscendFlashAttentionImpl,
    AscendFlashAttentionMetadataBuilder,
)
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import _selector_walk_kernel_ascend


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
@pytest.mark.parametrize("graph_mode", [False, True])
def test_strided_paged_flash_attn_matches_golden(dtype, window, layout, causal, head_dim, graph_mode):
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
    if graph_mode:
        torch.npu.synchronize()
        graph = UpdatableGraph()
        with patch.object(_EXTRA_CTX, "capturing", True), torch.npu.graph(graph):
            result = run()
        # Device-length attention must not register FIA host update tasks.
        assert not graph.tasks

    def execute():
        if graph_mode:
            graph.replay()
            torch.npu.synchronize()
            return result
        return run()

    result = execute()
    reference = golden(
        query, key[..., :head_dim], value[..., :head_dim], block_table, [4096, 3070], window, query_len, causal
    )
    torch.testing.assert_close(result[:12], reference, rtol=1e-2, atol=1e-3)
    rejected.copy_(torch.tensor([3, 5, 0], dtype=torch.int32, device=device))
    result = execute()
    reference = golden(
        query, key[..., :head_dim], value[..., :head_dim], block_table, [4093, 3067], window, query_len, causal
    )
    torch.testing.assert_close(result[:12], reference, rtol=1e-2, atol=1e-3)

    # Reorder requests without recapturing, then shrink to one real request.
    block_table[:2].copy_(block_table[:2].flip(0))
    optimistic.copy_(torch.tensor([3072, 4096, 0], dtype=torch.int32, device=device))
    rejected.copy_(torch.tensor([5, 3, 0], dtype=torch.int32, device=device))
    result = execute()
    reference = golden(
        query, key[..., :head_dim], value[..., :head_dim], block_table, [3067, 4093], window, query_len, causal
    )
    torch.testing.assert_close(result[:12], reference, rtol=1e-2, atol=1e-3)
    common.query_start_loc.copy_(torch.tensor([0, 6, 6, 6], dtype=torch.int32, device=device))
    optimistic[1] = 0
    rejected[1] = 0
    result = execute()
    reference = golden(
        query, key[..., :head_dim], value[..., :head_dim], block_table, [3067], window, query_len, causal
    )
    torch.testing.assert_close(result[:6], reference, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("probabilistic", [False, True])
def test_selector_walk_graph_replay_matches_eager(probabilistic):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    device = "npu"
    num_reqs, num_steps, top_k = 3, 5, 4
    scores = torch.randn(num_reqs * num_steps, top_k, top_k, device=device)
    candidates = torch.arange(num_reqs * num_steps * top_k, device=device, dtype=torch.int64)
    positions = torch.arange(num_reqs * num_steps, device=device, dtype=torch.int64) + 1
    mapping = torch.tensor([0] * num_steps + [1] * num_steps + [-1] * num_steps, device=device, dtype=torch.int64)
    temperatures = torch.tensor([0.7, 0.9, 1.0], device=device)
    seeds = torch.tensor([123, 456, 789], device=device, dtype=torch.int64)
    tokens = torch.full((num_reqs * num_steps,), -123, device=device, dtype=torch.int64)
    realized = torch.full((num_reqs * num_steps, top_k), -777.0, device=device)

    def run(out_tokens, out_scores):
        _selector_walk_kernel_ascend[(num_reqs,)](
            scores,
            candidates,
            positions,
            mapping,
            temperatures,
            seeds,
            out_tokens,
            out_scores,
            num_steps,
            top_k,
            top_k,
            probabilistic,
            False,
        )

    run(tokens, realized)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run(tokens, realized)

    for step in range(2):
        scores.copy_(torch.randn_like(scores))
        positions.add_(num_steps)
        if step:
            mapping[:num_steps].fill_(1)
            mapping[num_steps : 2 * num_steps].fill_(0)
        expected_tokens = torch.full_like(tokens, -123)
        expected_scores = torch.full_like(realized, -777.0)
        run(expected_tokens, expected_scores)
        tokens.fill_(-123)
        realized.fill_(-777.0)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(tokens, expected_tokens)
        torch.testing.assert_close(realized, expected_scores)
