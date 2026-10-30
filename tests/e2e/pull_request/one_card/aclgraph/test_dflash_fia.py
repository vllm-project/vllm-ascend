# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check non-causal draft attention with the production FIA graph path."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph


def _reference_attention(query, key_cache, value_cache, block_table, kv_lens, window):
    """Dense CPU reference with explicit right-aligned window visibility."""
    num_reqs = len(kv_lens)
    query_len = query.shape[0] // num_reqs
    query = query.float().cpu().reshape(num_reqs, query_len, *query.shape[1:])
    key_cache, value_cache = key_cache.float().cpu(), value_cache.float().cpu()
    rows = []
    for req_idx, kv_len in enumerate(kv_lens):
        keys = key_cache[block_table[req_idx]].flatten(0, 1)[:kv_len]
        values = value_cache[block_table[req_idx]].flatten(0, 1)[:kv_len]
        keys = keys.repeat_interleave(query.shape[2] // keys.shape[1], dim=1)
        values = values.repeat_interleave(query.shape[2] // values.shape[1], dim=1)
        scores = torch.einsum("qhd,khd->hqk", query[req_idx], keys) * query.shape[-1] ** -0.5
        if window is not None:
            q_positions = torch.arange(kv_len - query_len, kv_len)
            distance = torch.arange(kv_len)[None, :] - q_positions[:, None]
            scores.masked_fill_(distance.abs()[None, :, :] >= window, float("-inf"))
        rows.append(torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), values))
    return torch.cat(rows)


@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode,aclgraph",
    parallel="",
    deploy="",
    hardware="",
    quantization="",
    graph_mode="full_graph",
)
@pytest.mark.parametrize("num_speculative_tokens", [4, 5, 6])
@pytest.mark.parametrize("sliding_window", [None, 1, 3, 16, 256])
@pytest.mark.parametrize("random_qk", [False, True])
@torch.inference_mode()
def test_non_causal_draft_fia_graph_matches_eager(monkeypatch, num_speculative_tokens, sliding_window, random_qk):
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    query_len = num_speculative_tokens + 1
    head_size, num_heads, block_size, num_reqs = 128, 2, 128, 2
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads = num_heads, 1
    impl.head_size, impl.scale = head_size, head_size**-0.5
    impl.sliding_window = sliding_window
    impl.sinks = None
    impl.enable_c8_quant = False
    impl.use_bnsd_kv_cache = False
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "draft.attn"
    generator = torch.Generator().manual_seed(17)
    impl.key_cache = torch.randn(5, block_size, 1, head_size, generator=generator).to(device, torch.bfloat16)
    impl.value_cache = torch.zeros_like(impl.key_cache)
    query = torch.randn(num_reqs * query_len, num_heads, head_size, generator=generator).to(device, torch.bfloat16)
    if not random_qk:
        query.zero_()
        impl.key_cache.zero_()
    output = torch.empty_like(query)
    eager_output = torch.empty_like(query)
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding,
        num_actual_tokens=num_reqs * query_len,
        actual_seq_lengths_q=[query_len, num_reqs * query_len],
        seq_lens_list=[140, 72],
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device),
        causal=False,
        attn_mask=None,
    )
    monkeypatch.setattr(_EXTRA_CTX, "is_draft_model", True)
    monkeypatch.setattr(_EXTRA_CTX, "capturing", False)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = UpdatableGraph()
    update_stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        impl.full_graph_fia(query, query, query, metadata, output)
        torch.npu.synchronize()
        # Warmup also initializes the persistent compressed band mask.
        with torch.npu.graph(graph, stream=stream):
            impl.full_graph_fia(query, query, query, metadata, output)
        assert len(graph.tasks) == 1
        # Vary host lengths and physical blocks across replay, including a
        # window crossing the 128-token block boundary and different KV lengths.
        for offset, kv_lens, blocks in (
            (0, [129, 63], [[1, 2], [3, 4]]),
            (8, [137, 70], [[3, 4], [1, 2]]),
        ):
            values = torch.arange(5 * block_size, device=device) % 97 + offset
            impl.value_cache.copy_(values.view(5, block_size, 1, 1).expand_as(impl.value_cache))
            metadata.seq_lens_list = kv_lens
            metadata.block_tables = torch.tensor(blocks, dtype=torch.int32, device=device)
            update_stream.wait_stream(stream)
            graph.update(update_stream, graph.resolve_tasks(ContextSource({"draft.attn": metadata})))
            graph.replay()
            impl.forward_fused_infer_attention(query, query, query, metadata, eager_output)
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), eager_output.cpu(), rtol=0.01, atol=0.01)
            expected = _reference_attention(query, impl.key_cache, impl.value_cache, blocks, kv_lens, sliding_window)
            torch.testing.assert_close(output.float().cpu(), expected, rtol=0.01, atol=0.01)
