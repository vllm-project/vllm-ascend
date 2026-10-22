# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check non-causal draft attention with the production FIA graph path."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState


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
@torch.inference_mode()
def test_non_causal_draft_fia_graph_matches_eager(monkeypatch, num_speculative_tokens):
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    query_len = num_speculative_tokens + 1
    head_size, num_heads, kv_len, block_size = 128, 2, 32, 128
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads = num_heads, 1
    impl.head_size, impl.scale = head_size, head_size**-0.5
    impl.sliding_window = 16
    impl.sinks = None
    impl.enable_c8_quant = False
    impl.use_bnsd_kv_cache = False
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "draft.attn"
    impl.key_cache = torch.zeros(1, block_size, 1, head_size, dtype=torch.bfloat16, device=device)
    impl.value_cache = torch.zeros_like(impl.key_cache)
    query = torch.zeros(query_len, num_heads, head_size, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)
    eager_output = torch.empty_like(query)
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding,
        num_actual_tokens=query_len,
        actual_seq_lengths_q=[query_len],
        seq_lens_list=[kv_len],
        block_tables=torch.zeros(1, 1, dtype=torch.int32, device=device),
        causal=False,
        attn_mask=None,
    )
    monkeypatch.setattr(_EXTRA_CTX, "is_draft_model", True)
    monkeypatch.setattr(_EXTRA_CTX, "capturing", False)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        impl.full_graph_fia(query, query, query, metadata, output)
        torch.npu.synchronize()
        # Capture the real FIA operator. Parameter-provider updates are covered
        # by the unit regression; here metadata stays fixed across replay.
        with torch.npu.graph(graph, stream=stream):
            impl.full_graph_fia(query, query, query, metadata, output)
        for offset in (0, 8):
            values = torch.arange(kv_len, dtype=query.dtype, device=device) + offset
            impl.value_cache[:, :kv_len].copy_(values.view(1, kv_len, 1, 1).expand(1, kv_len, 1, head_size))
            graph.replay()
            impl.forward_fused_infer_attention(query, query, query, metadata, eager_output)
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), eager_output.cpu(), rtol=0.01, atol=0.01)
            # Zero Q/K implies uniform attention over all 32 KV tokens. A
            # causal or sliding-window fallback gives a different mean per row.
            expected = torch.full_like(output.cpu(), (kv_len - 1) / 2 + offset)
            torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.01)
