# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Model-free C8 numerical regressions using real FIA and ACL graph replay.

The standard D128 kernel is intentional: these tests need neither a private
quantized checkpoint nor the separate D256 custom operator used by Qwen3.8.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

import vllm_ascend.attention.attention_v1 as attention_module
import vllm_ascend.compilation.acl_graph as acl_graph
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AscendMetadata,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an Ascend NPU")

BLOCK_SIZE = 128
HEAD_SIZE = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2


def _pack_nz(tensor):
    return (
        tensor.reshape(8, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // 32, 32)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(tensor.shape)
        .npu()
    )


def _make_layer(seed, name):
    generator = torch.Generator().manual_seed(seed)
    key = torch.randint(-20, 21, (8, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), generator=generator, dtype=torch.int8)
    value = torch.randint(-20, 21, key.shape, generator=generator, dtype=torch.int8)
    scale = torch.full((1, NUM_KV_HEADS, 1, HEAD_SIZE), 0.05, dtype=torch.bfloat16)

    impl = AscendC8AttentionBackendImpl.__new__(AscendC8AttentionBackendImpl)
    impl.num_heads = NUM_HEADS
    impl.num_kv_heads = NUM_KV_HEADS
    impl.head_size = HEAD_SIZE
    impl.scale = HEAD_SIZE**-0.5
    impl.key_cache = _pack_nz(key)
    impl.value_cache = _pack_nz(value)
    impl.enable_c8_quant = True
    impl.sliding_window = None
    impl.kv_sharing_target_layer_name = None
    impl._use_layer_aware_fia_graph_replay = False
    impl._use_max_workspace_for_fia_graph = False
    layer = SimpleNamespace(
        layer_name=name,
        _c8_k_aq_scale_nz_bnsd=scale.squeeze(0).npu(),
        _c8_v_aq_scale_nz_bnsd=scale.squeeze(0).npu(),
    )
    return impl, layer, key, value, scale


def _metadata(query_lens, kv_lens, tables):
    return AscendMetadata(
        attn_state=AscendAttentionState.SpecDecoding,
        actual_seq_lengths_q=torch.tensor(query_lens).cumsum(0).tolist(),
        seq_lens_list=kv_lens,
        block_tables=torch.tensor(tables, dtype=torch.int32, device="npu"),
        num_actual_tokens=sum(query_lens),
        num_decode_tokens=sum(query_lens),
        num_decodes=len(query_lens),
    )


def _reference(query, key, value, scale, query_lens, kv_lens, tables):
    result = []
    start = 0
    channel_scale = scale.reshape(NUM_KV_HEADS, HEAD_SIZE).float()
    for q_len, kv_len, pages in zip(query_lens, kv_lens, tables):
        keys = key[pages].reshape(-1, NUM_KV_HEADS, HEAD_SIZE).float() * channel_scale
        values = value[pages].reshape(-1, NUM_KV_HEADS, HEAD_SIZE).float() * channel_scale
        keys = keys.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        values = values.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        for index in range(q_len):
            visible = kv_len - q_len + index + 1
            scores = torch.einsum("hd,thd->ht", query[start + index].float(), keys[:visible]) * HEAD_SIZE**-0.5
            result.append(torch.einsum("ht,thd->hd", scores.softmax(-1), values[:visible]))
        start += q_len
    return torch.stack(result)


@pytest.mark.parametrize("query_lens,kv_lens", [([1], [130]), ([8], [136]), ([1, 3], [130, 250])])
def test_c8_eager_matches_per_query_causal_reference(query_lens, kv_lens):
    impl, layer, key, value, scale = _make_layer(42, "model.layers.0.self_attn.attn")
    tables = [[4, 1], [6, 3]][: len(query_lens)]
    generator = torch.Generator().manual_seed(24)
    query = torch.randn(sum(query_lens), NUM_HEADS, HEAD_SIZE, generator=generator, dtype=torch.bfloat16)
    metadata = _metadata(query_lens, kv_lens, tables)
    result = impl._forward_c8_decode(query.npu(), metadata, torch.empty_like(query, device="npu"), layer)
    expected = _reference(query, key, value, scale, query_lens, kv_lens, tables)
    torch.testing.assert_close(result.cpu().float(), expected, atol=2e-3, rtol=2e-2)


def test_c8_does_not_expose_unverified_future_values():
    impl, layer, _, value, _ = _make_layer(42, "model.layers.0.self_attn.attn")
    impl.key_cache.zero_()
    value.zero_()
    # Pages [4, 1], final length 136: the first query may see indices 0..128.
    # Only the seven FUTURE values at logical indices 129..135 are nonzero.
    value[1, 1:8] = 100
    impl.value_cache.copy_(_pack_nz(value))
    query = torch.zeros(8, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="npu")
    metadata = _metadata([8], [136], [[4, 1]])
    result = impl._forward_c8_decode(query, metadata, torch.empty_like(query), layer).cpu()
    torch.testing.assert_close(result[0], torch.zeros_like(result[0]), atol=0, rtol=0)
    assert result[-1].float().min() > 0.2


def test_c8_full_replay_updates_two_layers_without_dummy_table_overwrite(monkeypatch):
    """Replay several lengths/tables through the actual capture/update methods."""
    config = VllmConfig()
    # Select the FIA speculative path without loading a checkpoint.
    config.speculative_config = SimpleNamespace(num_speculative_tokens=7)
    capture_sizes = [4, 8]
    graph_params = acl_graph.GraphParams(
        {size: [] for size in capture_sizes},
        {size: None for size in capture_sizes},
        {size: [] for size in capture_sizes},
        {size: [] for size in capture_sizes},
    )
    monkeypatch.setattr(acl_graph, "_graph_params", graph_params)
    monkeypatch.setattr(attention_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False, sinks=False))
    layers = [_make_layer(42 + i, f"model.layers.{i}.self_attn.attn") for i in range(2)]
    query = torch.zeros(max(capture_sizes), NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="npu")
    outputs = [torch.empty_like(query) for _ in layers]
    dummy_metadata = {size: _metadata([size], [size], [[0, 0]]) for size in capture_sizes}
    # Mixed FULL capture can use one token per dummy request, then replay a
    # multi-query request of the same total size. Ownership must not depend on
    # the dummy query lengths used at capture time.
    dummy_metadata[8] = _metadata([1] * 8, [1] * 8, [[0, 0]] * 8)
    for impl, *_ in layers:
        impl.prepare_graph_block_tables(dummy_metadata[8].block_tables, capture_sizes)

    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    with set_current_vllm_config(config), torch.npu.stream(capture_stream):
        # Warm up FIA independently; the capture below uses the implementation's
        # real graph-task/event/workspace registration, not a mocked kernel.
        for impl, layer, *_ in layers:
            impl._forward_c8_decode(query, dummy_metadata[8], torch.empty_like(query), layer)
        torch.npu.synchronize()
        graphs = {}
        for size in reversed(capture_sizes):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=capture_stream):
                for (impl, layer, *_), output in zip(layers, outputs):
                    impl.full_graph_fia(query[:size], None, None, dummy_metadata[size], output[:size], layer)
            graphs[size] = graph

    update_stream = torch.npu.Stream()
    generator = torch.Generator().manual_seed(123)
    with set_current_vllm_config(config):
        for iteration, num_tokens in enumerate((8, 4, 8)):
            query_cpu = torch.randn(query.shape, generator=generator, dtype=torch.bfloat16)
            query.copy_(query_cpu)
            tables = ([[4, 1]], [[6, 3]]) if iteration % 2 == 0 else ([[6, 3]], [[4, 1]])
            kv_lens = ([136 + iteration], [248 - iteration])
            # Reverse insertion order to exercise captured layer-name lookup.
            metadata = {layers[i][1].layer_name: _metadata([num_tokens], kv_lens[i], tables[i]) for i in (1, 0)}
            update_stream.wait_stream(torch.npu.current_stream())
            AscendAttentionBackendImpl.update_graph_params(
                update_stream, SimpleNamespace(attn_metadata=metadata), num_tokens, config
            )
            torch.npu.current_stream().wait_stream(update_stream)
            graphs[num_tokens].replay()
            torch.npu.synchronize()
            for i, (_, _, key, value, scale) in enumerate(layers):
                expected = _reference(query_cpu[:num_tokens], key, value, scale, [num_tokens], kv_lens[i], tables[i])
                torch.testing.assert_close(outputs[i][:num_tokens].cpu().float(), expected, atol=2e-3, rtol=2e-2)
