"""Check FIA's zero-copy block-strided cache views on real NPU kernels."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.fia_kv_layout import fia_kv_cache_views
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import UpdatableGraph


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


@pytest.mark.parametrize("gap", [2, 4])
def test_fia_full_graph_replays_with_different_block_ids(gap):
    """One FULL capture must rebind both layers to each step's remapped table."""
    torch.manual_seed(17)
    device = "npu:0"
    blocks, block_size, head_size, query_heads = 6, 128, 256, 4
    query = torch.randn(2, query_heads, head_size, device=device, dtype=torch.bfloat16)
    layers, outputs = {}, {}
    for layer_name in ("first", "second"):
        backing = torch.randn(blocks, gap, block_size, 1, head_size, device=device, dtype=query.dtype)
        impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
        impl.key_cache, impl.value_cache = backing[:, 0], backing[:, 1]
        impl.use_bnsd_kv_cache = False
        impl.enable_c8_quant = False
        impl.sliding_window = None
        impl.num_heads = query_heads
        impl.num_kv_heads = 1
        impl.scale = head_size**-0.5
        impl._layer_name = layer_name
        impl._use_max_workspace_for_fia_graph = True
        layers[layer_name] = SimpleNamespace(impl=impl, kv_cache=(impl.key_cache, impl.value_cache))
        outputs[layer_name] = torch.empty_like(query)

    def make_metadata(block_ids):
        return AscendMetadata(
            block_tables=torch.tensor(block_ids, dtype=torch.int32, device=device),
            actual_seq_lengths_q=[1, 2],
            seq_lens_list=[257, 141],
            num_actual_tokens=2,
            attn_state=AscendAttentionState.DecodeOnly,
            causal=False,
        )

    metadata = make_metadata([[0, 1, 2], [3, 4, 5]])
    builder = AscendAttentionMetadataBuilder.__new__(AscendAttentionMetadataBuilder)
    builder.layer_names = list(layers)
    builder.compilation_config = SimpleNamespace(static_forward_context=layers)
    builder._prepare_fia_block_tables(metadata)
    assert gap in metadata.fia_block_tables
    capture_context = ForwardContext(
        no_compile_layers=layers,
        attn_metadata={name: metadata for name in layers},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )

    def forward():
        with override_forward_context(capture_context):
            for layer_name, layer in layers.items():
                layer.impl.full_graph_fia(query, query, query, metadata, outputs[layer_name])

    # Warm up the actual backend path before recording its updatable FIA tasks.
    forward()
    torch.npu.synchronize()
    graph = UpdatableGraph()
    with torch.npu.graph(graph):
        forward()
    assert len(graph.tasks) == len(layers)
    replay_context = SimpleNamespace(update_stream=torch.npu.Stream(), enable_enpu=False)
    previous_outputs = None
    for block_ids in ([[5, 1, 3], [2, 4, 0]], [[1, 0, 4], [5, 3, 2]]):
        next_metadata = make_metadata(block_ids)
        context = ForwardContext(
            no_compile_layers=layers,
            attn_metadata={name: next_metadata for name in layers},
            slot_mapping={},
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        )
        torch.npu.current_stream().synchronize()
        # Use the FULL graph replay path, including provider resolution, task
        # rebinding and the update-stream event protocol used in production.
        with override_forward_context(context):
            ACLGraphWrapper._updatable_graph_replay(replay_context, context, graph)
        torch.npu.synchronize()
        for layer_name, layer in layers.items():
            key = layer.impl.key_cache.flatten(2).contiguous()
            value = layer.impl.value_cache.flatten(2).contiguous()
            expected, _ = torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                block_table=next_metadata.block_tables,
                num_heads=query_heads,
                num_key_value_heads=1,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=[1, 2],
                actual_seq_lengths_kv=[257, 141],
                scale=head_size**-0.5,
                sparse_mode=0,
            )
            torch.testing.assert_close(outputs[layer_name], expected, rtol=0.01, atol=0.01)
            if previous_outputs is not None:
                assert not torch.allclose(outputs[layer_name], previous_outputs[layer_name], rtol=0.01, atol=0.01)
        previous_outputs = {name: output.clone() for name, output in outputs.items()}
