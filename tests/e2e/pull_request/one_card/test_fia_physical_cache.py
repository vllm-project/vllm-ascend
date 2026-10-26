# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import FIAParamProvider
from vllm_ascend.attention.utils import fia_physical_block_table, fia_physical_kv_views
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph, register_task


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("query_len", [1, 9])
def test_fia_physical_pages_and_graph_updates(factor, head_size, query_len):
    """Change addresses and lengths between replays, with K/V sharing storage."""
    torch.manual_seed(14340)
    blocks, batch, block_size, heads, kv_heads = 8, 2, 128, 4, 2
    backing = torch.randn(blocks, factor, block_size, kv_heads * head_size, dtype=torch.bfloat16, device="npu")
    key, value = backing[:, 0], backing[:, 1]
    physical_key, physical_value, stride = fia_physical_kv_views(key, value)
    assert stride == factor and physical_key.is_contiguous() and physical_value.is_contiguous()
    query = torch.randn(batch * query_len, heads, head_size, dtype=torch.bfloat16, device="npu")
    table = torch.tensor([[3, 6], [1, 4]], dtype=torch.int32, device="npu")
    metadata = SimpleNamespace(
        block_tables=table, actual_seq_lengths_q=[query_len, 2 * query_len], seq_lens_list=[159, 201]
    )
    common = dict(
        query=query,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=metadata.actual_seq_lengths_q,
        actual_seq_lengths_kv=metadata.seq_lens_list,
        num_heads=heads,
        num_key_value_heads=kv_heads,
        scale=head_size**-0.5,
        sparse_mode=0,
    )
    physical = dict(
        common, key=physical_key, value=physical_value, block_table=fia_physical_block_table(metadata, stride)
    )
    expected, _ = torch_npu.npu_fused_infer_attention_score(
        **dict(common, key=key.contiguous(), value=value.contiguous(), block_table=table)
    )
    actual, _ = torch_npu.npu_fused_infer_attention_score(**physical)
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    output = torch.empty_like(actual)
    lse = torch.empty(1, dtype=query.dtype, device="npu")
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**physical)
    provider = FIAParamProvider("layer", None, block_stride=stride)
    graph = UpdatableGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        register_task(
            torch_npu.npu_fused_infer_attention_score.out,
            dict(physical, workspace=workspace, out=[output, lse]),
            provider,
        )
    update_stream = torch.npu.Stream()
    for ids, lengths in [([[3, 6], [1, 4]], [159, 201]), ([[7, 0], [2, 5]], [191, 137])]:
        current_table = torch.tensor(ids, dtype=torch.int32, device="npu")
        current = SimpleNamespace(
            block_tables=current_table, actual_seq_lengths_q=metadata.actual_seq_lengths_q, seq_lens_list=lengths
        )
        expected, _ = torch_npu.npu_fused_infer_attention_score(
            **dict(
                common,
                key=key.contiguous(),
                value=value.contiguous(),
                block_table=current_table,
                actual_seq_lengths_kv=lengths,
            )
        )
        tasks = graph.resolve_tasks(ContextSource({"layer": current}))
        update_stream.wait_stream(torch.npu.current_stream())
        graph.replay()
        graph.update(update_stream, tasks)
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.002, atol=0.002)
