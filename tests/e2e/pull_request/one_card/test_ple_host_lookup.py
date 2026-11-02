# SPDX-License-Identifier: Apache-2.0
"""Embedding resource-lifecycle regression for PLE host offload teardown."""

from types import SimpleNamespace

import torch

from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer
from vllm_ascend.models.qwen4_exp.ple_embedding import AscendPLEHostEmbedding


def test_embedding_close_after_graph_releases_host_owner():
    torch.npu.set_device(0)
    embedding = object.__new__(AscendPLEHostEmbedding)
    torch.nn.Module.__init__(embedding)
    table = HostUvaBuffer((32, 160), torch.bfloat16, "npu:0")
    embedding._host_table = table
    embedding.weight = torch.nn.Parameter(table.tensor, requires_grad=False)
    embedding.embedding_dim = 160
    embedding.etp_data_parallel_size = embedding.tp_size = 1
    embedding.reduce_results = True
    embedding.shard_indices = SimpleNamespace(org_vocab_start_index=10, org_vocab_end_index=42)
    graph = torch.npu.NPUGraph()
    try:
        table.tensor.fill_(3)
        ids = torch.tensor([[10, 41, 42]], device="npu")
        for _ in range(3):
            embedding(ids)
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            output = embedding(ids)
        graph.replay()
        torch.testing.assert_close(output.cpu(), torch.tensor([[[3] * 160, [3] * 160, [0] * 160]]).bfloat16())
        torch.npu.synchronize()
    finally:
        graph.reset()
        embedding.close_host_offload()
        embedding.close_host_offload()
    assert embedding.weight is None and embedding._host_table is None
    assert table.tensor is None and table.ptrs is None and not table.pointer.value
