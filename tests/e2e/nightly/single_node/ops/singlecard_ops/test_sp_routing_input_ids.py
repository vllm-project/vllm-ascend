# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.models.common.ops import sequence_parallel as sp_module


@pytest.mark.parametrize("num_tokens", [1, 12, 16, 50, 4096])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sp_routing_ids_graph_refresh(monkeypatch, num_tokens, dtype):
    # The transformation is local. Exercise TP8's row layout on one NPU;
    # the distributed benchmark separately compares the removed collectives.
    monkeypatch.setattr(sp_module, "get_tensor_model_parallel_world_size", lambda: 8)
    cpu_ids = torch.arange(num_tokens, dtype=dtype) + 10
    cpu_ids[0] = -1
    input_ids = cpu_ids.npu()
    for _ in range(3):
        sp_module.prepare_sp_routing_input_ids(input_ids)
    torch.npu.synchronize()

    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        output = sp_module.prepare_sp_routing_input_ids(input_ids)

    for offset in (0, 100, 200):
        updated = cpu_ids + offset
        updated[0] = -1
        input_ids.copy_(updated)
        graph.replay()
        torch.npu.synchronize()
        padded = F.pad(updated.to(torch.int64), (0, (-num_tokens) % 8))
        expected = torch.where(padded == -1, 0, padded)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
