# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops import gdn_projection


def test_projection_overlap_aclgraph_matches_serial(monkeypatch):
    class Projection(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(width, 128, device="npu", dtype=torch.bfloat16))
            self.quant_method = UnquantizedLinearMethod()

        def forward(self, inputs):
            return torch.nn.functional.linear(inputs, self.weight), None

    monkeypatch.setattr(
        gdn_projection, "get_ascend_config", lambda: SimpleNamespace(gdn_projection_overlap_max_tokens=8)
    )
    layer = SimpleNamespace(
        in_proj_qkvz=Projection(512),
        in_proj_ba=Projection(16),
        vllm_config=SimpleNamespace(
            lora_config=None,
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            parallel_config=SimpleNamespace(use_ubatching=False),
        ),
    )
    inputs = torch.randn(4, 128, device="npu", dtype=torch.bfloat16)
    gdn_projection.initialize_gdn_projection_overlap(layer)
    for _ in range(3):
        gdn_projection.project_gdn_inputs(layer, inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        qkvz, ba = gdn_projection.project_gdn_inputs(layer, inputs)
    for _ in range(3):
        inputs.normal_()
        graph.replay()
        torch.testing.assert_close(qkvz, layer.in_proj_qkvz(inputs)[0], rtol=0, atol=0)
        torch.testing.assert_close(ba, layer.in_proj_ba(inputs)[0], rtol=0, atol=0)
