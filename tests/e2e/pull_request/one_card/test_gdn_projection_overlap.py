# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_ascend.ops import gdn_projection
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod


@torch.inference_mode()
def test_projection_overlap_aclgraph_matches_serial(monkeypatch):
    class Projection(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(width, 128, device="npu", dtype=torch.bfloat16))
            self.quant_method = AscendUnquantizedLinearMethod()
            self.stream = None

        def forward(self, inputs):
            self.stream = torch.npu.current_stream()
            return self.quant_method.apply(self, inputs), None

    monkeypatch.setattr(
        gdn_projection, "get_ascend_config", lambda: SimpleNamespace(gdn_projection_overlap_max_tokens=8)
    )
    layer = SimpleNamespace(
        in_proj_qkvz=Projection(512),
        in_proj_ba=Projection(16),
    )
    config = SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(use_ubatching=False, prefill_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            static_forward_context={"gdn": layer},
        ),
    )
    gdn_projection.initialize_gdn_projection_overlap(layer, config)
    graphs = []
    for rows in (4, 8):
        inputs = torch.randn(rows, 128, device="npu", dtype=torch.bfloat16)
        gdn_projection.project_gdn_inputs(layer, inputs + 1)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            qkvz, ba = gdn_projection.project_gdn_inputs(layer, inputs + 1)
            # Consume both outputs on the capture stream, as GDN does.
            outputs = (qkvz.clone(), ba.clone())
        assert layer.in_proj_ba.stream == layer._ascend_ba_stream
        assert layer.in_proj_qkvz.stream != layer._ascend_ba_stream
        graphs.append((graph, inputs, outputs))

    # Alternate capture buckets to exercise event reuse across graph replays.
    for _ in range(3):
        for graph, inputs, (qkvz, ba) in graphs:
            inputs.normal_()
            graph.replay()
            torch.testing.assert_close(qkvz, layer.in_proj_qkvz(inputs + 1)[0], rtol=0, atol=0)
            torch.testing.assert_close(ba, layer.in_proj_ba(inputs + 1)[0], rtol=0, atol=0)
