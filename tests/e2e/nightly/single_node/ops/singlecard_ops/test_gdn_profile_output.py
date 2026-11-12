# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


@torch.inference_mode()
def test_gdn_profile_output_is_initialized_on_graph_replay():
    output = torch.full((16, 32, 128), float("nan"), device="npu", dtype=torch.bfloat16)
    with patch("vllm_ascend.ops.gdn.get_forward_context", return_value=SimpleNamespace(attn_metadata=None)):
        for _ in range(3):
            AscendGatedDeltaNetAttention._forward_core(None, None, None, None, output)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            AscendGatedDeltaNetAttention._forward_core(None, None, None, None, output)
        for _ in range(2):
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
