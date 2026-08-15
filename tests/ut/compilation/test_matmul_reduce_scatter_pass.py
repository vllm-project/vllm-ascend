from unittest.mock import MagicMock

import torch
from torch import fx

from vllm_ascend.compilation.passes.matmul_reduce_scatter_pass import MatmulReduceScatterFusionPass
from vllm_ascend.ops import register_custom_ops  # noqa: F401


def test_matmul_reduce_scatter_pass_keeps_layer_wrapper():
    graph = fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.vllm.matmul_and_reduce.default, args=(x, "layer"))
    graph.output(node)

    config = MagicMock()
    config.model_config.dtype = torch.bfloat16

    MatmulReduceScatterFusionPass(config)(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets == [torch.ops.vllm.matmul_and_reduce.default]
