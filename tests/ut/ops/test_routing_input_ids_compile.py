# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.fused_moe.runner import moe_runner

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.router import fused_topk_router as router_module


@pytest.fixture
def cpu_moe_forward(monkeypatch):
    # CPU tests register an NPU stub, but Dynamo must not query real NPU streams.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    # vLLM registers on the selected platform: PrivateUse1 in CPU CI, CPU in
    # the local harness. Use the real op implementation for CPU in both cases.
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("moe_forward", moe_runner._moe_forward, allow_override=True)
    try:
        yield
    finally:
        library._destroy()


def test_compiled_moe_ops_refresh_shared_routing_ids(monkeypatch, cpu_moe_forward):
    """Keep the actual opaque MoE op boundary while replacing expert computation."""
    gather = MagicMock(side_effect=lambda ids: torch.cat((ids, ids.flip(0))))
    context = SimpleNamespace(
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(all_gather_input_ids=gather)),
    )
    monkeypatch.setattr(router_module, "_EXTRA_CTX", context)
    buffer = torch.empty(16, dtype=torch.int64)
    num_layers = 4

    class Layer:
        def __init__(self, index):
            self.router = router_module.AscendFusedTopKRouter(top_k=2, global_num_experts=4)
            self.router.routing_input_ids_buffer = buffer
            self.router.is_routing_input_ids_producer = index == 0

        def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids):
            ids = self.router._prepare_routing_input_ids(input_ids, hidden_states.shape[0])
            return hidden_states + ids.to(hidden_states.dtype).unsqueeze(1)

    layers = {f"routing_cache_layer_{i}": Layer(i) for i in range(num_layers)}
    monkeypatch.setattr(moe_runner, "get_layer_from_name", layers.__getitem__)

    def forward(hidden_states, input_ids):
        for i in range(num_layers):
            hidden_states = torch.ops.vllm.moe_forward(
                hidden_states, hidden_states, None, input_ids, f"routing_cache_layer_{i}", 0
            )
        return hidden_states

    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
    storage = torch.empty(8, dtype=torch.int32)
    for iteration, num_tokens in enumerate((3, 3, 5, 2, 3)):
        storage.copy_(torch.arange(8, dtype=torch.int32) + 10 * iteration)
        storage[0] = -1
        input_ids = storage[:num_tokens]
        hidden_states = torch.zeros(2 * num_tokens, 4)
        actual = compiled(hidden_states, input_ids)
        expected_ids = torch.cat((input_ids, input_ids.flip(0)))
        expected = expected_ids.float().unsqueeze(1).expand_as(hidden_states) * num_layers
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert gather.call_count == iteration + 1
    assert len(graphs) == 1
