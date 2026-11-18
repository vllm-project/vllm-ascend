"""Regress whole-capacity copies introduced by functionalizing MTP state writes."""

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm_ascend.models.deepseek_v41 import model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp_prefix_copy_survives_aot_without_copying_capacity(monkeypatch, dtype):
    capacity, hc_mult, hidden_size = 128, 4, 16
    sentinel = -37.0
    buffer_key = "target._mtp_hidden_buffer"
    buffer = torch.full((capacity, hc_mult * hidden_size), sentinel, dtype=dtype)
    pointer = buffer.data_ptr()
    context = SimpleNamespace(no_compile_layers={buffer_key: buffer})
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    graphs = []

    def compiler(graph, example_inputs):
        graphs.append(graph)
        return make_boxed_func(graph.forward)

    def forward(hidden_states, pre_mix):
        return torch.ops.vllm.ascend_dsv41_mtp_save_and_collapse(hidden_states, pre_mix, buffer_key)

    # The production kernel is registered for PrivateUse1. Exercise its actual
    # implementation through the same schema/fake kernel on the CPU AOT path.
    cpu_impl = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        cpu_impl.impl("ascend_dsv41_mtp_save_and_collapse", model._mtp_save_and_collapse)
        compiled = torch.compile(forward, backend=aot_autograd(fw_compiler=compiler), fullgraph=True, dynamic=True)
        for num_tokens in (0, 1, 24, 6, capacity, 12):
            buffer.fill_(sentinel)
            hidden_states = torch.randn(num_tokens, hc_mult, hidden_size, dtype=dtype)
            pre_mix = torch.randn(num_tokens, hc_mult)
            expected = model.DeepseekV41DecoderLayer.hc_collapse(hidden_states, pre_mix)

            actual = compiled(hidden_states, pre_mix)

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(buffer[:num_tokens], hidden_states.flatten(1), rtol=0, atol=0)
            assert torch.all(buffer[num_tokens:] == sentinel)
            assert buffer.data_ptr() == pointer
    finally:
        cpu_impl._destroy()

    assert graphs
    for graph in graphs:
        targets = {str(node.target) for node in graph.graph.nodes if node.op == "call_function"}
        assert "vllm.ascend_dsv41_mtp_save_and_collapse.default" in targets
        assert not any(name in target for target in targets for name in ("clone", "slice_scatter", "copy_"))


def test_original_mtp_slice_mutation_reproduces_full_buffer_scatter():
    buffer = torch.zeros(128, 64)
    graphs = []

    def compiler(graph, example_inputs):
        graphs.append(graph)
        return make_boxed_func(graph.forward)

    def original(hidden_states, pre_mix):
        buffer[: hidden_states.shape[0]].copy_(hidden_states.flatten(1))
        return model.DeepseekV41DecoderLayer.hc_collapse(hidden_states, pre_mix)

    compiled = torch.compile(original, backend=aot_autograd(fw_compiler=compiler), fullgraph=True, dynamic=True)
    compiled(torch.randn(24, 4, 16), torch.randn(24, 4))

    assert any("slice_scatter" in str(node.target) for graph in graphs for node in graph.graph.nodes)