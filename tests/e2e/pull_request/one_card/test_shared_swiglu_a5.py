# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""A5 shared-expert standard SwiGLU must retain its unbounded semantics."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import shared_experts
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="A5 SwiGLU ABI")


@pytest.mark.parametrize("weight_nz_mode", [0, 1])
@pytest.mark.parametrize("tokens", [1, 16])
def test_a5_shared_mlp_standard_swiglu(tokens, weight_nz_mode):
    """Exercise real weight postprocessing, shared Gate-Up, activation, and Down."""
    torch.manual_seed(19)
    hidden, intermediate = 6144, 256
    config = SimpleNamespace(
        weight_nz_mode=weight_nz_mode, multistream_overlap_shared_expert=False, enable_shared_expert_dp=False
    )
    linear_method = AscendW8A8DynamicLinearMethod()
    originals = []
    layers = []
    with patch("vllm_ascend.utils.get_ascend_config", return_value=config):
        for k, n in ((hidden, 2 * intermediate), (intermediate, hidden)):
            weight = torch.randint(-16, 16, (n, k), dtype=torch.int8)
            scale = (torch.rand(n) * 0.02 + 0.01).to(torch.bfloat16)
            layer = torch.nn.Module()
            for name, tensor in (
                ("weight", weight),
                ("weight_scale", scale[:, None]),
                ("weight_offset", torch.zeros(n, 1, dtype=torch.bfloat16)),
            ):
                layer.register_parameter(name, torch.nn.Parameter(tensor.npu(), requires_grad=False))
            linear_method.process_weights_after_loading(layer)
            torch.testing.assert_close(layer.weight.cpu(), weight.T.contiguous())
            torch.testing.assert_close(layer.weight_scale_fp32.cpu(), scale.float())
            originals.append((weight.T.float(), scale.float()))
            layers.append(layer)
    layer = SimpleNamespace(gate_up_proj=layers[0], down_proj=layers[1], act_fn=torch.nn.SiLU())
    moe_config = SimpleNamespace(
        hidden_dim=hidden,
        in_dtype=torch.bfloat16,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_sequence_parallel=False,
    )
    with patch.object(shared_experts, "get_ascend_config", return_value=config):
        executor = shared_experts.AscendSharedExperts(layer, moe_config, QuantType.W8A8, linear_method)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16).npu()

    def run():
        prepared = shared_experts.PreparedSharedExpertExecution(x, executor._prepare_gate_up_input(x), None)
        submitted = executor.enqueue_gate_up_after_router(prepared, None)
        return executor._run_a8_int_mlp(submitted, RoutedMoEMilestones())

    quantized, token_scale = executor._prepare_gate_up_input(x)
    w1, s1 = originals[0]
    w2, s2 = originals[1]
    gate_up = (quantized.cpu().float() @ w1) * s1 * token_scale.cpu()[:, None]
    gate, up = gate_up.chunk(2, -1)
    assert gate_up.abs().max() > 7, "Inputs must distinguish standard SwiGLU from the native mode-1 default clamp."
    activation = torch.nn.functional.silu(gate) * up
    output_scale = activation.abs().amax(-1) / 127
    q = (activation / output_scale[:, None]).round().clamp(-128, 127)
    expected = (q @ w2) * s2 * output_scale[:, None]
    actual = run()
    relative_error = (actual.float().cpu() - expected).norm() / expected.norm()
    assert relative_error < 0.02, float(relative_error)
    for _ in range(2):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        graph_output = run()
    for _ in range(3):
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_output, actual, rtol=0, atol=0)
