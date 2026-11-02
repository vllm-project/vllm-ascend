# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""A5 INT4 W4A8 loading, signed-matmul correctness, and graph regression."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.quantization.methods.w4a8 import w4a8
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="A5 INT4 ABI")


def _cpu_gmm(x, weight, token_scale, weight_scale, counts):
    outputs = []
    start = 0
    for expert, count in enumerate(counts):
        output = x[start : start + count].float() @ weight[expert].float()
        outputs.append(output * weight_scale[expert] * token_scale[start : start + count, None])
        start += count
    return torch.cat(outputs)


def _assert_close(actual, expected, relative_tolerance):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    relative_error = (actual - expected).norm() / expected.norm()
    assert relative_error < relative_tolerance, float(relative_error)


@pytest.mark.parametrize("weight_nz_mode", [0, 1])
def test_a5_w4a8_modelslim_mlp(weight_nz_mode):
    """Keep signed INT4 math at GLM TP8 dimensions, including empty experts."""
    torch.manual_seed(175)
    counts = [0, 7, 9, 0]
    experts, tokens, hidden, intermediate = 4, sum(counts), 6144, 256
    layer = torch.nn.Module()
    originals = {}
    for name, k, n in (("w13", hidden, intermediate * 2), ("w2", intermediate, hidden)):
        weight = torch.randint(-8, 8, (experts, k, n), dtype=torch.int8)
        scale = torch.rand(experts, n) * 0.002 + 0.001
        bias = 8 * (weight.float() * scale[:, None]).sum(1)
        packed = ((weight[..., 0::2] & 15) | (weight[..., 1::2] << 4)).contiguous()
        for suffix, value in (
            ("weight", packed.transpose(1, 2).contiguous()),
            ("weight_scale", scale[:, :, None]),
            ("scale_bias", bias[:, :, None]),
        ):
            layer.register_parameter(f"{name}_{suffix}", torch.nn.Parameter(value.npu(), requires_grad=False))
        originals[name] = weight, scale, bias

    method = object.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.quant_method = ASCEND_QUANTIZATION_METHOD
    method.use_a5_split_gmm = True
    method.use_expert_weight_list = False
    config = SimpleNamespace(weight_nz_mode=weight_nz_mode)
    with (
        patch.object(w4a8, "get_current_vllm_config", return_value=None),
        patch.object(w4a8, "get_ascend_config", return_value=config),
        patch.object(w4a8, "use_cann_megamoe", return_value=False),
        patch.object(w4a8, "maybe_trans_nz", side_effect=AssertionError("A5 INT4 must remain ND")),
    ):
        method.process_weights_after_loading(layer)
    assert not method.supports_fused_activation(MoEActivation.SILU)
    for name, k in (("w13", hidden), ("w2", intermediate)):
        weight = getattr(layer, f"{name}_weight")
        assert weight.dtype == torch.int32
        assert torch_npu.get_npu_format(weight) == 2
        scale = getattr(layer, f"{name}_weight_scale").cpu()
        expected = originals[name][1].view(torch.int32).to(torch.int64)[:, None].repeat(1, k // 256, 1)
        torch.testing.assert_close(scale, expected, rtol=0, atol=0)
        torch.testing.assert_close(getattr(layer, f"{name}_scale_bias").cpu(), originals[name][2])

    x_cpu = torch.randint(-100, 100, (tokens, hidden), dtype=torch.int8)
    sx_cpu = torch.rand(tokens) * 0.01 + 0.005
    x, sx = x_cpu.npu(), sx_cpu.npu()
    group_list = torch.tensor(counts, dtype=torch.int64, device="npu")

    def inputs():
        return SimpleNamespace(
            hidden_states=x.clone(),
            dynamic_scale=sx,
            layer=layer,
            group_list=group_list,
            group_list_type=1,
            activation=MoEActivation.SILU,
            swiglu_limit=0,
            quant=SimpleNamespace(is_quant=True),
            lora_context=None,
        )

    g1 = method.apply_gmm1(inputs())
    g1_ref = _cpu_gmm(x_cpu, originals["w13"][0], sx_cpu, originals["w13"][1], counts)
    _assert_close(g1, g1_ref, 0.004)
    act = torch_npu.npu_swiglu(g1)
    qact, sact = torch_npu.npu_dynamic_quant(act, dst_type=torch.int8)
    g2 = method.apply_gmm2(inputs(), qact, sact)
    g2_ref = _cpu_gmm(qact.cpu(), originals["w2"][0], sact.cpu(), originals["w2"][1], counts)
    _assert_close(g2, g2_ref, 0.004)

    gate, up = g1_ref.to(torch.bfloat16).chunk(2, dim=-1)
    act_ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    sref = act_ref.float().abs().amax(-1) / 127
    qref = (act_ref.float() / sref[:, None]).round().clamp(-128, 127).to(torch.int8)
    full_ref = _cpu_gmm(qref, originals["w2"][0], sref, originals["w2"][1], counts)

    with patch.object(moe_mlp, "maybe_record_event", return_value=None):
        actual, _ = moe_mlp.apply_moe_mlp(inputs(), method)
        _assert_close(actual, full_ref, 0.015)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output, _ = moe_mlp.apply_moe_mlp(inputs(), method)
        for _ in range(3):
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_output, actual, rtol=0, atol=0)
