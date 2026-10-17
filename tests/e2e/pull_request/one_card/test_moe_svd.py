from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.moe_low_rank import low_rank_apply_mlp, low_rank_linear
from vllm_ascend.quantization.methods.w4a8.w4a8_svd import (
    AscendW4A8SVDFusedMoEMethod,
    AscendW4A8SVDMoEScheme,
)
from vllm_ascend.quantization.moe_svd import quantize_factor

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")


def make_layer():
    method = object.__new__(AscendW4A8SVDFusedMoEMethod)
    method.rank = 256
    layer = torch.nn.Module()
    layer.expert_map = torch.tensor([-1, 0, 1, 2])
    with torch.device("npu"):
        method.create_weights(layer, 3, 512, 256, torch.bfloat16)
    references = {}
    torch.manual_seed(92)
    for prefix, out_size, in_size, shards in (("w13", 256, 512, ("w1", "w3")), ("w2", 512, 256, ("w2",))):
        for branch, shard in enumerate(shards):
            for side, out_width, in_width in (("left", out_size, 256), ("right", 256, in_size)):
                factors = []
                for expert in range(1, 4):
                    factor = quantize_factor(torch.randn(out_width, in_width) / in_width**0.5)
                    factors.append(factor.dequantize())
                    for kind, value in (
                        ("weight", factor.weight),
                        ("scale", factor.scale),
                        ("bias", 8 * factor.dequantize().sum(dim=1)),
                    ):
                        param = getattr(layer, f"{prefix}_{side}_{kind}")
                        assert param.weight_loader(param, value, shard_id=shard, expert_id=expert, return_success=True)
                        assert not param.weight_loader(param, value, shard_id=shard, expert_id=0, return_success=True)
                references[prefix, branch, side] = factors
    scheme = object.__new__(AscendW4A8SVDMoEScheme)
    before = sum(p.numel() * p.element_size() for p in layer.parameters())
    scheme.process_weights_after_loading(layer)
    assert before == sum(p.numel() * p.element_size() for p in layer.parameters())
    assert all(p.storage_offset() == 0 for p in layer.parameters())
    return layer, scheme.get_low_rank_weights(layer), references


@pytest.mark.parametrize("group_type", [0, 1])
def test_grouped_factors_replay_and_shared_quantized_input(group_type):
    layer, factors, reference = make_layer()
    assert not hasattr(layer, "w13_weight")
    assert all(param.dtype in (torch.int32, torch.int64, torch.float32) for param in layer.parameters())
    counts = [3, 0, 5]
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    if group_type == 0:
        groups = groups.cumsum(0)
    x = torch.randn(8, 512, dtype=torch.bfloat16, device="npu")
    quantized, scales = torch_npu.npu_dynamic_quant(x)
    pristine = quantized.clone()

    def compute():
        return tuple(low_rank_linear(quantized, factor, groups, group_type, scales) for factor in factors[:2])

    outputs = compute()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = compute()
    for phase in range(2):
        if phase:
            counts = [0, 4, 4]
            groups.copy_(torch.tensor(counts, device="npu", dtype=torch.int64))
            if group_type == 0:
                groups.copy_(groups.cumsum(0))
            fresh, new_scales = torch_npu.npu_dynamic_quant(x * 0.5)
            quantized.copy_(fresh)
            scales.copy_(new_scales)
            pristine.copy_(fresh)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(quantized, pristine, atol=0, rtol=0)
        inputs = quantized.cpu().float() * scales.cpu()[:, None]
        for branch in range(2):
            expected = []
            start = 0
            for expert, count in enumerate(counts):
                right, left = reference["w13", branch, "right"][expert], reference["w13", branch, "left"][expert]
                latent = (inputs[start : start + count] @ right.T).bfloat16()
                if count:
                    q, s = torch_npu.npu_dynamic_quant(latent.npu())
                    latent = q.cpu().float() * s.cpu()[:, None]
                expected.append(latent.float() @ left.T)
                start += count
            expected = torch.cat(expected)
            actual = captured[branch].cpu().float()
            assert float((actual - expected).norm() / expected.norm()) < 0.015, (phase, branch)
            if not phase:
                torch.testing.assert_close(captured[branch], outputs[branch], atol=0, rtol=0)


@pytest.mark.parametrize("swiglu_limit,scale_routes", [(0.0, False), (0.25, True)])
def test_complete_expert_mlp_preserves_activation_position(swiglu_limit, scale_routes):
    _, factors, reference = make_layer()
    counts = [2, 0, 3]
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    x = torch.randn(5, 512, dtype=torch.bfloat16, device="npu")
    topk_scales = torch.linspace(0.2, 0.8, 5, device="npu", dtype=torch.bfloat16)[:, None] if scale_routes else None
    inputs = SimpleNamespace(
        hidden_states=x,
        group_list=groups,
        group_list_type=1,
        dynamic_scale=None,
        activation=MoEActivation.SILU,
        dynamic_eplb=False,
        weights=SimpleNamespace(low_rank=factors),
        layer=None,
        lora_context=None,
        topk_scales=topk_scales,
        swiglu_limit=swiglu_limit,
    )
    actual, _ = low_rank_apply_mlp(inputs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured, before_down = low_rank_apply_mlp(inputs)
    graph.replay()
    before_down.synchronize()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, actual, atol=0, rtol=0)
    actual = actual.cpu().float()

    def linear(value, prefix, branch, expert):
        for side in ("right", "left"):
            q, scale = torch_npu.npu_dynamic_quant(value.npu())
            quantized = q.cpu().float() * scale.cpu()[:, None]
            value = (quantized @ reference[prefix, branch, side][expert].T).bfloat16()
        return value

    expected = []
    start = 0
    for expert, count in enumerate(counts):
        if count:
            value = x[start : start + count].cpu()
            gate = linear(value, "w13", 0, expert)
            up = linear(value, "w13", 1, expert)
            if swiglu_limit > 0:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up
            if topk_scales is not None:
                hidden = hidden * topk_scales[start : start + count].cpu()
            expected.append(linear(hidden, "w2", 0, expert))
        start += count
    expected = torch.cat(expected).float()
    assert float((actual - expected).norm() / expected.norm()) < 0.03
