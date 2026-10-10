import gc
import os
from pathlib import Path

import pytest
import torch
import torch_npu
from safetensors.torch import safe_open

from vllm_ascend.utils import enable_custom_op


DSV4_W4A8_WEIGHT_PATH = Path(
    os.environ.get(
        "DSV4_W4A8_WEIGHT_PATH",
        "/data/ms-probe/Eco-Tech_DeepSeek-V4-Pro-w4a8-mtp/"
        "quant_model_weights-00001-of-00205.safetensors",
    )
)
GROUP_LIST_TYPE_COUNT = 1


def _dsv4_expert_key(layer: int, expert: int, suffix: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{suffix}"


@pytest.fixture(scope="module")
def dsv4_w13_tensors():
    if not DSV4_W4A8_WEIGHT_PATH.exists():
        pytest.skip(f"DSV4 W4A8 weight shard is not available: {DSV4_W4A8_WEIGHT_PATH}")

    torch.npu.config.allow_internal_format = True
    enable_custom_op()

    layer = 0
    experts = [0, 1, 2, 3]
    with safe_open(DSV4_W4A8_WEIGHT_PATH, framework="pt", device="cpu") as weights:
        w13_weight = torch.stack(
            [
                torch.cat(
                    [
                        weights.get_tensor(_dsv4_expert_key(layer, expert, "w1.weight")),
                        weights.get_tensor(_dsv4_expert_key(layer, expert, "w3.weight")),
                    ],
                    dim=0,
                )
                for expert in experts
            ],
            dim=0,
        ).contiguous()
        w13_weight_scale = torch.stack(
            [
                torch.cat(
                    [
                        weights.get_tensor(_dsv4_expert_key(layer, expert, "w1.weight_scale")),
                        weights.get_tensor(_dsv4_expert_key(layer, expert, "w3.weight_scale")),
                    ],
                    dim=0,
                )
                for expert in experts
            ],
            dim=0,
        ).contiguous()
        w13_scale_bias = torch.stack(
            [
                torch.cat(
                    [
                        weights.get_tensor(_dsv4_expert_key(layer, expert, "w1.scale_bias")),
                        weights.get_tensor(_dsv4_expert_key(layer, expert, "w3.scale_bias")),
                    ],
                    dim=0,
                )
                for expert in experts
            ],
            dim=0,
        ).contiguous()

    weight = torch_npu.npu_format_cast(w13_weight.npu().transpose(1, 2).contiguous(), 29)
    weight = weight.view(torch.int32).contiguous()

    weight_scale = w13_weight_scale.transpose(1, 2).contiguous()
    weight_scale_3d = weight_scale.view(torch.int32).to(torch.int64).npu().contiguous()
    weight_scale_2d = weight_scale_3d.squeeze(1).contiguous()
    weight_bias = w13_scale_bias.transpose(1, 2).sum(dim=1).npu().contiguous()

    yield weight, weight_scale_2d, weight_scale_3d, weight_bias

    del weight, weight_scale_2d, weight_scale_3d, weight_bias
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _make_int8_input(num_tokens: int, hidden_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        -127,
        128,
        (num_tokens, hidden_size),
        dtype=torch.int8,
        generator=generator,
    ).npu()


def _small_op_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias: torch.Tensor,
    swiglu_limit: float,
):
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[x.clone()],
        weight=[weight],
        scale=[weight_scale.clone()],
        bias=[bias.clone()],
        per_token_scale=[x_scale],
        split_item=2,
        group_type=0,
        group_list=group_list,
        group_list_type=GROUP_LIST_TYPE_COUNT,
        output_dtype=torch.bfloat16,
    )[0]
    act, gate = hidden_states.chunk(2, dim=-1)
    if swiglu_limit > 0:
        act = torch.clamp(act, max=swiglu_limit)
        gate = torch.clamp(gate, min=-swiglu_limit, max=swiglu_limit)
    return torch_npu.npu_dynamic_quant(torch.nn.functional.silu(act) * gate)


def _gmmswigluquantv2(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias: torch.Tensor,
    swiglu_limit: float,
):
    return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=x.clone(),
        weight=[weight],
        weight_scale=[weight_scale.clone()],
        x_scale=x_scale,
        group_list=group_list,
        weight_assist_matrix=[bias.clone()],
        dequant_mode=0,
        group_list_type=GROUP_LIST_TYPE_COUNT,
        swiglu_limit=swiglu_limit,
    )


@pytest.mark.parametrize(
    ("expert_token_counts", "seed", "x_scale_value", "swiglu_limit"),
    [
        pytest.param([2, 3, 1, 4], 131, 1.0, 0.05, id="clamp-stress-varied-counts"),
        pytest.param([1, 1, 1, 1], 23, 1.0, 0.05, id="clamp-stress-one-token-per-expert"),
    ],
)
@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_matches_dsv4_w4a8_small_ops(
    dsv4_w13_tensors,
    expert_token_counts,
    seed,
    x_scale_value,
    swiglu_limit,
):
    weight, fused_weight_scale, reference_weight_scale, bias = dsv4_w13_tensors
    num_tokens = sum(expert_token_counts)
    hidden_size = 7168

    x = _make_int8_input(num_tokens, hidden_size, seed)
    x_scale = torch.full((num_tokens,), x_scale_value, dtype=torch.float32, device="npu")
    group_list = torch.tensor(expert_token_counts, dtype=torch.int64, device="npu")

    reference_output, reference_scale = _small_op_swiglu_quant(
        x=x,
        weight=weight,
        weight_scale=reference_weight_scale,
        x_scale=x_scale,
        group_list=group_list,
        bias=bias,
        swiglu_limit=swiglu_limit,
    )
    fused_output, fused_scale = _gmmswigluquantv2(
        x=x,
        weight=weight,
        weight_scale=fused_weight_scale,
        x_scale=x_scale,
        group_list=group_list,
        bias=bias,
        swiglu_limit=swiglu_limit,
    )

    quant_diff = (fused_output.cpu().to(torch.int16) - reference_output.cpu().to(torch.int16)).abs()
    scale_diff = (fused_scale.cpu() - reference_scale.cpu()).abs()
    assert int(quant_diff.max()) <= 1, {
        "expert_token_counts": expert_token_counts,
        "seed": seed,
        "x_scale_value": x_scale_value,
        "swiglu_limit": swiglu_limit,
        "qmax": int(quant_diff.max()),
        "qmean": float(quant_diff.float().mean()),
        "exact_ratio": float((quant_diff == 0).float().mean()),
        "scale_max": float(scale_diff.max()),
        "scale_mean": float(scale_diff.mean()),
    }
    torch.testing.assert_close(fused_scale.cpu(), reference_scale.cpu(), atol=1e-6, rtol=1e-3)
