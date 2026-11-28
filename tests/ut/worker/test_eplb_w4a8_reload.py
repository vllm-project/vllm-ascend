# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check that the FT W4A8 expert reload lands the same tensors as a normal load."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD
from vllm_ascend.worker.sentinel import eplb_quant_schemes as schemes
from vllm_ascend.worker.sentinel.eplb_redistribute import _reload_expert_stack

EXPERTS = 2
INPUT_SIZE = 16
OUTPUT_SIZE = 56
SCALE_BIAS_PARTITIONS = 16

# Runtime per-expert attributes the reload writes.
RELOADED_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_scale_bias",
    "w2_scale_bias",
)


def _expert_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def int8(*shape: int) -> torch.Tensor:
        return torch.randint(-8, 8, shape, generator=generator, dtype=torch.int8)

    def float32(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=generator, dtype=torch.float32)

    return {
        "gate_proj.weight": int8(INPUT_SIZE, OUTPUT_SIZE),
        "up_proj.weight": int8(INPUT_SIZE, OUTPUT_SIZE),
        "down_proj.weight": int8(OUTPUT_SIZE // 2, INPUT_SIZE),
        "gate_proj.weight_scale": float32(INPUT_SIZE, 1),
        "up_proj.weight_scale": float32(INPUT_SIZE, 1),
        "down_proj.weight_scale": float32(OUTPUT_SIZE, 1),
        "gate_proj.scale_bias": float32(INPUT_SIZE, SCALE_BIAS_PARTITIONS),
        "up_proj.scale_bias": float32(INPUT_SIZE, SCALE_BIAS_PARTITIONS),
        "down_proj.scale_bias": float32(OUTPUT_SIZE, SCALE_BIAS_PARTITIONS),
    }


def _build_reference_layer(checkpoints: list[dict[str, torch.Tensor]]) -> torch.nn.Module:
    """Lay out the stack the normal load path expects before processing."""

    def stacked(*keys: str) -> torch.Tensor:
        return torch.stack([torch.cat([checkpoint[key] for key in keys], dim=0) for checkpoint in checkpoints])

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(stacked("gate_proj.weight", "up_proj.weight"), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(stacked("down_proj.weight"), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        stacked("gate_proj.weight_scale", "up_proj.weight_scale"), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(stacked("down_proj.weight_scale"), requires_grad=False)
    layer.w13_scale_bias = torch.nn.Parameter(
        stacked("gate_proj.scale_bias", "up_proj.scale_bias"), requires_grad=False
    )
    layer.w2_scale_bias = torch.nn.Parameter(stacked("down_proj.scale_bias"), requires_grad=False)
    return layer


def _normal_load(layer: torch.nn.Module) -> None:
    method = object.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.quant_method = ASCEND_QUANTIZATION_METHOD
    method.use_expert_weight_list = True
    with (
        patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config", new=lambda: None),
        patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False),
        patch("vllm_ascend.quantization.methods.w4a8.w4a8.torch_npu.npu_format_cast") as format_cast,
        patch("torch.Tensor.npu", new=lambda self: self, create=True),
    ):
        format_cast.side_effect = lambda tensor, *args, **kwargs: tensor
        method.process_weights_after_loading(layer)


def _reload_slot_zero(layer: torch.nn.Module, checkpoint: dict[str, torch.Tensor]) -> int:
    layer.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_rank=0, tp_size=1))
    with (
        patch("vllm_ascend.worker.sentinel.eplb_quant_schemes.torch_npu.npu_format_cast") as format_cast,
        patch("torch.npu.synchronize", create=True),
    ):
        format_cast.side_effect = lambda tensor, *args, **kwargs: tensor
        return _reload_expert_stack(
            schemes._W4A8_ASCEND_DEFINITION,
            [layer],
            [(0, 0)],
            {(0, 0): 0},
            {(0, 0): checkpoint},
        )


def test_w4a8_reload_matches_normal_load_and_keeps_neighbor_slot():
    checkpoints = [_expert_checkpoint(seed) for seed in (1, 2)]

    reference = _build_reference_layer(checkpoints)
    _normal_load(reference)

    reloaded = torch.nn.Module()
    for name in RELOADED_NAMES:
        reference_list = getattr(reference, f"{name}_list")
        # Slot 1 is a neighbour the reload must never touch.
        setattr(reloaded, f"{name}_list", [torch.full_like(tensor, 7) for tensor in reference_list])

    assert _reload_slot_zero(reloaded, checkpoints[0]) == 1

    for name in RELOADED_NAMES:
        expected = getattr(reference, f"{name}_list")[0]
        landed = getattr(reloaded, f"{name}_list")[0]
        assert landed.shape == expected.shape
        assert landed.dtype == expected.dtype
        torch.testing.assert_close(landed, expected)
        torch.testing.assert_close(
            getattr(reloaded, f"{name}_list")[1],
            torch.full_like(expected, 7),
        )
