# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Present Home/Slot aliases to the v0.30 quant-method MLP hooks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp


_WEIGHT_FIELDS = MappingProxyType({
    "w13_weight": "w1", "w2_weight": "w2",
    "w13_weight_scale": "w1_scale", "w2_weight_scale": "w2_scale",
    "w13_bias": "w1_bias", "w2_bias": "w2_bias",
    "w13_weight_scale_bias": "w1_scale_bias", "w2_weight_scale_bias": "w2_scale_bias",
    "w13_weight_offset": "w1_offset", "w2_weight_offset": "w2_offset",
})


@dataclass(frozen=True, slots=True)
class SlotMLPLayerView:
    """Read-only weight source; the model's Parameters are never swapped per step."""

    original: object
    weights: MoEWeights

    def __getattr__(self, name):
        field = _WEIGHT_FIELDS.get(name)
        if field is not None:
            return getattr(self.weights, field)
        if name in ("w13_weight_list", "w2_weight_list", "w13_weight_scale_list", "w2_weight_scale_list"):
            # Split expert tensors have a different layout and are rejected
            # at layer registration. BF16's optional-list getter uses tensors.
            return None
        return getattr(self.original, name)


def apply_slot_mlp(mlp_compute_input: MoEMlpComputeInput, quant_method):
    if quant_method is None or mlp_compute_input.layer is None:
        raise RuntimeError("MoonEP v0.30 MLP requires its routed-expert layer and quant method")
    if getattr(quant_method, "use_expert_weight_list", False):
        raise NotImplementedError("MoonEP Slot does not support split expert weight lists")
    slot_input = replace(
        mlp_compute_input,
        layer=SlotMLPLayerView(mlp_compute_input.layer, mlp_compute_input.weights),
    )
    return apply_moe_mlp(slot_input, quant_method)
