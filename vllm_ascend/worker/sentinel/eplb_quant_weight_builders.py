# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Weight reload builders for the EPLB scale-down.

Each quant method declares how to read one expert's tensors from the
checkpoint and how to convert them into the same form the normal load path's
``process_weights_after_loading`` produces.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch_npu

from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    flatten_expert_dims,
)
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    ASCEND_QUANTIZATION_METHOD,
    maybe_trans_nz,
    transpose_expert_weight_last_two_dims,
)

# Checkpoint tensor suffixes (relative to "<layer_name>.<expert_id>.") the
# reload may read; weight_offset is not used by the MoE apply path.
_W13_WEIGHT_SUFFIXES = ("gate_up_proj.weight", "gate_proj.weight", "up_proj.weight")
_W2_WEIGHT_SUFFIX = "down_proj.weight"
_W13_SCALE_SUFFIXES = ("gate_up_proj.weight_scale", "gate_proj.weight_scale", "up_proj.weight_scale")
_W2_SCALE_SUFFIX = "down_proj.weight_scale"
# scale_bias exists only in the msModelSlim ("ascend") format; compressed_tensors
# derives it, so those checkpoints have no such keys.
_W13_SCALE_BIAS_SUFFIXES = ("gate_up_proj.scale_bias", "gate_proj.scale_bias", "up_proj.scale_bias")
_W2_SCALE_BIAS_SUFFIX = "down_proj.scale_bias"

# Every checkpoint key the reload may consume.
_WANTED_SUFFIXES = frozenset(
    _W13_WEIGHT_SUFFIXES
    + _W13_SCALE_SUFFIXES
    + _W13_SCALE_BIAS_SUFFIXES
    + (_W2_WEIGHT_SUFFIX, _W2_SCALE_SUFFIX, _W2_SCALE_BIAS_SUFFIX)
)


class _WeightBuilder(NamedTuple):
    """How to build one weight tensor for a single expert.

    ``source`` fetches the tensor from the checkpoint, already split for this TP
    rank; ``processor`` applies the same function the normal load path's
    ``process_weights_after_loading`` calls.
    """

    source: Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]
    processor: Callable[[torch.Tensor], torch.Tensor]


# Checkpoint readers: one expert's tensors as this TP rank holds them.
def _gate_up(suffix: str) -> Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]:
    """Read w13 (gate then up) for one expert, row-sharded for this TP rank.

    The checkpoint stores either a fused ``gate_up_proj`` or separate
    ``gate_proj``/``up_proj`` tensors.
    """

    def weight_source(tensors: dict[str, torch.Tensor], tp_rank: int, tp_size: int) -> torch.Tensor:
        fused = tensors.get(f"gate_up_proj.{suffix}")
        if fused is not None:
            gate, up = fused.chunk(2)
        else:
            gate, up = tensors[f"gate_proj.{suffix}"], tensors[f"up_proj.{suffix}"]
        if tp_size > 1:
            shard = gate.shape[0] // tp_size
            start = tp_rank * shard
            gate, up = gate.narrow(0, start, shard), up.narrow(0, start, shard)
        return torch.cat([gate, up], dim=0)

    return weight_source


def _down_proj(suffix: str, *, shard: bool = False) -> Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]:
    """Read one expert's ``down_proj`` tensor, column-sharded when ``shard``."""

    def weight_source(tensors: dict[str, torch.Tensor], tp_rank: int, tp_size: int) -> torch.Tensor:
        tensor = tensors[f"down_proj.{suffix}"]
        if not shard or tp_size == 1:
            return tensor
        shard_size = tensor.shape[1] // tp_size
        return tensor.narrow(1, tp_rank * shard_size, shard_size)

    return weight_source


def _apply_stack_transform_to_one_expert(
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Apply a transform that expects the full expert stack to one expert's tensor."""

    def convert(expert: torch.Tensor) -> torch.Tensor:
        return transform(expert.unsqueeze(0)).squeeze(0)

    return convert


# The two MoE expert weight tensors every quant method reloads. Only these
# are cast to FRACTAL_NZ; quant scales and scale biases are converted
# separately and never NZ-cast.
_MOE_WEIGHT_PIPELINES: dict[str, _WeightBuilder] = {
    "w13_weight": _WeightBuilder(_gate_up("weight"), transpose_expert_weight_last_two_dims),
    "w2_weight": _WeightBuilder(_down_proj("weight", shard=True), transpose_expert_weight_last_two_dims),
}
_MOE_WEIGHT_NAMES = tuple(_MOE_WEIGHT_PIPELINES)


def _to_fractal_nz(weight: torch.Tensor) -> torch.Tensor:
    """Cast to FRACTAL_NZ, matching the W8A8/W4A8 MoE loaders."""
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)


@dataclass(eq=False)  # compared by identity: used as a dict key in the reload
class _QuantReloadPlans:
    """The reload plans for one quant method's expert tensors.

    ``pipelines`` maps each weight tensor name (``w13_weight``, ...) to the
    reader/converter pair that rebuilds it; ``cast_weights_to_nz`` converts the expert
    weight tensors to FRACTAL_NZ afterwards (a no-op when the policy does not
    require it).
    """

    quant_method: type
    pipelines: dict[str, _WeightBuilder]
    cast_weights_to_nz: Callable[[torch.Tensor], torch.Tensor]

    def matches(self, method) -> bool:
        if type(method) is not self.quant_method:
            return False
        # W4A8 has several checkpoint sub-formats; only msModelSlim ("ascend") is supported.
        if self.quant_method is AscendW4A8DynamicFusedMoEMethod:
            return method.quant_method == ASCEND_QUANTIZATION_METHOD
        return True


_UNQUANTIZED_PLANS = _QuantReloadPlans(
    AscendUnquantizedFusedMoEMethod,
    _MOE_WEIGHT_PIPELINES,
    maybe_trans_nz,
)
_W8A8_DYNAMIC_PLANS = _QuantReloadPlans(
    AscendW8A8DynamicFusedMoEMethod,
    {
        **_MOE_WEIGHT_PIPELINES,
        "w13_weight_scale_fp32": _WeightBuilder(
            _gate_up("weight_scale"), _apply_stack_transform_to_one_expert(flatten_expert_dims)
        ),
        "w2_weight_scale": _WeightBuilder(
            _down_proj("weight_scale"), _apply_stack_transform_to_one_expert(flatten_expert_dims)
        ),
    },
    _to_fractal_nz,
)
_W4A8_ASCEND_PLANS = _QuantReloadPlans(
    AscendW4A8DynamicFusedMoEMethod,
    {
        **_MOE_WEIGHT_PIPELINES,
        "w13_weight_scale": _WeightBuilder(
            _gate_up("weight_scale"),
            _apply_stack_transform_to_one_expert(AscendW4A8DynamicFusedMoEMethod.w13_scale_to_runtime),
        ),
        "w2_weight_scale": _WeightBuilder(_down_proj("weight_scale"), AscendW4A8DynamicFusedMoEMethod._process_scale),
        "w13_scale_bias": _WeightBuilder(
            _gate_up("scale_bias"), AscendW4A8DynamicFusedMoEMethod.reduce_scale_bias_modelslim
        ),
        "w2_scale_bias": _WeightBuilder(
            _down_proj("scale_bias"), AscendW4A8DynamicFusedMoEMethod.reduce_scale_bias_modelslim
        ),
    },
    _to_fractal_nz,
)

# Checked in this order by _find_reload_plans().
_RELOAD_PLANS: tuple[_QuantReloadPlans, ...] = (_UNQUANTIZED_PLANS, _W8A8_DYNAMIC_PLANS, _W4A8_ASCEND_PLANS)


def _find_reload_plans(routed_layer: torch.nn.Module) -> _QuantReloadPlans:
    """Pick the reload plans matching a routed-experts layer's quant method."""
    method = getattr(routed_layer.quant_method, "quant_method", routed_layer.quant_method)
    for plans in _RELOAD_PLANS:
        if plans.matches(method):
            return plans
    hint = f"checkpoint sub-format: {getattr(method, 'quant_method', None) or 'unknown'}"
    if isinstance(method, AscendW4A8DynamicFusedMoEMethod):
        hint += f'; only msModelSlim ("{ASCEND_QUANTIZATION_METHOD}") is supported'
    raise NotImplementedError(
        f"[FT] scale_down weight reload is not implemented for quant method {type(method).__name__} ({hint})."
    )
