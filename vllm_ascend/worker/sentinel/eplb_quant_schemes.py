# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Quant schemes for the EPLB scale-down expert weight reload.

Each scheme maps a routed-experts quant method to the checkpoint tensor
suffixes it consumes, the per-TP-rank gather, and the load path's
``process_weights_after_loading`` transforms, so the reload cannot drift from
a normal weight load.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch_npu

from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8 import (
    AscendW4A8DynamicFusedMoEMethod,
    process_scale_to_runtime,
    reduce_scale_bias_modelslim,
    w13_scale_to_runtime,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    flatten_expert_dims,
)
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    ASCEND_QUANTIZATION_METHOD,
    maybe_trans_nz,
    transpose_weight_to_runtime,
)

# Checkpoint suffixes (relative to "<layer_name>.<expert_id>.") the reload may
# consume; weight_offset is unused by the MoE apply path and not reloaded.
_W13_WEIGHT_SUFFIXES = ("gate_up_proj.weight", "gate_proj.weight", "up_proj.weight")
_W2_WEIGHT_SUFFIX = "down_proj.weight"
_W13_SCALE_SUFFIXES = ("gate_up_proj.weight_scale", "gate_proj.weight_scale", "up_proj.weight_scale")
_W2_SCALE_SUFFIX = "down_proj.weight_scale"
# scale_bias exists only in the msModelSlim ("ascend") format; compressed_tensors
# derives it, so those checkpoints have no such keys.
_W13_SCALE_BIAS_SUFFIXES = ("gate_up_proj.scale_bias", "gate_proj.scale_bias", "up_proj.scale_bias")
_W2_SCALE_BIAS_SUFFIX = "down_proj.scale_bias"

# Every checkpoint key the reload may consume, across all supported schemes.
_WANTED_SUFFIXES = frozenset(
    _W13_WEIGHT_SUFFIXES
    + _W13_SCALE_SUFFIXES
    + _W13_SCALE_BIAS_SUFFIXES
    + (_W2_WEIGHT_SUFFIX, _W2_SCALE_SUFFIX, _W2_SCALE_BIAS_SUFFIX)
)


def _shard_row(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Take this TP rank's row shard (dim 0) of a full checkpoint tensor."""
    if tp_size == 1:
        return t
    shard = t.shape[0] // tp_size
    return t.narrow(0, tp_rank * shard, shard)


def _shard_col(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Take this TP rank's column shard (dim 1) of a full checkpoint tensor."""
    if tp_size == 1:
        return t
    shard = t.shape[1] // tp_size
    return t.narrow(1, tp_rank * shard, shard)


class _WeightPipeline(NamedTuple):
    """One runtime attribute's reload: checkpoint gather + runtime transform.

    ``source`` gathers the checkpoint tensor (sharded for this TP rank).
    ``processor`` is the load path's ``process_weights_after_loading``
    transform, imported so the reload cannot drift from it.
    """

    source: Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]
    processor: Callable[[torch.Tensor], torch.Tensor]


# Weight sources: the checkpoint-side gather for one expert, sharded for this TP rank.
def _gate_up(suffix: str) -> Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]:
    """w13 order out of the checkpoint: fused ``gate_up_proj``, or split gate+up."""

    def weight_source(tensors: dict[str, torch.Tensor], tp_rank: int, tp_size: int) -> torch.Tensor:
        fused = tensors.get(f"gate_up_proj.{suffix}")
        if fused is not None:
            gate, up = fused.chunk(2)
        else:
            gate, up = tensors[f"gate_proj.{suffix}"], tensors[f"up_proj.{suffix}"]
        return torch.cat([_shard_row(gate, tp_rank, tp_size), _shard_row(up, tp_rank, tp_size)], dim=0)

    return weight_source


def _down_proj(suffix: str, *, shard: bool = False) -> Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]:
    """``down_proj.<suffix>`` as this rank holds it."""

    def weight_source(tensors: dict[str, torch.Tensor], tp_rank: int, tp_size: int) -> torch.Tensor:
        tensor = tensors[f"down_proj.{suffix}"]
        return _shard_col(tensor, tp_rank, tp_size) if shard else tensor

    return weight_source


def _single_expert(transform: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add the leading expert dim that ``[E, ...]``-shaped transforms expect (views only)."""

    def processor(expert: torch.Tensor) -> torch.Tensor:
        return transform(expert.unsqueeze(0)).squeeze(0)

    return processor


# The weight pipelines every scheme reloads; their runtime attributes are the
# ones that take the layout cast.
_WEIGHT_PIPELINES: dict[str, _WeightPipeline] = {
    "w13_weight": _WeightPipeline(_gate_up("weight"), transpose_weight_to_runtime),
    "w2_weight": _WeightPipeline(_down_proj("weight", shard=True), transpose_weight_to_runtime),
}
_WEIGHT_ATTRS = tuple(_WEIGHT_PIPELINES)


def _to_fractal_nz(weight: torch.Tensor) -> torch.Tensor:
    """Unconditional FRACTAL_NZ cast, as W8A8's weights are loaded with."""
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)


@dataclass(eq=False)  # identity semantics: registry entries are singletons
class _ReloadScheme:
    """Everything the reload needs for one quant method."""

    quant_method: type
    pipelines: dict[str, _WeightPipeline]
    cast_layout: Callable[[torch.Tensor], torch.Tensor]

    def matches(self, method) -> bool:
        if type(method) is not self.quant_method:
            return False
        # W4A8 has several checkpoint sub-formats; only msModelSlim ("ascend") maps to a scheme.
        if self.quant_method is AscendW4A8DynamicFusedMoEMethod:
            return method.quant_method == ASCEND_QUANTIZATION_METHOD
        return True


# One variable per scheme; ``_SCHEMES`` is only the resolution order.
_UNQUANTIZED_SCHEME = _ReloadScheme(
    AscendUnquantizedFusedMoEMethod,
    _WEIGHT_PIPELINES,
    maybe_trans_nz,
)
_W8A8_DYNAMIC_SCHEME = _ReloadScheme(
    AscendW8A8DynamicFusedMoEMethod,
    {
        **_WEIGHT_PIPELINES,
        "w13_weight_scale_fp32": _WeightPipeline(_gate_up("weight_scale"), _single_expert(flatten_expert_dims)),
        "w2_weight_scale": _WeightPipeline(_down_proj("weight_scale"), _single_expert(flatten_expert_dims)),
    },
    _to_fractal_nz,
)
_W4A8_ASCEND_SCHEME = _ReloadScheme(
    AscendW4A8DynamicFusedMoEMethod,
    {
        **_WEIGHT_PIPELINES,
        "w13_weight_scale": _WeightPipeline(_gate_up("weight_scale"), _single_expert(w13_scale_to_runtime)),
        "w2_weight_scale": _WeightPipeline(_down_proj("weight_scale"), process_scale_to_runtime),
        "w13_scale_bias": _WeightPipeline(_gate_up("scale_bias"), reduce_scale_bias_modelslim),
        "w2_scale_bias": _WeightPipeline(_down_proj("scale_bias"), reduce_scale_bias_modelslim),
    },
    maybe_trans_nz,
)

_SCHEMES: tuple[_ReloadScheme, ...] = (_UNQUANTIZED_SCHEME, _W8A8_DYNAMIC_SCHEME, _W4A8_ASCEND_SCHEME)


def _resolve_scheme(routed: torch.nn.Module) -> _ReloadScheme:
    """The reload rules a routed-experts layer uses."""

    method = getattr(routed.quant_method, "quant_method", routed.quant_method)
    for scheme in _SCHEMES:
        if scheme.matches(method):
            return scheme
    hint = f"checkpoint sub-format: {getattr(method, 'quant_method', None) or 'unknown'}"
    if isinstance(method, AscendW4A8DynamicFusedMoEMethod):
        hint += f'; only msModelSlim ("{ASCEND_QUANTIZATION_METHOD}") is supported'
    raise NotImplementedError(
        f"[FT] scale_down weight reload is not implemented for quant method {type(method).__name__} ({hint})."
    )
