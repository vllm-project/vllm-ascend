from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm_ascend.quantization.quant_type import QuantType


@dataclass(frozen=True, slots=True)
class MoERoutingParams:
    """Routing and dispatch side inputs for one MoE invocation.

    `pertoken_scale` is intentionally kept here even though it is not a pure
    routing concept. It is used by pre-quantized activation flows, currently
    the AllGather + EP W8A8 prepare path, where prepare emits per-token
    activation scales and dispatch needs to carry them forward so the MLP
    quant path can reuse those scales instead of requantizing activations.
    """

    expert_map: torch.Tensor | None
    global_redundant_expert_num: int
    mc2_mask: torch.Tensor | None
    apply_router_weight_on_input: bool
    log2phy: torch.Tensor | None = None
    # Precomputed activation scales from prepare stage for quantized dispatch.
    pertoken_scale: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEMxfpParams:
    """Internal MXFP-only precision settings used by fused_moe runtime."""

    act_quant_type: torch.dtype | None = None
    weight_quant_type: torch.dtype | None = None
    scale_dtype: torch.dtype | None = None
    per_token_scale_dtype: torch.dtype | None = None
    use_bf16: bool = True


@dataclass(frozen=True, slots=True)
class MoEReservedQuantParams:
    """Internal placeholder for deferred quant runtime knobs."""

    round_mode: str = "rint"
    rollback_quant_config: dict | None = None


@dataclass(frozen=True, slots=True)
class MoEQuantParams:
    """Quant mode, backend override, and optional internal MXFP leaf config."""

    quant_type: QuantType = QuantType.NONE
    comm_quant_mode: int | None = None
    mxfp: MoEMxfpParams | None = None
    reserved: MoEReservedQuantParams = field(default_factory=MoEReservedQuantParams)

    @property
    def is_quant(self) -> bool:
        return self.quant_type != QuantType.NONE

    @property
    def is_mxfp(self) -> bool:
        return self.quant_type == QuantType.MXFP8

    @property
    def is_int_quant(self) -> bool:
        return self.quant_type in (QuantType.W8A8, QuantType.W4A8)

    @property
    def dispatch_with_quant(self) -> bool:
        return self.quant_type in (QuantType.W8A8, QuantType.W4A8, QuantType.MXFP8)


__all__ = [
    "MoERoutingParams",
    "MoEMxfpParams",
    "MoEReservedQuantParams",
    "MoEQuantParams",
]
