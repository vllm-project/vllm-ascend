# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capability-driven adapter for the external CANN MegaMoe operator.

The adapter deliberately reasons about the instantiated MoE layer instead of
checkpoint metadata or model names. This keeps communication selection
independent of how a quantized checkpoint describes the same runtime layout.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig


_LAYER_CAPABILITY_ATTR = "cann_mega_moe_capability"
_MODEL_CAPABILITY_ATTR = "_ascend_cann_mega_moe_capability"

_A3_QUANT_TYPES = frozenset({QuantType.NONE, QuantType.W8A8, QuantType.W4A8})
_A5_QUANT_TYPES = frozenset({QuantType.W4A8MXFP})
_SWIGLU_ACTIVATIONS = frozenset({"silu", "swiglu", "moeactivation.silu"})
_SITU_ACTIVATIONS = frozenset({"situ", "situglu", "moeactivation.situ"})
_COMMON_MEGA_MOE_KEYWORDS = frozenset(
    {
        "activation_clamp",
        "l1_bias",
        "l1_weights_sf",
        "l2_bias",
        "l2_weights_sf",
        "weight1_type",
        "weight2_type",
        "x_active_mask",
    }
)


@dataclass(frozen=True, slots=True)
class CannMegaMoeApiCapability:
    available: bool
    supports_situ: bool = False
    supports_comm_context_preload: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CannMegaMoeActivation:
    name: str
    alpha: float | None = None
    beta: float | None = None


@dataclass(frozen=True, slots=True)
class CannMegaMoeLayerCapability:
    supported: bool
    reason: str
    quant_type: QuantType
    activation: CannMegaMoeActivation | None = None


@cache
def probe_cann_mega_moe_api() -> CannMegaMoeApiCapability:
    """Probe the Python API surface without compiling the comm extension."""
    try:
        ops_module = import_module("cann_ops_transformer.ops")
        mega_moe = ops_module.mega_moe
        _ = ops_module.get_symm_buffer_for_mega_moe
        parameters = inspect.signature(mega_moe).parameters
        accepts_arbitrary_keywords = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if not accepts_arbitrary_keywords and not _COMMON_MEGA_MOE_KEYWORDS.issubset(parameters):
            missing = sorted(_COMMON_MEGA_MOE_KEYWORDS.difference(parameters))
            raise TypeError(f"mega_moe is missing keyword parameters: {missing}")
    except Exception as exc:
        return CannMegaMoeApiCapability(False, reason=f"incompatible cann_ops_transformer API: {exc}")

    try:
        comm_context_module = import_module("cann_ops_transformer.ops.comm_context")
        comm_context_builder = comm_context_module.comm_context_op_builder
        _ = comm_context_builder.load
        supports_comm_context_preload = True
    except Exception:
        # The A2/A3 operator does not require this extension. Keep the base
        # backend available and let the A5 capability check reject preloading.
        supports_comm_context_preload = False

    return CannMegaMoeApiCapability(
        True,
        supports_situ={"activation", "activation_params"}.issubset(parameters),
        supports_comm_context_preload=supports_comm_context_preload,
    )


def resolve_cann_mega_moe_activation(
    activation: Any,
    *,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
) -> CannMegaMoeActivation | None:
    """Normalize both legacy activation objects and main's MoEActivation."""
    activation_name = str(getattr(activation, "value", activation)).lower()
    linear_beta = getattr(activation, "linear_beta", situ_linear_beta)
    beta = getattr(activation, "beta", situ_beta)

    if activation_name in _SITU_ACTIVATIONS or linear_beta is not None:
        if linear_beta is None:
            return None
        return CannMegaMoeActivation(name="situglu", alpha=linear_beta, beta=beta)

    if activation_name in _SWIGLU_ACTIVATIONS:
        return CannMegaMoeActivation(name="swiglu")
    return None


def _unsupported(reason: str, quant_type: QuantType) -> CannMegaMoeLayerCapability:
    return CannMegaMoeLayerCapability(False, reason, quant_type)


def evaluate_cann_mega_moe_layer(
    moe_config: FusedMoEConfig,
    quant_method: object,
    activation: Any,
    *,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
    device_type: AscendDeviceType | None = None,
    api_capability: CannMegaMoeApiCapability | None = None,
) -> CannMegaMoeLayerCapability:
    """Evaluate an instantiated MoE layer against the operator contract."""
    device_type = device_type or get_ascend_device_type()
    api_capability = api_capability or probe_cann_mega_moe_api()
    # ModelSlim wraps the concrete MoE scheme in AscendFusedMoEMethod, which
    # delegates to an inner scheme that carries quant_type / group_size.
    inner_scheme = getattr(quant_method, "quant_method", None)
    if inner_scheme is not None and hasattr(inner_scheme, "quant_type"):
        quant_method = inner_scheme
    quant_type = getattr(quant_method, "quant_type", QuantType.NONE)

    if not api_capability.available:
        return _unsupported(api_capability.reason, quant_type)

    if device_type == AscendDeviceType.A5:
        if quant_type not in _A5_QUANT_TYPES:
            return _unsupported(f"A5 MegaMoe does not support {quant_type.name}", quant_type)
        if getattr(quant_method, "group_size", None) != 32:
            return _unsupported("A5 MXFP MegaMoe requires group_size=32", quant_type)
        if not api_capability.supports_comm_context_preload:
            return _unsupported("A5 MegaMoe requires comm-context preloading", quant_type)
    elif device_type in {AscendDeviceType.A2, AscendDeviceType.A3}:
        if quant_type not in _A3_QUANT_TYPES:
            return _unsupported(f"A2/A3 MegaMoe does not support {quant_type.name}", quant_type)
    else:
        return _unsupported(f"MegaMoe is not supported on {device_type}", quant_type)

    resolved_activation = resolve_cann_mega_moe_activation(
        activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if resolved_activation is None:
        return _unsupported(f"unsupported MegaMoe activation: {activation}", quant_type)
    if resolved_activation.name == "situglu" and not api_capability.supports_situ:
        return _unsupported("installed MegaMoe API does not expose SiTU parameters", quant_type)
    if resolved_activation.name == "situglu" and device_type != AscendDeviceType.A5:
        return _unsupported("SiTU MegaMoe is supported only on A5", quant_type)

    in_dtype = getattr(moe_config, "in_dtype", None)
    if in_dtype not in {torch.bfloat16, torch.float16}:
        return _unsupported(f"unsupported MegaMoe input dtype: {in_dtype}", quant_type)

    hidden = int(moe_config.hidden_dim)
    if hidden < 1024 or hidden > 8192 or hidden % 512 != 0:
        return _unsupported(f"unsupported MegaMoe hidden size: {hidden}", quant_type)

    intermediate_hidden = 2 * int(moe_config.intermediate_size_per_partition)
    if intermediate_hidden < 1024 or intermediate_hidden % 512 != 0:
        return _unsupported(f"unsupported MegaMoe intermediate size: {intermediate_hidden}", quant_type)

    num_topk = int(moe_config.experts_per_token)
    max_topk = 32 if device_type == AscendDeviceType.A5 else 16
    if num_topk < 1 or num_topk > max_topk:
        return _unsupported(f"unsupported MegaMoe top-k: {num_topk}", quant_type)

    ep_size = int(moe_config.ep_size)
    max_ep_size = 1024 if device_type == AscendDeviceType.A5 else 64
    if ep_size < 2 or ep_size > max_ep_size:
        return _unsupported(f"unsupported MegaMoe EP size: {ep_size}", quant_type)

    num_experts = int(moe_config.num_experts)
    max_num_experts = 2048 if device_type == AscendDeviceType.A5 else 1024
    if num_experts < ep_size or num_experts > max_num_experts or num_experts % ep_size != 0:
        return _unsupported(
            f"MegaMoe requires experts divisible by EP size, got experts={num_experts}, EP={ep_size}",
            quant_type,
        )

    return CannMegaMoeLayerCapability(True, "", quant_type, resolved_activation)


def get_model_cann_mega_moe_capability(model_instance: torch.nn.Module | None) -> CannMegaMoeLayerCapability:
    """Aggregate immutable layer capabilities and cache the model result."""
    if model_instance is None:
        return _unsupported("model instance is unavailable", QuantType.NONE)

    cached = getattr(model_instance, _MODEL_CAPABILITY_ATTR, None)
    if isinstance(cached, CannMegaMoeLayerCapability):
        return cached

    layer_capabilities = [
        capability
        for module in model_instance.modules()
        if isinstance((capability := getattr(module, _LAYER_CAPABILITY_ATTR, None)), CannMegaMoeLayerCapability)
    ]
    if not layer_capabilities:
        # The model can be queried while its module tree is still being built.
        # Do not cache this transient result.
        return _unsupported("model has no registered MegaMoe-capable layers", QuantType.NONE)

    first_unsupported = next((capability for capability in layer_capabilities if not capability.supported), None)
    result = first_unsupported or layer_capabilities[0]
    setattr(model_instance, _MODEL_CAPABILITY_ATTR, result)
    return result
