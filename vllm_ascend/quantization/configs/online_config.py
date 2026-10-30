#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend configuration for vLLM's online (load-time) quantization.

Online quantization turns a plain bf16/fp16 checkpoint into a quantized model
while the weights are being loaded. vLLM owns the user-facing syntax
(``--quantization`` / ``--quantization-config``), the pattern matching and the
layerwise "quantize one layer at a time" machinery; this module only decides
which Ascend scheme each layer gets, reusing the offline ModelSlim schemes so
the runtime path is shared.

Currently supported on Ascend:

* ``int8_per_channel_weight_only`` -- INT8 per-output-channel expert weights
  with dynamically per-token quantized INT8 activations (W8A8_DYNAMIC), applied
  to MoE experts. Dense linear layers stay in the checkpoint dtype, matching
  the upstream meaning of the name.

The same INT8 recipe can also be requested per layer kind::

    # experts only (equivalent to --quantization int8_per_channel_weight_only)
    --quantization online --quantization-config.moe int8_per_channel_static
    # dense linears only
    --quantization online --quantization-config.linear int8_per_channel_static

FP8 / MXFP8 online quantization is intentionally not offered here: those
recipes need FP8 matmul support that Ascend A2/A3 does not provide.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config.quantization import (
    QuantizationConfigArgs,
    resolve_quantization_config,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

from vllm_ascend.quantization.methods.online.int8 import (
    AscendOnlineInt8W8A8FusedMoEMethod,
    AscendOnlineInt8W8A8LinearMethod,
)
from vllm_ascend.quantization.online_methods import ASCEND_ONLINE_QUANT_METHODS

__all__ = [
    "ASCEND_ONLINE_QUANT_METHODS",
    "AscendOnlineQuantizationConfig",
    "build_ascend_online_quant_config",
]

# Weight QuantKey -> Ascend scheme class. Only keys that have an Ascend
# execution path are listed; everything else is rejected with a message that
# explains which hardware would be needed.
_ASCEND_ONLINE_LINEAR_SCHEMES: dict[Any, type] = {}
_ASCEND_ONLINE_MOE_SCHEMES: dict[Any, type] = {}


def _scheme_tables() -> tuple[dict[Any, type], dict[Any, type]]:
    """Build the QuantKey -> scheme tables.

    Deferred because they need the QuantKey singletons from
    ``vllm.model_executor.layers.quantization.utils.quant_utils``, which drags
    in the kernel registry. The tables are built once and cached.
    """
    if _ASCEND_ONLINE_LINEAR_SCHEMES and _ASCEND_ONLINE_MOE_SCHEMES:
        return _ASCEND_ONLINE_LINEAR_SCHEMES, _ASCEND_ONLINE_MOE_SCHEMES

    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8StaticChannelSym,
    )

    _ASCEND_ONLINE_LINEAR_SCHEMES[kInt8StaticChannelSym] = AscendOnlineInt8W8A8LinearMethod
    _ASCEND_ONLINE_MOE_SCHEMES[kInt8StaticChannelSym] = AscendOnlineInt8W8A8FusedMoEMethod
    return _ASCEND_ONLINE_LINEAR_SCHEMES, _ASCEND_ONLINE_MOE_SCHEMES


def _is_moe_layer(layer: torch.nn.Module) -> bool:
    from vllm_ascend.quantization.utils import is_fused_moe_layer

    return is_fused_moe_layer(layer)


def _is_moe_layer_cls(layer_cls: type) -> bool:
    """Class-level counterpart of :func:`_is_moe_layer`.

    vLLM resolves online targets from the layer type
    (``resolve_quant_method_cls(type(layer), prefix)``), so the Ascend scheme
    lookup has to answer the same question without an instance.
    """
    from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts

    return issubclass(layer_cls, (MoERunner, RoutedExperts))


def _unquantized_method(layer: torch.nn.Module):
    """Ascend's unquantized method for a layer that online quant leaves alone."""
    if isinstance(layer, LinearBase):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        return AscendUnquantizedLinearMethod()

    if _is_moe_layer(layer):
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

        return AscendUnquantizedFusedMoEMethod(getattr(layer, "moe_config", None))

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    if isinstance(layer, VocabParallelEmbedding):
        return UnquantizedEmbeddingMethod()

    return None


@register_quantization_config("online")
@register_quantization_config("int8_per_channel_weight_only")
class AscendOnlineQuantizationConfig(OnlineQuantizationConfig):
    """Online quantization for Ascend, driven by the shared vLLM frontend."""

    def __init__(self, args: QuantizationConfigArgs) -> None:
        from vllm_ascend.device.hardware_profile import (
            QuantizationBackendFamily,
            get_current_hardware_profile,
        )

        if get_current_hardware_profile().quantization_backend_family is not QuantizationBackendFamily.STANDARD:
            raise NotImplementedError(
                "Online quantization is not supported on this hardware profile: only the "
                "standard Ascend quantization backend (A2/A3/A5) ships the INT8 W8A8 "
                "schemes it reuses."
            )
        super().__init__(args)

    @classmethod
    def get_min_capability(cls) -> int:
        # Ascend reports no compute-capability tuple, so the upstream check is
        # skipped; keep this callable for callers that probe it unconditionally.
        return 0

    def _get_method_cls(self, spec, table, layer_cls: type) -> type | None:
        """Resolve the Ascend scheme class for a layer's quantization spec.

        Overrides the upstream table lookup so that a spec resolves to an
        Ascend scheme instead of a CUDA-only vLLM method class.

        This is vLLM's only dispatch hook for online methods, so it is pinned by
        ``test_override_still_matches_the_upstream_signature``. The third
        argument became the layer's *type* in vLLM #55686 (``layer`` ->
        ``layer_cls``); both spellings are accepted so that a revert or a
        forward-port keeps working, and other layers keep the upstream default.
        """
        if not isinstance(layer_cls, type):
            layer_cls = type(layer_cls)

        if spec is None or spec.weight is None:
            return None

        if spec.activation is not None:
            raise ValueError(
                f"activation override (activation={spec.activation}) is not supported by Ascend "
                "online quantization: the Ascend W8A8 kernels always quantize activations "
                "dynamically per token."
            )

        linear_schemes, moe_schemes = _scheme_tables()
        if _is_moe_layer_cls(layer_cls):
            schemes = moe_schemes
        elif issubclass(layer_cls, LinearBase):
            schemes = linear_schemes
        else:
            return None

        scheme_cls = schemes.get(spec.weight)
        if scheme_cls is None:
            supported = sorted({str(key) for key in schemes})
            raise NotImplementedError(
                f"Ascend online quantization does not support weight={spec.weight} for "
                f"{layer_cls.__name__}. Supported on this build: {supported}. INT8 is the only "
                "online recipe Ascend A2/A3 can execute; FP8 recipes need an FP8-capable device."
            )
        return scheme_cls

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        """Return the Ascend quant method for ``layer``."""
        from vllm_ascend.quantization.method_adapters import (
            AscendOnlineFusedMoEMethod,
            AscendOnlineLinearMethod,
        )

        resolved = self.resolve_quant_method_cls(type(layer), prefix)
        if resolved is None:
            return _unquantized_method(layer)

        source, quant_key_str, target_pattern, _, scheme_cls = resolved
        self.quantized_layers[prefix] = (source.value, quant_key_str, target_pattern)

        scheme = scheme_cls()
        if _is_moe_layer(layer):
            return AscendOnlineFusedMoEMethod(scheme, layer.moe_config)
        return AscendOnlineLinearMethod(scheme)


def build_ascend_online_quant_config(model_config) -> AscendOnlineQuantizationConfig:
    """Instantiate the Ascend online quantization config for ``model_config``.

    Entry point of ``patch_online_quant``: vLLM's loader hardcodes
    ``OnlineQuantizationConfig`` on the online path, so Ascend resolves the
    arguments and builds the config itself. This also performs the validation
    that ``VllmConfig._get_quantization_config`` would otherwise apply, since
    the patch replaces that call.
    """
    quantization = model_config.quantization
    args = getattr(model_config, "quantization_config", None)
    if args is None:
        args = resolve_quantization_config(quantization, None)
    elif isinstance(args, dict):
        args = QuantizationConfigArgs(**args)

    if args is None:
        raise ValueError(
            f"--quantization {quantization} needs an explicit --quantization-config, for example "
            "--quantization-config.moe int8_per_channel_static (experts only) or "
            "--quantization-config.linear int8_per_channel_static (dense layers only)."
        )
    if not isinstance(args, QuantizationConfigArgs):
        raise TypeError(
            f"--quantization {quantization} expects a quantization config object, got {type(args).__name__}."
        )

    config = AscendOnlineQuantizationConfig(args)
    supported_dtypes = config.get_supported_act_dtypes()
    model_dtype = getattr(model_config, "dtype", None)
    if model_dtype is not None and model_dtype not in supported_dtypes:
        raise ValueError(
            f"{model_dtype} is not supported for online quantization {quantization}; "
            f"supported activation dtypes: {supported_dtypes}."
        )
    return config
