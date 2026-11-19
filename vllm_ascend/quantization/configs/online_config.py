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

import torch
from vllm.config.quantization import (
    QuantizationConfigArgs,
    resolve_quantization_config,
)
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8StaticChannelSym,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_ascend.quantization.methods.online.int8 import (
    AscendInt8PerChannelOnlineLinearMethod,
    AscendInt8PerChannelOnlineMoEMethod,
)
from vllm_ascend.quantization.utils import is_fused_moe_layer
from vllm_ascend.utils import ASCEND_ONLINE_QUANT_METHODS

__all__ = [
    "AscendOnlineQuantizationConfig",
    "build_ascend_online_quant_config",
]

# Weight QuantKey -> Ascend online method class, one table per layer kind. Only
# keys with an Ascend execution path are listed; everything else is rejected
# with a message that explains which hardware would be needed.
_ASCEND_ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kInt8StaticChannelSym: AscendInt8PerChannelOnlineLinearMethod,
}
_ASCEND_ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    kInt8StaticChannelSym: AscendInt8PerChannelOnlineMoEMethod,
}


def _unquantized_method(layer: torch.nn.Module):
    """Ascend's unquantized method for a layer that online quant leaves alone."""
    if isinstance(layer, LinearBase):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        return AscendUnquantizedLinearMethod()

    if is_fused_moe_layer(layer):
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

        return AscendUnquantizedFusedMoEMethod(getattr(layer, "moe_config", None))

    if isinstance(layer, VocabParallelEmbedding):
        return UnquantizedEmbeddingMethod()

    return None


def _register_online_quant_names(cls: type) -> type:
    """Register ``cls`` under every Ascend online ``--quantization`` name.

    vLLM's ``register_quantization_config`` takes a single name, so the names
    live in ``vllm_ascend.utils.ASCEND_ONLINE_QUANT_METHODS`` (shared with the
    platform routing) and this decorator registers ``cls`` once per name.
    """
    for name in ASCEND_ONLINE_QUANT_METHODS:
        cls = register_quantization_config(name)(cls)
    return cls


@_register_online_quant_names
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
        """Resolve the Ascend online method class for a layer's quantization spec.

        Overrides the upstream table lookup so that a spec resolves to an Ascend
        method (a subclass of ``OnlineLinearBase`` / ``OnlineMoEMethodBase``,
        which vLLM instantiates and drives) instead of a CUDA-only vLLM method.

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

        if issubclass(layer_cls, (MoERunner, RoutedExperts)):
            methods = _ASCEND_ONLINE_MOE_METHODS
        elif issubclass(layer_cls, LinearBase):
            methods = _ASCEND_ONLINE_LINEAR_METHODS
        else:
            return None

        method_cls = methods.get(spec.weight)
        if method_cls is None:
            supported = sorted({str(key) for key in methods})
            raise NotImplementedError(
                f"Ascend online quantization does not support weight={spec.weight} for "
                f"{layer_cls.__name__}. Supported on this build: {supported}. INT8 is the only "
                "online recipe Ascend A2/A3 can execute; FP8 recipes need an FP8-capable device."
            )
        return method_cls

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        """Return the Ascend online method for ``layer``, if one applies.

        vLLM instantiates the class resolved above and records it in
        `quantized_layers`, so only the unquantized fallback has to be Ascend's
        own (it runs the NPU-specific post-load weight processing).
        """
        method = super().get_quant_method(layer, prefix)
        if method is None or isinstance(method, (UnquantizedLinearMethod, UnquantizedFusedMoEMethod)):
            return _unquantized_method(layer)
        return method


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
