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
"""Ascend's online (load-time) quantization config.

Ascend reuses vLLM's online quantization machinery: ``OnlineQuantizationConfig``
owns the per-layer weight staging and the ``--quantization`` /
``--quantization-config`` frontend, and ``patch_online_quant`` writes Ascend's
``(layer kind, QuantKey) -> method class`` entries into vLLM's dispatch tables.
This subclass adds the two things a table cannot express:

* the request is validated when the config is built, so an unsupported recipe,
  an activation override or a non-standard quantization backend fails before any
  weight is loaded, with the supported set in the message;
* a layer that online quantization leaves alone gets Ascend's unquantized
  method, which applies the NPU post-load weight processing (NZ layout, fp32
  precast, the wo_a reshape).

``patch_online_quant`` rebinds ``online.base.OnlineQuantizationConfig`` to this
class, so vLLM's construction sites -- the plain online path, the shorthands,
the "checkpoint format + online overlay" path and the deprecated
``--quantization fp8`` alias -- instantiate it directly.
"""

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec, quant_key_name
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

from vllm_ascend.device.hardware_profile import (
    QuantizationBackendFamily,
    get_current_hardware_profile,
)
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods.online import SUPPORTED_ONLINE_WEIGHT_KEYS

__all__ = ["AscendOnlineQuantizationConfig"]


def _supported_names() -> str:
    return ", ".join(sorted(quant_key_name(key) for key in SUPPORTED_ONLINE_WEIGHT_KEYS))


def _check_spec(spec: QuantSpec | None, where: str) -> None:
    if spec is None or spec.weight is None:
        # Activation-only specs override the *checkpoint* method; they are not
        # an online quantization request and stay vLLM's business.
        return
    if spec.weight not in SUPPORTED_ONLINE_WEIGHT_KEYS:
        raise ValueError(
            f"online quantization on Ascend does not support {where} weight={quant_key_name(spec.weight)}; "
            f"supported: {_supported_names()}"
        )
    if "activation" in spec.fields_set:
        # Mirrors `OnlineRequantizationMixin.resolve_activation_quant_key`: the
        # Ascend W8A8 kernels always quantize activations dynamically per
        # token, so there is no activation key to honour.
        raise NotImplementedError(
            f"online quantization on Ascend does not support {where} activation overrides: the Ascend W8A8 "
            "kernels always quantize activations dynamically per token"
        )


def _check_targets(args: QuantizationConfigArgs) -> None:
    from vllm.config import quantization as vllm_quantization

    for pattern, target in (args.targets or {}).items():
        if isinstance(target, str):
            shorthand = vllm_quantization._ONLINE_SHORTHANDS.get(target)
            if shorthand is None:
                # vLLM already rejected unknown `targets` names.
                continue
            _check_spec(shorthand.linear, f"targets[{pattern}] linear")
            _check_spec(shorthand.moe, f"targets[{pattern}] moe")
        else:
            _check_spec(target, f"targets[{pattern}]")


def _validate_args(args: QuantizationConfigArgs) -> None:
    _check_spec(args.linear, "linear")
    _check_spec(args.moe, "moe")
    _check_targets(args)


class AscendOnlineQuantizationConfig(OnlineQuantizationConfig):
    """vLLM's online quantization config with Ascend's validation and methods."""

    def __init__(self, args: QuantizationConfigArgs | None = None) -> None:
        if args is None:
            # `--quantization online` without a recipe: vLLM reaches this with a
            # bare `quant_cls()` call, so give the actionable message here.
            raise ValueError(
                "--quantization online needs an explicit --quantization-config, for example "
                "--quantization-config.moe int8_per_channel_static (experts only) or "
                "--quantization-config.linear int8_per_channel_static (dense layers only)."
            )
        if not isinstance(args, QuantizationConfigArgs):
            raise TypeError(
                f"online quantization on Ascend expects a QuantizationConfigArgs, got {type(args).__name__}."
            )
        if get_current_hardware_profile().quantization_backend_family is not QuantizationBackendFamily.STANDARD:
            raise NotImplementedError(
                "online quantization is not supported on this hardware profile: only the standard Ascend "
                "quantization backend (A2/A3/A5) ships the INT8 W8A8 schemes it reuses."
            )
        _validate_args(args)
        super().__init__(args)

    @classmethod
    def get_min_capability(cls) -> int:
        # Ascend reports no CUDA-style capability tuple, so vLLM's check is
        # skipped; keep this callable for code that probes it unconditionally.
        return 0

    def get_quant_method(self, layer, prefix):
        if self.resolve_quant_method_cls(type(layer), prefix) is None:
            if isinstance(layer, LinearBase):
                return AscendUnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                return AscendUnquantizedFusedMoEMethod(moe=layer.moe_config)
            return None
        return super().get_quant_method(layer, prefix)
