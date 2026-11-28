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
"""Validate an online quantization request against Ascend's tables.

vLLM accepts every weight key it knows about in ``--quantization-config`` and
only fails once a layer is built with a method it cannot dispatch. Ascend's
supported recipes are a small, explicit set (``ONLINE_METHODS`` /
``ONLINE_MOE_METHODS`` in ``quantization.methods.online``), so the request is
validated from ``NPUPlatform.apply_config_platform_defaults`` -- before the model
is constructed -- and an undeclared recipe is rejected there with the supported
set in the message.

This module keeps its own imports light: it runs for every ``VllmConfig``, so
the quantization method package (and the ops package behind it) is imported only
when the request really is an online one.
"""

import functools
from typing import Any

from vllm.config import quantization as vllm_quantization
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec, quant_key_name

from vllm_ascend.device.hardware_profile import (
    QuantizationBackendFamily,
    get_current_hardware_profile,
)

__all__ = ["is_online_quantization_request", "validate_online_quantization"]

# `get_name()` of vLLM's online quantization config.
_ONLINE_QUANTIZATION_NAME = "online"


@functools.cache
def _supported_weight_keys() -> frozenset:
    """Weight keys Ascend declares a method for (imported on demand)."""
    from .methods.online import SUPPORTED_ONLINE_WEIGHT_KEYS

    return SUPPORTED_ONLINE_WEIGHT_KEYS


def _supported_names() -> str:
    return ", ".join(sorted(quant_key_name(key) for key in _supported_weight_keys()))


def _is_online_config(quant_config: Any) -> bool:
    """Whether ``quant_config`` is vLLM's online quantization config."""
    get_name = getattr(quant_config, "get_name", None)
    return callable(get_name) and get_name() == _ONLINE_QUANTIZATION_NAME


def is_online_quantization_request(quant_config: Any, online_overlay: Any) -> bool:
    """Whether the request is served by an online quantization config.

    True for vLLM's online config itself -- whatever name produced it, including
    the deprecated ``--quantization fp8`` alias -- and for a checkpoint format
    carrying an online overlay (``--quantization <checkpoint>
    --quantization-config ...``). A plain checkpoint format stays untouched.
    """
    return online_overlay is not None or _is_online_config(quant_config)


def _check_spec(spec: QuantSpec | None, where: str) -> None:
    if spec is None or spec.weight is None:
        # Activation-only specs override the *checkpoint* method; they are not
        # an online quantization request and stay upstream's business.
        return
    if spec.weight not in _supported_weight_keys():
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


def _online_args(quant_config: Any, online_overlay: Any, model_args: Any) -> QuantizationConfigArgs | None:
    """The quantization args the online request was resolved to."""
    if online_overlay is not None:
        args = online_overlay.args
    elif _is_online_config(quant_config):
        args = getattr(quant_config, "args", None)
    else:
        args = model_args
    if isinstance(args, dict):
        args = QuantizationConfigArgs(**args)
    return args if isinstance(args, QuantizationConfigArgs) else None


def validate_online_quantization(vllm_config: Any) -> None:
    """Reject an online quantization request Ascend cannot execute."""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return

    quant_config = getattr(vllm_config, "quant_config", None)
    online_overlay = getattr(quant_config, "online_quantization_config", None)
    if not is_online_quantization_request(quant_config, online_overlay):
        return

    if get_current_hardware_profile().quantization_backend_family is not QuantizationBackendFamily.STANDARD:
        raise NotImplementedError(
            "online quantization is not supported on this hardware profile: only the standard Ascend "
            "quantization backend (A2/A3/A5) ships the INT8 W8A8 schemes it reuses."
        )

    args = _online_args(quant_config, online_overlay, getattr(model_config, "quantization_config", None))
    if args is None:
        return

    _check_spec(args.linear, "linear")
    _check_spec(args.moe, "moe")
    _check_targets(args)
