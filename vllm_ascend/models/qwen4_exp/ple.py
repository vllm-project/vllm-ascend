# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Ascend runtime adaptation for Qwen4Exp PLE short convolution."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch_npu
from vllm.logger import logger


def _get_allow_internal_format() -> bool:
    raw_value = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    # NPUModelRunner enables this option before model construction. Preserve
    # the torch-npu default when the model is instantiated in isolation.
    return raw_value is None or raw_value.decode() == "enable"


def _set_allow_internal_format(enabled: bool) -> None:
    # Prefer the high-level config sugar where torch_npu provides it.
    config = getattr(torch.npu, "config", None)
    if config is not None and hasattr(config, "allow_internal_format"):
        config.allow_internal_format = enabled
        return
    # torch_npu 2.10 (910B) ships an empty torch.npu.config module: assigning
    # the attribute would succeed silently without toggling anything. Fall
    # back to the low-level option API, which is present there.
    set_option = getattr(torch_npu._C, "_npu_setOption", None)  # noqa: SLF001
    if set_option is not None:
        set_option({"ALLOW_INTERNAL_FORMAT": "enable" if enabled else "disable"})
        return
    logger.warning_once(
        "No ALLOW_INTERNAL_FORMAT setter available on this torch_npu version; "
        "the PLE short convolution follows the process-wide setting."
    )


def _wrap_ple_short_conv(original: Callable[..., Any]) -> Callable[..., Any]:
    """Run only the PLE depthwise convolution with internal formats disabled.

    With internal formats enabled, torch-npu lowers PLE's dilated depthwise
    ``F.conv1d`` to the legacy ACLop Conv2D path. ACLop operators cannot run
    during NPU graph capture. Disabling internal formats selects the graph-safe
    path, but changing the process-wide option permanently would regress other
    Ascend kernels. Keep the override scoped to the upstream PLE custom op and
    restore the previous value even when convolution raises.
    """

    @wraps(original)
    def wrapped(self: Any, inputs: torch.Tensor) -> torch.Tensor:
        previous = _get_allow_internal_format()
        _set_allow_internal_format(False)
        try:
            return original(self, inputs)
        finally:
            _set_allow_internal_format(previous)

    return wrapped


def patch_upstream_ple_short_conv() -> None:
    """Install the scoped PLE override once on the upstream model class."""
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer

    original_attr = "_ascend_original_short_conv"
    if hasattr(Qwen4ExpPLELayer, original_attr):
        return
    original = Qwen4ExpPLELayer._short_conv
    setattr(Qwen4ExpPLELayer, original_attr, original)
    Qwen4ExpPLELayer._short_conv = _wrap_ple_short_conv(original)


__all__ = ["patch_upstream_ple_short_conv"]
