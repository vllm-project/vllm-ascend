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
"""Give vLLM's online quantization Ascend's method classes.

Why
---
vLLM's online quantization is table driven:

* ``online.base._ONLINE_LINEAR_METHODS`` / ``_ONLINE_MOE_METHODS`` map
  ``(layer kind, QuantKey)`` to the method class a layer is built with;
* a layer that no online target applies to falls back to vLLM's own unquantized
  method.

Ascend has no entries in the first table, and Ascend's unquantized methods are
the ones that apply the NPU post-load weight processing.

How
---
Write Ascend's ``ONLINE_METHODS`` / ``ONLINE_MOE_METHODS`` into the dispatch
tables and take over ``OnlineQuantizationConfig.get_quant_method``'s fallback.

vLLM's ``--quantization`` table is left alone: every upstream recipe and every
checkpoint format keeps resolving exactly as before, and a recipe Ascend cannot
execute is rejected explicitly by ``vllm_ascend.quantization.online_validate``
(called from ``NPUPlatform.apply_config_platform_defaults``, before the model is
built) rather than by removing names here. An Ascend-only ``--quantization``
name is *added* on top of vLLM's table by ``_add_ascend_online_names``.

Because nothing is removed and no config class is subclassed or registered, the
patch behaves identically on the v0.34.0 lane and on vLLM main (#64056), and it
covers the "checkpoint format + online overlay" path, which instantiates the
upstream ``OnlineQuantizationConfig`` directly in
``weight_utils.maybe_compose_online_quantization``.

``patch/platform/__init__.py`` applies this last, after the MoE factory patch, so
the Ascend ops package is imported in the documented order.

Future Plan
-----------
Delete this module, its import in ``patch/platform/__init__.py`` and entry 37 in
``patch/__init__.py`` once vLLM lets a platform contribute its own ``(QuantKey ->
method class)`` entries.
"""

import functools

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online import base as online_base

from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods.online import (
    ONLINE_METHODS,
    ONLINE_MOE_METHODS,
)

__all__ = ["apply_online_quantization_patch"]

_FALLBACK_FLAG = "_vllm_ascend_online_fallback"


def _patch_method_tables() -> None:
    """Dispatch Ascend's QuantKeys to Ascend's method classes."""
    # In place: `online.base.resolve_quant_method_cls` reads these module
    # globals at call time, and the keys double as the CLI-supported set.
    online_base._ONLINE_LINEAR_METHODS.update(ONLINE_METHODS)
    online_base._ONLINE_MOE_METHODS.update(ONLINE_MOE_METHODS)


def _add_ascend_online_names() -> None:
    """Apply Ascend's ``--quantization`` recipes on top of vLLM's table.

    An entry whose name vLLM already knows replaces vLLM's definition with the
    config Ascend tests; every other vLLM entry is kept, and a recipe Ascend
    cannot execute is rejected by ``online_validate`` rather than removed here.

    A name vLLM does not know yet (not in its ``QuantizationMethods`` Literal)
    would additionally need ``register_quantization_config(name)(...)``, which
    appends it to ``QUANTIZATION_METHODS`` and to
    ``NPUPlatform.supported_quantization``. The recipes here are all names vLLM
    already ships, so nothing has to be registered.
    """
    from vllm.config import quantization as vllm_quantization

    from vllm_ascend.quantization.methods.online import ONLINE_SHORTHANDS

    if not ONLINE_SHORTHANDS:
        return

    # In place, not a rebind: `online/base.py` imported this dict by value at
    # module import, so it has to keep seeing the same object.
    vllm_quantization._ONLINE_SHORTHANDS.update(ONLINE_SHORTHANDS)
    # The name tuple is derived at import time, so it has to be rebuilt.
    vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES = (*vllm_quantization._ONLINE_SHORTHANDS, "online")


def _patch_unquantized_fallback() -> None:
    """Keep Ascend's unquantized methods for layers online quantization skips.

    ``OnlineQuantizationConfig.get_quant_method`` falls back to vLLM's
    unquantized linear/MoE method for a layer no online target applies to.
    Ascend needs its own: they apply the NPU post-load weight processing (NZ
    layout, fp32 precast, the wo_a reshape), exactly like a layer built without a
    quant config. The fallback is decided up front so vLLM's own unquantized
    method is never constructed for those layers; a layer with an online target
    still goes through the upstream body, bookkeeping included.
    """
    original = online_base.OnlineQuantizationConfig.get_quant_method
    if getattr(original, _FALLBACK_FLAG, False):
        return

    @functools.wraps(original)
    def get_quant_method(self, layer, prefix):
        if self.resolve_quant_method_cls(type(layer), prefix) is None:
            if isinstance(layer, LinearBase):
                return AscendUnquantizedLinearMethod()
            if isinstance(layer, RoutedExperts):
                return AscendUnquantizedFusedMoEMethod(moe=layer.moe_config)
            return None
        return original(self, layer, prefix)

    setattr(get_quant_method, _FALLBACK_FLAG, True)
    online_base.OnlineQuantizationConfig.get_quant_method = get_quant_method


def apply_online_quantization_patch() -> None:
    """Apply every part of the patch. Idempotent."""
    _patch_method_tables()
    _add_ascend_online_names()
    _patch_unquantized_fallback()
