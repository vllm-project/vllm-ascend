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
"""Give vLLM's online quantization Ascend's config, methods and recipe names.

Why
---
vLLM's online quantization is table driven:

* ``online.base._ONLINE_LINEAR_METHODS`` / ``_ONLINE_MOE_METHODS`` map
  ``(layer kind, QuantKey)`` to the method class a layer is built with;
* ``config.quantization`` maps the ``--quantization`` names to default configs;
* the online config class itself is constructed at four sites of
  ``weight_utils.get_quant_config`` -- the plain online path, the shorthand
  paths, the "checkpoint format + online overlay" path
  (``maybe_compose_online_quantization``) and the deprecated
  ``--quantization fp8`` alias. All of them hardcode
  ``OnlineQuantizationConfig`` (the v0.34.0 lane additionally compares it by
  identity), so a subclass registered through ``register_quantization_config``
  is never instantiated.

How
---
1. Write Ascend's ``ONLINE_LINEAR_METHODS`` / ``ONLINE_MOE_METHODS`` into the
   dispatch tables, so vLLM's own ``_get_method_cls`` resolves Ascend's classes.
2. Rebind ``online.base.OnlineQuantizationConfig`` to
   ``AscendOnlineQuantizationConfig``. Every construction site imports the class
   *inside* its function, so they all pick the rebound symbol up, and the
   v0.34.0 identity test still holds (both sides resolve to the subclass). The
   subclass carries the request validation and the unquantized fallback, so no
   other method is monkey-patched.
3. Apply Ascend's ``--quantization`` recipes on top of vLLM's table
   (``int8_per_channel``); vLLM's other entries are kept and a recipe Ascend
   cannot execute is rejected by the subclass, not removed here.

Because vLLM's table is only extended and its class is only replaced by a
subclass, the patch behaves identically on the v0.34.0 lane and on vLLM main
(#64056), including the "checkpoint format + online overlay" path.

``patch/platform/__init__.py`` applies this last, after the MoE factory patch, so
the Ascend ops package is imported in the documented order.

Future Plan
-----------
Delete this module, its import in ``patch/platform/__init__.py`` and entry 37 in
``patch/__init__.py`` once vLLM lets a platform contribute its own online
quantization config class and ``(QuantKey -> method class)`` entries.
"""

from vllm.model_executor.layers.quantization.online import base as online_base

from vllm_ascend.quantization.configs.online_config import AscendOnlineQuantizationConfig
from vllm_ascend.quantization.methods.online import (
    ONLINE_LINEAR_METHODS,
    ONLINE_MOE_METHODS,
)

__all__ = ["apply_online_quantization_patch"]


def _patch_method_tables() -> None:
    """Dispatch Ascend's QuantKeys to Ascend's method classes."""
    # In place: `online.base.resolve_quant_method_cls` reads these module
    # globals at call time, and the keys double as the CLI-supported set.
    online_base._ONLINE_LINEAR_METHODS.update(ONLINE_LINEAR_METHODS)
    online_base._ONLINE_MOE_METHODS.update(ONLINE_MOE_METHODS)


def _patch_online_config_class() -> None:
    """Make vLLM construct Ascend's online config at every online site."""
    # Both `weight_utils.get_quant_config` and `get_quantization_config` import
    # the class inside the function, so the rebind reaches them, and the
    # v0.34.0 `quant_cls is OnlineQuantizationConfig` test still holds.
    if online_base.OnlineQuantizationConfig is AscendOnlineQuantizationConfig:
        return
    online_base.OnlineQuantizationConfig = AscendOnlineQuantizationConfig


def _add_ascend_online_names() -> None:
    """Apply Ascend's ``--quantization`` recipes on top of vLLM's table.

    An entry whose name vLLM already knows replaces vLLM's definition with the
    config Ascend tests; every other vLLM entry is kept, and a recipe Ascend
    cannot execute is rejected by ``AscendOnlineQuantizationConfig`` rather than
    removed here.

    A name vLLM does not know yet (``int8_per_channel`` today) also has to join
    ``QUANTIZATION_METHODS`` so ``--quantization`` accepts it. The platform
    whitelist lists it through ``ASCEND_ONLINE_QUANT_METHODS``, and
    ``get_quantization_config`` resolves its class through the shorthand loop.
    """
    from vllm.config import quantization as vllm_quantization
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    from vllm_ascend.quantization.methods.online import ONLINE_SHORTHANDS

    if not ONLINE_SHORTHANDS:
        return

    # In place, not a rebind: `online/base.py` imported this dict by value at
    # module import, so it has to keep seeing the same object.
    vllm_quantization._ONLINE_SHORTHANDS.update(ONLINE_SHORTHANDS)
    # The name tuple is derived at import time, so it has to be rebuilt.
    vllm_quantization.ONLINE_QUANT_SHORTHAND_NAMES = (*vllm_quantization._ONLINE_SHORTHANDS, "online")
    for name in ONLINE_SHORTHANDS:
        if name not in QUANTIZATION_METHODS:
            QUANTIZATION_METHODS.append(name)


def apply_online_quantization_patch() -> None:
    """Apply every part of the patch. Idempotent."""
    _patch_method_tables()
    _patch_online_config_class()
    _add_ascend_online_names()
