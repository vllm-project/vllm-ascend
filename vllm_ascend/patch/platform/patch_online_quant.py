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
"""Route Ascend's online-quantization method names to the Ascend config class.

Why
---
``VllmConfig._get_quantization_config`` -- the single funnel from
``--quantization`` to a ``QuantizationConfig`` -- ends in
``vllm.model_executor.model_loader.weight_utils.get_quant_config``, which
builds the concrete upstream ``OnlineQuantizationConfig`` at each of its online
return sites. One of those sites also guards with an identity test
(``quant_cls is OnlineQuantizationConfig``). Construction is therefore
hardcoded, and a subclass registered through ``register_quantization_config``
for the same ``--quantization`` name is never instantiated: the registered
Ascend config would be silently replaced by vLLM's, whose method classes select
CUDA/ROCm kernels.

How
---
Wrap ``VllmConfig._get_quantization_config`` and divert only the method names
the Ascend platform implements (``online``, ``int8_per_channel_weight_only``) to
``build_ascend_online_quant_config``. Every other quantization method --
checkpoint formats such as ModelSlim, compressed-tensors, Quark and ModelOpt
included -- still takes the upstream path untouched.

Patching the config funnel rather than ``get_quant_config`` keeps this module
clear of the model-loader import graph, which the platform hook must not pull in
this early.

Future Plan
-----------
Remove this module, its import in ``patch/platform/__init__.py`` and entry 37 in
``patch/__init__.py`` once upstream instantiates the class resolved by
``get_quantization_config`` on the online path. ``AscendOnlineQuantizationConfig``
is already registered for both method names, so no other change is needed: the
upstream dispatch takes over as soon as this patch is gone.
"""

import functools

from vllm.config.vllm import VllmConfig

from vllm_ascend.quantization.online_methods import ASCEND_ONLINE_QUANT_METHODS

__all__ = ["patch_online_quantization_config"]

_PATCHED_FLAG = "_vllm_ascend_online_quant_patched"


def patch_online_quantization_config() -> None:
    """Divert Ascend's online quantization method names to the Ascend config."""
    original = VllmConfig._get_quantization_config
    if getattr(original, _PATCHED_FLAG, False):
        return

    @functools.wraps(original)
    def _get_quantization_config(model_config, load_config):
        quantization = getattr(model_config, "quantization", None)
        if quantization in ASCEND_ONLINE_QUANT_METHODS:
            # Imported lazily: the quantization package must not be pulled in
            # while the platform class itself is still being imported.
            from vllm_ascend.quantization.configs.online_config import (
                build_ascend_online_quant_config,
            )

            return build_ascend_online_quant_config(model_config)
        return original(model_config, load_config)

    setattr(_get_quantization_config, _PATCHED_FLAG, True)
    VllmConfig._get_quantization_config = staticmethod(_get_quantization_config)


patch_online_quantization_config()
