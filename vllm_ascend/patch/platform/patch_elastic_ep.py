# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Allow --enable-elastic-ep on Ascend NPU.

Upstream requires ``enable_eplb=True`` when ``enable_elastic_ep=True``,
and ``enable_eplb`` is gated by ``current_platform.is_cuda_alike()``.

.. important::
   ``current_platform.is_cuda_alike``, ``enable_eplb``, and
   ``eplb_config.use_async`` are modified **temporarily** (set to
   ``True`` / ``True`` / ``False`` respectively) only to pass
   ``_validate_parallel_config``, and are **restored** to the original
   values in a ``finally`` block after ``__init__`` completes.  The
   modified values never leak into runtime logic.

   Additionally, the ``FusedMoE`` factory (called during model
   construction) is wrapped to force ``enable_eplb=True`` when
   ``enable_elastic_ep`` is True, because ``FusedMoE`` asserts
   ``num_redundant_experts == 0`` when ``enable_eplb`` is False.

.. note::
   This patch must be imported **before** any ``ParallelConfig`` is
   constructed, otherwise ``enable_eplb`` / ``enable_elastic_ep``
   inference will not take effect during validation.
"""

from vllm.config.parallel import ParallelConfig, EPLBConfig
from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# When enable_elastic_ep=True, temporarily (a) replace
# current_platform.is_cuda_alike with ``lambda: True``, (b) set
# enable_eplb=True, and (c) set eplb_config.use_async=False, so that
# _validate_parallel_config passes.  All three are restored to their
# original values in a ``finally`` block after __init__ completes,
# ensuring the modified values never leak into runtime logic.
# ---------------------------------------------------------------------------
_original_init = ParallelConfig.__init__
_original_is_cuda_alike = current_platform.is_cuda_alike


def _patched_init(self, **data: object):
    if data.get("enable_elastic_ep", False):
        current_platform.is_cuda_alike = lambda: True

        _orig_enable_eplb = data.get("enable_eplb", False)
        _orig_eplb_cfg = data.get("eplb_config")
        if _orig_eplb_cfg is None:
            _orig_use_async = True
        elif isinstance(_orig_eplb_cfg, EPLBConfig):
            _orig_use_async = _orig_eplb_cfg.use_async
        else:
            _orig_use_async = _orig_eplb_cfg.get("use_async", True)

        data["enable_eplb"] = True
        if _orig_eplb_cfg is None:
            data["eplb_config"] = EPLBConfig(use_async=False)
        elif isinstance(_orig_eplb_cfg, EPLBConfig):
            _orig_eplb_cfg.use_async = False
        else:
            _orig_eplb_cfg["use_async"] = False

        try:
            _original_init(self, **data)
        finally:
            current_platform.is_cuda_alike = _original_is_cuda_alike
            self.enable_eplb = _orig_enable_eplb
            self.eplb_config.use_async = _orig_use_async
    else:
        _original_init(self, **data)


ParallelConfig.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Patch FusedMoE factory to force enable_eplb=True when
# enable_elastic_ep=True.
#
# After __init__ restores enable_eplb to the user-provided value (or the
# upstream default False), model construction calls FusedMoE, which
# asserts ``num_redundant_experts == 0`` when enable_eplb is False
# (layer.py:258).  Since ascend_config.py may have set
# num_redundant_experts > 0 from additional_config, the assertion fires.
#
# We wrap the already-patched FusedMoE (set to _ascend_FusedMoE by
# patch_fused_moe.py) so that enable_eplb=True is in effect for the
# duration of the factory call, then the caller's original config value
# persists for all other code paths.
# ---------------------------------------------------------------------------
import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer

_original_fused_moe = _fused_moe_layer.FusedMoE


def _patched_fused_moe(*args, **kwargs):
    from vllm.config import get_current_vllm_config

    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        vllm_config = None

    if vllm_config is not None and vllm_config.parallel_config.enable_elastic_ep:
        kwargs["enable_eplb"] = True

    return _original_fused_moe(*args, **kwargs)


_fused_moe_layer.FusedMoE = _patched_fused_moe
_fused_moe_pkg.FusedMoE = _patched_fused_moe