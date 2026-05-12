#
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
#
"""Native worker hook: activate the PTO megakernel for ``chunk_gated_delta_rule``.

Called from ``vllm_ascend.patch.worker.__init__`` when
``VLLM_ASCEND_PTO_CHUNK_GDN=1``.  Must run **after** the Triton patches
install the baseline implementation and **before** any model module imports
``gdn.py``.
"""
from __future__ import annotations

import logging
import sys

_log = logging.getLogger(__name__)
_PATCH_ACTIVE = False


def apply_pto_gdn_patch() -> None:
    """Replace ``chunk_gated_delta_rule`` with the PTO megakernel.

    Patches three locations so all existing import paths see the PTO version:
    1. ``vllm_ascend.ops.triton.fla.chunk`` — primary defining module.
    2. ``vllm.model_executor.layers.fla.ops`` — vLLM public FLA namespace.
    3. ``vllm_ascend.ops.gdn`` if already imported (refreshes module attribute).
    """
    global _PATCH_ACTIVE

    import vllm.model_executor.layers.fla.ops as fla_ops
    import vllm_ascend.ops.triton.fla.chunk as _ascend_chunk_mod
    from vllm_ascend.ops.pto_chunk_gdn.chunk_gated_delta_wrapper import (
        chunk_gated_delta_rule_pto,
    )

    triton_impl = _ascend_chunk_mod.chunk_gated_delta_rule

    def _pto_bound(
        q, k, v, g, beta,
        scale=None, initial_state=None, output_final_state=False,
        cu_seqlens=None, prebuilt_meta=None, head_first=False,
        use_qk_l2norm_in_kernel=False,
    ):
        return chunk_gated_delta_rule_pto(
            q, k, v, g, beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, prebuilt_meta=prebuilt_meta,
            head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            _triton_impl=triton_impl,
        )

    _pto_bound.__name__ = "chunk_gated_delta_rule"
    _pto_bound._vllm_ascend_pto_gdn = True

    _ascend_chunk_mod.chunk_gated_delta_rule = _pto_bound
    fla_ops.chunk_gated_delta_rule = _pto_bound

    _gdn_mod = sys.modules.get("vllm_ascend.ops.gdn")
    if _gdn_mod is not None and hasattr(_gdn_mod, "chunk_gated_delta_rule"):
        _gdn_mod.chunk_gated_delta_rule = _pto_bound

    _PATCH_ACTIVE = True
    _log.warning(
        "PTO GDN megakernel active (chunk_size=128, Ascend 910B)."
    )


def is_pto_gdn_patch_active() -> bool:
    return _PATCH_ACTIVE
