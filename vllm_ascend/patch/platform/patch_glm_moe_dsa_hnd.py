# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Apply the GLM-MoE-DSA planner's physical layout during negotiation."""

from functools import wraps

from vllm import envs
from vllm.v1.attention.backends import utils as attention_utils
from vllm.v1.engine import core as engine_core

from vllm_ascend.models import glm_moe_dsa_dflash_hnd as planner

_resolve = attention_utils.resolve_kv_cache_layout


@wraps(_resolve)
def resolve_kv_cache_layout(vllm_config, supported_layouts, kv_cache_specs=None):
    requested = getattr(envs, "VLLM_KV_CACHE_LAYOUT", None)
    if requested in ("HND", "LBHNC") and planner.applies_to(vllm_config, requested) and kv_cache_specs is not None:
        # These target layers have per-spec page strides. The generic
        # shared-page check still applies to the draft and any other layers.
        kv_cache_specs = [
            entry for entry in kv_cache_specs if not planner.is_target_spec(getattr(entry, "spec", entry))
        ]
    return _resolve(vllm_config, supported_layouts, kv_cache_specs)


# EngineCore holds its own reference to the function imported at startup.
engine_core.resolve_kv_cache_layout = attention_utils.resolve_kv_cache_layout = resolve_kv_cache_layout
