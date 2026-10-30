# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated DFlash2 reference: retain history required by its native consumer."""

from typing import Any

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, SlidingWindowSpec, replace_as


def apply_dflash2_reference_kv_specs(specs: dict[str, KVCacheSpec], speculator: Any) -> dict[str, KVCacheSpec]:
    if not getattr(speculator, "_dflash2_full_kv_reference", False):
        return specs
    result = specs.copy()
    for name in speculator.draft_attn_layer_names:
        spec = result[name]
        if type(spec) is SlidingWindowSpec:
            # Existing FullAttentionSpec keeps the model's native window but
            # does not recycle historical rows. Do not promote Target/GDN
            # groups or disable other engine cache optimizations.
            result[name] = replace_as(spec, FullAttentionSpec, drop=("extra_retained_tokens",))
        elif type(spec) is not FullAttentionSpec:
            raise NotImplementedError("DFlash2 reference needs an ordinary Full/SWA cache spec")
    return result
