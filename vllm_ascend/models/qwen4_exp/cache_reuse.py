# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp spec eligibility for layerwise buffer reuse."""

from types import MappingProxyType

from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules


def _is_gdn_state(spec: KVCacheSpec) -> bool:
    # GDN owns conv and recurrent state. PLE owns one dense conv component.
    return isinstance(spec, MambaSpec) and len(spec.shapes) == 2


QWEN4_EXP_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {
        FullAttentionSpec: True,
        MambaSpec: _is_gdn_state,
        MLAAttentionSpec: True,
        CircularBufferSpec: False,
        HiddenStateCacheSpec: False,
    }
)
