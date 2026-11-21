# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi spec eligibility for layerwise buffer reuse."""

from types import MappingProxyType

from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules

# The planner excludes draft/independent layers and validates KDA state semantics.
KIMI_K3_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {
        AscendMLAAttentionSpec: True,
        MLAAttentionSpec: True,
        MambaSpec: True,
    }
)
