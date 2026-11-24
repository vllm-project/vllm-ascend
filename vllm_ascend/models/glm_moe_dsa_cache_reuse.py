# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GLM-5 MLA and indexer eligibility for prefill buffer reuse."""

from types import MappingProxyType

from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules

GLM_MOE_DSA_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {
        AscendMLAAttentionSpec: True,
        AscendSFAIndexerCacheSpec: True,
        MLAAttentionSpec: True,
    }
)
