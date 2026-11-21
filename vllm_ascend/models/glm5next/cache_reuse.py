# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM cache spec eligibility for layerwise buffer reuse."""

from types import MappingProxyType

import torch
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules
from vllm_ascend.models.glm5next.cache_config import _is_glm5_next_main_spec

GLM5_NEXT_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {
        AscendMLAAttentionSpec: _is_glm5_next_main_spec,
        MLAAttentionSpec: _is_glm5_next_main_spec,
        MambaSpec: True,
        AscendIndexerKPoolTailSpec: False,
    }
)


def _is_fp8_main_cache(spec: KVCacheSpec) -> bool:
    return isinstance(spec, AscendMLAAttentionSpec) and spec.cache_sparse_sfa_c8 and spec.dtype == torch.float8_e4m3fn


def _is_fp8_indexer_cache(spec: KVCacheSpec) -> bool:
    return isinstance(spec, AscendSFAIndexerCacheSpec) and spec.cache_sparse_li_c8 and spec.dtype == torch.float8_e4m3fn


GLM_DSA_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {
        AscendMLAAttentionSpec: _is_fp8_main_cache,
        AscendSFAIndexerCacheSpec: _is_fp8_indexer_cache,
    }
)
