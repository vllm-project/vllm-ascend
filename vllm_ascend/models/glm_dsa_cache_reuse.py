# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MappingProxyType

import torch
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules


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
