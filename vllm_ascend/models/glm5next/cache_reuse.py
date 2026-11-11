# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Next spec eligibility for layerwise buffer reuse."""

from types import MappingProxyType

from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules
from vllm_ascend.models.glm5next.cache_config import _is_glm5_next_main_spec

GLM5_NEXT_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {MLAAttentionSpec: _is_glm5_next_main_spec, MambaSpec: True, AscendIndexerKPoolTailSpec: False}
)
