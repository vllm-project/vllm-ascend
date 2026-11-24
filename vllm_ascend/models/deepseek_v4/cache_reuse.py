# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 eligibility for the common layerwise cache planner."""

from types import MappingProxyType

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec


DEEPSEEK_V4_SPEC_REUSE_RULES = MappingProxyType(
    {
        AscendMLAAttentionSpec: lambda spec: spec.model_version == "deepseek_v4",
        AscendSlidingWindowMLASpec: True,
    }
)
