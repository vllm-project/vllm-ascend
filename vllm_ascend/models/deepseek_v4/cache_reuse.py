# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 layerwise cache reuse."""

from types import MappingProxyType

from vllm.distributed.kv_transfer import get_kv_transfer_group, is_v1_kv_transfer_group

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    SpecReuseRules,
    get_layerwise_reuse_layout,
)

DEEPSEEK_V4_SPEC_REUSE_RULES: SpecReuseRules = MappingProxyType(
    {AscendMLAAttentionSpec: True, AscendSlidingWindowMLASpec: True}
)


def get_swa_save_layer_name(attn_layer_name: str, swa_cache_name: str) -> str:
    """Use the registered SWA cache name when it shares a rotating slot."""
    if not is_v1_kv_transfer_group():
        return attn_layer_name
    connector = get_kv_transfer_group()
    layout = get_layerwise_reuse_layout(getattr(connector, "_kv_cache_config", None))
    if layout is not None and swa_cache_name in layout.reused_cache_names:
        return swa_cache_name
    return attn_layer_name
