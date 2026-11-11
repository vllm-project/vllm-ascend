# SPDX-License-Identifier: Apache-2.0
"""Select the retained CANN range protocol without replacing native Store."""


def uses_cann_store(config) -> bool:
    extra = config.kv_transfer_config.kv_connector_extra_config or {}
    return any(
        extra.get(key) is True
        for key in ("use_layerwise_range", "hybrid_cache_c128_chunk", "layerwise_host_kv_offload")
    )


def uses_cann_workspace(config) -> bool:
    extra = config.kv_transfer_config.kv_connector_extra_config or {}
    return extra.get("layerwise_host_kv_offload") is True
